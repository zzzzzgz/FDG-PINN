from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "FDG_CODE" / "Figures"

DATASETS = {
    "India": {
        "forecast_dir": BASE_DIR / "Temperature_forecast_india_50day" / "fdg_temperature_forecast_results",
        "recon_dir": BASE_DIR / "Temperature_forecast_india_50day" / "fdg_temperature_results",
        "recon_prefix": "reconstruction_fdg_aw_",
        "data_csv": BASE_DIR / "Temperature_forecast_india_50day" / "FDG_PINN_India_50DAY_Temperature_preprocessed.csv",
    },
    "North China": {
        "forecast_dir": BASE_DIR / "Temperature_forecast_huabei_50day" / "fdg_temperature_forecast_results",
        "recon_dir": BASE_DIR / "Temperature_forecast_huabei_50day" / "fdg_temperature_results",
        "recon_prefix": "reconstruction_fdg_aw_landmask_",
        "data_csv": BASE_DIR / "Temperature_forecast_huabei_50day" / "FDG_PINN_HUABEI_50DAY_Temperature_preprocessed.csv",
    },
}

VARIANTS = {
    "pure": {"prefix": "forecast_fdg_aw", "label": "Pure extrapolation", "color": "#1f77b4", "ls": "-"},
    "bc": {"prefix": "forecast_bc_fdg_aw", "label": "Boundary-assisted", "color": "#d55e00", "ls": "--"},
    "data": {"prefix": "forecast_data_fdg_aw", "label": "Interior-data-assisted", "color": "#2ca02c", "ls": "-."},
}

RECON_STYLE = {"label": "Reconstruction", "color": "#4d4d4d", "ls": "-", "lw": 2.0}

DATASET_STYLE = {
    "Indian subcontinent": {"marker": "o"},
    "North China": {"marker": "^"},
}

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["ps.fonttype"] = 42


class FourierDoubleGateNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_depth: int,
        freqs: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.freqs = tuple(float(freq) for freq in freqs)
        encoded_dim = input_dim * (1 + 2 * len(self.freqs))
        self.input_layer = nn.Linear(encoded_dim, hidden_width)
        self.value_layers = nn.ModuleList(nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth))
        self.gate_layers = nn.ModuleList(nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth))
        self.skip_layers = nn.ModuleList(nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth))
        self.output_layer = nn.Linear(hidden_width, output_dim)
        self.activation = nn.Tanh()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pieces = [x]
        for freq in self.freqs:
            z = 2.0 * np.pi * freq * x
            pieces.extend((torch.sin(z), torch.cos(z)))
        return torch.cat(pieces, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_layer(self.encode(x)))
        for value_layer, gate_layer, skip_layer in zip(self.value_layers, self.gate_layers, self.skip_layers):
            value = self.activation(value_layer(h))
            gate = torch.sigmoid(gate_layer(h))
            skip = self.activation(skip_layer(h))
            h = gate * value + (1.0 - gate) * skip
        return self.output_layer(h)


def latest_run(base: Path, prefix: str) -> Path:
    matches = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not matches:
        raise FileNotFoundError(f"No run directory found in {base} with prefix {prefix}")
    return matches[-1]


def load_metric_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    days: list[float] = []
    values: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(float(row["day"]))
            values.append(float(row["relative_l2"]))
    return np.asarray(days), np.asarray(values)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_dtype(checkpoint: dict) -> torch.dtype:
    return torch.float64 if "float64" in str(checkpoint.get("dtype", "float32")) else torch.float32


def load_generic_temperature_csv(csv_path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8")))
    return {
        "day": np.asarray([int(float(r["day_index"])) for r in rows], dtype=np.int64),
        "lat": np.asarray([float(r["lat"]) for r in rows], dtype=np.float32),
        "lon": np.asarray([float(r["lon"]) for r in rows], dtype=np.float32),
        "temp_c": np.asarray([float(r["temp_C"]) for r in rows], dtype=np.float32),
    }


@torch.no_grad()
def predict_generic(
    model: nn.Module,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 65536,
) -> np.ndarray:
    t = (data["day"][indices].astype(np.float32) - 1.0) / 19.0
    inputs = np.stack([data["lon"][indices], data["lat"][indices], t], axis=1).astype(np.float32)
    preds: list[np.ndarray] = []
    model.eval()
    for start in range(0, inputs.shape[0], batch_size):
        x = torch.as_tensor(inputs[start : start + batch_size], dtype=dtype, device=device)
        y = model(x).squeeze(1).detach().cpu().numpy()
        preds.append(y)
    return np.concatenate(preds).astype(np.float64) * std + mean


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def reconstruction_daily_series(dataset_name: str) -> tuple[np.ndarray, np.ndarray, Path]:
    cfg = DATASETS[dataset_name]
    run_dir = latest_run(cfg["recon_dir"], cfg["recon_prefix"])
    checkpoint = load_checkpoint(run_dir / "fdg_temperature_model.pt", torch.device("cpu"))
    dtype = checkpoint_dtype(checkpoint)
    model = FourierDoubleGateNet(
        input_dim=3,
        output_dim=1,
        hidden_width=int(checkpoint["hidden_width"]),
        hidden_depth=int(checkpoint["hidden_depth"]),
        freqs=tuple(checkpoint["fourier_freqs"]),
    ).to(device="cpu", dtype=dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    data = load_generic_temperature_csv(cfg["data_csv"])

    days: list[float] = []
    values: list[float] = []
    for day in range(1, 21):
        indices = np.flatnonzero(data["day"] == day)
        pred = predict_generic(
            model,
            data,
            indices,
            float(checkpoint["temp_mean_C"]),
            float(checkpoint["temp_std_C"]),
            torch.device("cpu"),
            dtype,
        )
        true = data["temp_c"][indices].astype(np.float64)
        days.append(float(day))
        values.append(relative_l2(pred, true))
    return np.asarray(days), np.asarray(values), run_dir


def collect_series() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, Path]]:
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, Path]] = {}
    for dataset_name, cfg in DATASETS.items():
        for variant_name, meta in VARIANTS.items():
            run_dir = latest_run(cfg["forecast_dir"], meta["prefix"])
            days, values = load_metric_csv(run_dir / "metrics_day01_50.csv")
            out[(dataset_name, variant_name)] = (days, values, run_dir)
    return out


def collect_reconstruction_series() -> dict[str, tuple[np.ndarray, np.ndarray, Path]]:
    return {dataset_name: reconstruction_daily_series(dataset_name) for dataset_name in DATASETS}


def plot_window(
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, Path]],
    recon_series: dict[str, tuple[np.ndarray, np.ndarray, Path]],
    day_min: int,
    day_max: int,
    stem: str,
    *,
    log_y: bool = False,
    use_reconstruction_prefix: bool = False,
    legend_loc: str = "best",
    legend_ncol: int = 2,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    if use_reconstruction_prefix:
        for dataset_name in ("India", "North China"):
            recon_days, recon_values, _ = recon_series[dataset_name]
            recon_mask = (recon_days >= day_min) & (recon_days <= min(day_max, 20))
            display_name = "Indian subcontinent" if dataset_name == "India" else "North China"
            dataset_style = DATASET_STYLE[display_name]
            ax.plot(
                recon_days[recon_mask],
                recon_values[recon_mask],
                label=f"{display_name} - {RECON_STYLE['label']}",
                color=RECON_STYLE["color"],
                linestyle=RECON_STYLE["ls"],
                marker=dataset_style["marker"],
                markersize=4.0,
                linewidth=RECON_STYLE["lw"],
                zorder=3,
            )

    for dataset_name in ("India", "North China"):
        for variant_name in ("pure", "bc", "data"):
            days, values, _ = series[(dataset_name, variant_name)]
            display_name = "Indian subcontinent" if dataset_name == "India" else "North China"
            if use_reconstruction_prefix:
                post_mask = (days >= max(day_min, 20)) & (days <= day_max)
                plot_days = days[post_mask]
                plot_values = values[post_mask]
                label = f"{display_name} - {VARIANTS[variant_name]['label']}"
            else:
                mask = (days >= day_min) & (days <= day_max)
                plot_days = days[mask]
                plot_values = values[mask]
                label = f"{display_name} - {VARIANTS[variant_name]['label']}"
            meta = VARIANTS[variant_name]
            dataset_style = DATASET_STYLE[display_name]
            ax.plot(
                plot_days,
                plot_values,
                label=label,
                color=meta["color"],
                linestyle=meta["ls"],
                marker=dataset_style["marker"],
                markersize=4.2,
                linewidth=1.9,
            )

    ax.set_xlim(day_min, day_max)
    ax.set_xticks(np.arange(day_min, day_max + 1, 2 if day_max - day_min > 20 else 1))
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Relative discrete $L_2$ error", fontsize=14)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(1.0e-3, 1.0e0)
    if use_reconstruction_prefix:
        ax.axvline(20.0, color="#888888", linestyle="--", linewidth=1.6, zorder=1)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, ncol=legend_ncol, loc=legend_loc)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.svg", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.eps", dpi=300)
    plt.close(fig)


def main() -> None:
    series = collect_series()
    recon_series = collect_reconstruction_series()
    plot_window(
        series,
        recon_series,
        1,
        50,
        "section53_l2_vs_day_01_50_logy",
        log_y=True,
        use_reconstruction_prefix=True,
        legend_loc="upper left",
        legend_ncol=1,
    )
    plot_window(
        series,
        recon_series,
        20,
        50,
        "section53_l2_vs_day_20_50_logy",
        log_y=True,
        use_reconstruction_prefix=False,
        legend_loc="upper left",
        legend_ncol=1,
    )

    summary_path = FIG_DIR / "section53_l2_curve_sources.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for (dataset_name, variant_name), (_, _, run_dir) in series.items():
            f.write(f"{dataset_name}\t{variant_name}\t{run_dir}\n")
        for dataset_name, (_, _, run_dir) in recon_series.items():
            f.write(f"{dataset_name}\treconstruction\t{run_dir}\n")


if __name__ == "__main__":
    main()
