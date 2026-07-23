"""Plot daily reconstruction statistics for Standard PINN and FDG-PINN."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import temperature_fdg_eq as eq


# -----------------------------------------------------------------------------
# Replace these paths to compare another dataset or another pair of checkpoints.
# The PINN and FDG checkpoints must have been trained for the selected CSV.
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "FDG_PINN_HUABEI_50DAY_Temperature_preprocessed.csv"
PINN_CHECKPOINT = (
    BASE_DIR
    / "pinn_temperature_results"
    / "reconstruction_pinn_fixed_landmask_20260629_121648"
    / "pinn_temperature_model.pt"
)
FDG_CHECKPOINT = (
    BASE_DIR
    / "fdg_temperature_results"
    / "reconstruction_fdg_aw_landmask_20260629_121501"
    / "fdg_temperature_model.pt"
)
OUTPUT_DIR = BASE_DIR / "temperature_model_comparison_results_landmask"
DAY_START = 1
DAY_END = 20
BATCH_SIZE = 65536
CURVE_FIGSIZE = (6.4, 4.6)
CURVE_LABEL_FONTSIZE = 16
CURVE_TICK_FONTSIZE = 13
CURVE_LEGEND_FONTSIZE = 12
MEAN_LEGEND_LOC = "lower right"


class PlainPINN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_width: int, hidden_depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_width), nn.Tanh()]
        for _ in range(max(0, hidden_depth - 1)):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_dtype(checkpoint: dict) -> torch.dtype:
    return torch.float64 if "float64" in str(checkpoint.get("dtype", "float32")) else torch.float32


@torch.no_grad()
def predict(
    model: nn.Module,
    data: eq.TemperatureData,
    indices: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    inputs = eq.make_inputs(data, indices)
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, inputs.shape[0], BATCH_SIZE):
        x = torch.as_tensor(inputs[start : start + BATCH_SIZE], dtype=dtype, device=device)
        pred = model(x).squeeze(1).detach().cpu().numpy()
        chunks.append(pred)
    return np.concatenate(chunks).astype(np.float64) * std + mean


def relative_discrete_l2(pred: np.ndarray, true: np.ndarray) -> float:
    denominator = float(np.linalg.norm(true))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(pred - true) / denominator)


def save_metrics(rows: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_axes(ax: plt.Axes) -> None:
    ax.set_xlim(DAY_START, DAY_END)
    ax.set_xticks(np.arange(DAY_START, DAY_END + 1))
    ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_daily_means(rows: list[dict[str, float]], output_path: Path) -> None:
    days = np.asarray([row["day"] for row in rows])
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE, constrained_layout=True)
    ax.plot(days, [row["true_mean_C"] for row in rows], color="#202124", marker="o", linewidth=2.2, label="True")
    ax.plot(days, [row["pinn_mean_C"] for row in rows], color="#d55e00", marker="s", linewidth=1.9, label="Standard PINN")
    ax.plot(days, [row["fdg_mean_C"] for row in rows], color="#0072b2", marker="^", linewidth=1.9, label="FDG-PINN")
    ax.set_xlabel("Day", fontsize=CURVE_LABEL_FONTSIZE)
    ax.set_ylabel("Spatial mean temperature ($^\\circ$C)", fontsize=CURVE_LABEL_FONTSIZE)
    configure_axes(ax)
    ax.legend(frameon=False, ncol=1, loc=MEAN_LEGEND_LOC, fontsize=CURVE_LEGEND_FONTSIZE)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".svg"))
    fig.savefig(output_path.with_suffix(".eps"))
    plt.close(fig)


def plot_daily_l2(rows: list[dict[str, float]], output_path: Path) -> None:
    days = np.asarray([row["day"] for row in rows])
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE, constrained_layout=True)
    ax.plot(days, [row["pinn_relative_l2"] for row in rows], color="#d55e00", marker="s", linewidth=1.9, label="Standard PINN")
    ax.plot(days, [row["fdg_relative_l2"] for row in rows], color="#0072b2", marker="^", linewidth=1.9, label="FDG-PINN")
    ax.set_xlabel("Day", fontsize=CURVE_LABEL_FONTSIZE)
    ax.set_ylabel("Relative discrete $L_2$ error", fontsize=CURVE_LABEL_FONTSIZE)
    configure_axes(ax)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=CURVE_LEGEND_FONTSIZE)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".svg"))
    fig.savefig(output_path.with_suffix(".eps"))
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = eq.load_temperature_data(DATA_CSV)

    pinn_checkpoint = load_checkpoint(PINN_CHECKPOINT, device)
    fdg_checkpoint = load_checkpoint(FDG_CHECKPOINT, device)
    pinn_dtype = checkpoint_dtype(pinn_checkpoint)
    fdg_dtype = checkpoint_dtype(fdg_checkpoint)

    pinn = PlainPINN(
        input_dim=eq.INPUT_DIM,
        output_dim=eq.OUTPUT_DIM,
        hidden_width=int(pinn_checkpoint["hidden_width"]),
        hidden_depth=int(pinn_checkpoint["hidden_depth"]),
    ).to(device=device, dtype=pinn_dtype)
    pinn.load_state_dict(pinn_checkpoint["model_state_dict"])

    fdg = FourierDoubleGateNet(
        input_dim=eq.INPUT_DIM,
        output_dim=eq.OUTPUT_DIM,
        hidden_width=int(fdg_checkpoint["hidden_width"]),
        hidden_depth=int(fdg_checkpoint["hidden_depth"]),
        freqs=tuple(fdg_checkpoint["fourier_freqs"]),
    ).to(device=device, dtype=fdg_dtype)
    fdg.load_state_dict(fdg_checkpoint["model_state_dict"])

    rows: list[dict[str, float]] = []
    for day in range(DAY_START, DAY_END + 1):
        indices = eq.day_indices(data, day)
        if indices.size == 0:
            raise ValueError(f"No data rows found for day {day}")
        true = data.temp_c[indices].astype(np.float64)
        pinn_pred = predict(
            pinn,
            data,
            indices,
            float(pinn_checkpoint["temp_mean_C"]),
            float(pinn_checkpoint["temp_std_C"]),
            device,
            pinn_dtype,
        )
        fdg_pred = predict(
            fdg,
            data,
            indices,
            float(fdg_checkpoint["temp_mean_C"]),
            float(fdg_checkpoint["temp_std_C"]),
            device,
            fdg_dtype,
        )
        rows.append(
            {
                "day": float(day),
                "true_mean_C": float(np.mean(true)),
                "pinn_mean_C": float(np.mean(pinn_pred)),
                "fdg_mean_C": float(np.mean(fdg_pred)),
                "pinn_relative_l2": relative_discrete_l2(pinn_pred, true),
                "fdg_relative_l2": relative_discrete_l2(fdg_pred, true),
            }
        )

    save_metrics(rows, OUTPUT_DIR / "daily_reconstruction_comparison.csv")
    plot_daily_means(rows, OUTPUT_DIR / "mean_temperature_vs_day.png")
    plot_daily_l2(rows, OUTPUT_DIR / "relative_l2_vs_day.png")
    print(f"Saved comparison outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
