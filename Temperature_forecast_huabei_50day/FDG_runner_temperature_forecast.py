"""
FDG runner for out-of-window temperature field forecasting.

Train on day 1--20 using IC/BC/internal-data samples, then extrapolate the
trained model to day 21--50 and evaluate against the observed temperature
fields.

Outputs:
    - animated GIF with Pred / True / Absolute Error for days 21--50
    - PNG and CSV forecast snapshots for each forecast day
    - per-day forecast metric CSV
    - training loss CSV
    - model checkpoint
    - forecast summary text
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", os.environ.get("TEMP_SEED", "2026"))

import csv
import random
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import temperature_fdg_eq as eq


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_optional_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip().lower() in {"", "none", "auto"}:
        return default
    return float(raw)


def env_list_float(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


SEED = env_int("TEMP_SEED", 2026)
ADAM_STEPS = env_int("TEMP_ADAM_STEPS", 2000)
LBFGS_STEPS = env_int("TEMP_LBFGS_STEPS", 300)
LR_THETA = env_float("TEMP_LR_THETA", 1.0e-3)
LR_Q = env_float("TEMP_LR_Q", 2.0e-3)
BC_FRACTION = env_float("TEMP_BC_FRACTION", 0.20)
DATA_FRACTION = env_float("TEMP_DATA_FRACTION", 0.10)
HIDDEN_WIDTH = env_int("TEMP_HIDDEN_WIDTH", 128)
HIDDEN_DEPTH = env_int("TEMP_HIDDEN_DEPTH", 4)
FOURIER_FREQS = env_list_float("TEMP_FOURIER_FREQS", (1.0, 2.0, 4.0, 8.0))
DTYPE_NAME = os.environ.get("TEMP_DTYPE", "float32").lower()
DTYPE = torch.float64 if DTYPE_NAME == "float64" else torch.float32
SNAPSHOT_DAYS = tuple(int(x) for x in os.environ.get("TEMP_SNAPSHOT_DAYS", "1,6,11,16").split(","))
TEMP_VMIN = env_optional_float("TEMP_PLOT_VMIN")
TEMP_VMAX = env_optional_float("TEMP_PLOT_VMAX")
ERR_VMAX = env_optional_float("TEMP_ERR_VMAX")
TEMP_CMAP = os.environ.get("TEMP_CMAP", "viridis")
ERROR_CMAP = os.environ.get("TEMP_ERROR_CMAP", "viridis")
AXIS_LABEL_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 14
FIGURE_TITLE_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
TEMP_CMAP_OBJECT = matplotlib.colormaps.get_cmap(TEMP_CMAP).copy()
TEMP_CMAP_OBJECT.set_bad("white")
ERROR_CMAP_OBJECT = matplotlib.colormaps.get_cmap(ERROR_CMAP).copy()
ERROR_CMAP_OBJECT.set_bad("white")
FORECAST_RESULTS_DIR = Path(__file__).resolve().parent / "fdg_temperature_forecast_results"
FORECAST_DAY_START = env_int("TEMP_FORECAST_DAY_START", 21)
FORECAST_DAY_END = env_int("TEMP_FORECAST_DAY_END", eq.FORECAST_DAY_END)
FORECAST_SNAPSHOT_DAYS = tuple(
    int(x)
    for x in os.environ.get(
        "TEMP_FORECAST_SNAPSHOT_DAYS",
        ",".join(str(d) for d in range(FORECAST_DAY_START, FORECAST_DAY_END + 1)),
    ).split(",")
    if x.strip()
)


def rounded_bounds(values: np.ndarray, step: float) -> tuple[float, float]:
    lo = float(np.floor(float(np.nanmin(values)) / step) * step)
    hi = float(np.ceil(float(np.nanmax(values)) / step) * step)
    if hi <= lo:
        hi = lo + step
    return lo, hi


def rounded_upper(value: float, step: float) -> float:
    hi = float(np.ceil(float(value) / step) * step)
    return hi if hi > 0.0 else step


def ticks_for(lo: float, hi: float, step: float) -> np.ndarray:
    return np.arange(lo, hi + 0.5 * step, step)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


class FourierDoubleGateNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_width: int = 128,
        hidden_depth: int = 4,
        freqs: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    ) -> None:
        super().__init__()
        self.freqs = tuple(float(f) for f in freqs)
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
            pieces.append(torch.sin(z))
            pieces.append(torch.cos(z))
        return torch.cat(pieces, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_layer(self.encode(x)))
        for value_layer, gate_layer, skip_layer in zip(self.value_layers, self.gate_layers, self.skip_layers):
            value = self.activation(value_layer(h))
            gate = torch.sigmoid(gate_layer(h))
            skip = self.activation(skip_layer(h))
            h = gate * value + (1.0 - gate) * skip
        return self.output_layer(h)


class AdaptiveWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_ic = nn.Parameter(torch.zeros(()))
        self.q_bc = nn.Parameter(torch.zeros(()))
        self.q_data = nn.Parameter(torch.zeros(()))

    def loss(self, ic_loss: torch.Tensor, bc_loss: torch.Tensor, data_loss: torch.Tensor) -> torch.Tensor:
        return (
            torch.exp(-self.q_ic) * ic_loss
            + self.q_ic
            + torch.exp(-self.q_bc) * bc_loss
            + self.q_bc
            + torch.exp(-self.q_data) * data_loss
            + self.q_data
        )

    def weights(self) -> tuple[float, float, float]:
        with torch.no_grad():
            return (
                float(torch.exp(-self.q_ic).detach().cpu()),
                float(torch.exp(-self.q_bc).detach().cpu()),
                float(torch.exp(-self.q_data).detach().cpu()),
            )


def tensor_from_np(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=DTYPE, device=device)


def make_train_tensors(
    data: eq.TemperatureData,
    indices: dict[str, np.ndarray],
    mean: float,
    std: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in ("ic", "bc", "data"):
        x_np = eq.make_inputs(data, indices[key])
        y_np = (eq.make_targets(data, indices[key]) - mean) / std
        out[f"x_{key}"] = tensor_from_np(x_np, device)
        out[f"y_{key}"] = tensor_from_np(y_np, device)
    return out


def train_model(
    model: FourierDoubleGateNet,
    aw: AdaptiveWeights,
    tensors: dict[str, torch.Tensor],
    out_dir: Path,
) -> list[dict[str, float]]:
    mse = nn.MSELoss()
    theta_params = list(model.parameters())
    theta_optim = torch.optim.Adam(theta_params, lr=LR_THETA)
    q_optim = torch.optim.Adam(aw.parameters(), lr=LR_Q)
    history: list[dict[str, float]] = []

    def compute_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ic = mse(model(tensors["x_ic"]), tensors["y_ic"])
        bc = mse(model(tensors["x_bc"]), tensors["y_bc"])
        data = mse(model(tensors["x_data"]), tensors["y_data"])
        total = aw.loss(ic, bc, data)
        return total, ic, bc, data

    for step in range(1, ADAM_STEPS + 1):
        theta_optim.zero_grad(set_to_none=True)
        q_optim.zero_grad(set_to_none=True)
        total, ic, bc, data = compute_losses()
        total.backward()
        torch.nn.utils.clip_grad_norm_(theta_params, 100.0)
        theta_optim.step()
        q_optim.step()
        if step == 1 or step % 20 == 0 or step == ADAM_STEPS:
            w_ic, w_bc, w_data = aw.weights()
            history.append(
                {
                    "iter": float(step),
                    "stage": 0.0,
                    "total": float(total.detach().cpu()),
                    "ic": float(ic.detach().cpu()),
                    "bc": float(bc.detach().cpu()),
                    "data": float(data.detach().cpu()),
                    "w_ic": w_ic,
                    "w_bc": w_bc,
                    "w_data": w_data,
                }
            )
        if step == 1 or step % 500 == 0 or step == ADAM_STEPS:
            print(
                f"[Adam {step:5d}/{ADAM_STEPS}] total={float(total.detach().cpu()):.3e} "
                f"ic={float(ic.detach().cpu()):.3e} bc={float(bc.detach().cpu()):.3e} "
                f"data={float(data.detach().cpu()):.3e} w=({w_ic:.2f},{w_bc:.2f},{w_data:.2f})",
                flush=True,
            )

    if LBFGS_STEPS > 0:
        for p in aw.parameters():
            p.requires_grad_(False)
        lbfgs = torch.optim.LBFGS(
            theta_params,
            lr=0.5,
            max_iter=LBFGS_STEPS,
            max_eval=LBFGS_STEPS,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-12,
        )

        lbfgs_calls = {"n": 0}

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            total, _, _, _ = compute_losses()
            total.backward()
            lbfgs_calls["n"] += 1
            return total

        print(f"Stage 2: L-BFGS max_iter={LBFGS_STEPS}", flush=True)
        lbfgs.step(closure)
        total, ic, bc, data = compute_losses()
        w_ic, w_bc, w_data = aw.weights()
        history.append(
            {
                "iter": float(ADAM_STEPS + LBFGS_STEPS),
                "stage": 1.0,
                "total": float(total.detach().cpu()),
                "ic": float(ic.detach().cpu()),
                "bc": float(bc.detach().cpu()),
                "data": float(data.detach().cpu()),
                "w_ic": w_ic,
                "w_bc": w_bc,
                "w_data": w_data,
            }
        )
        print(
            f"[Final] total={float(total.detach().cpu()):.3e} "
            f"ic={float(ic.detach().cpu()):.3e} bc={float(bc.detach().cpu()):.3e} "
            f"data={float(data.detach().cpu()):.3e} w=({w_ic:.2f},{w_bc:.2f},{w_data:.2f})",
            flush=True,
        )

    with (out_dir / "training_loss.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["iter", "stage", "total", "ic", "bc", "data", "w_ic", "w_bc", "w_data"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return history


@torch.no_grad()
def predict_indices(
    model: FourierDoubleGateNet,
    data: eq.TemperatureData,
    indices: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    xs = eq.make_inputs(data, indices)
    preds: list[np.ndarray] = []
    model.eval()
    for start in range(0, xs.shape[0], batch_size):
        xb = tensor_from_np(xs[start : start + batch_size], device)
        y = model(xb).detach().cpu().numpy().astype(np.float32)
        preds.append(y)
    return np.concatenate(preds, axis=0).reshape(-1) * float(std) + float(mean)


def build_masked_grid(
    data: eq.TemperatureData,
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return eq.values_to_masked_grid(data, indices, values)


def plot_triplet(
    pred_grid: np.ndarray,
    true_grid: np.ndarray,
    day: int,
    output_png: Path,
    temp_vmin: float,
    temp_vmax: float,
    err_vmax: float,
    title_suffix: str = "",
) -> None:
    err_grid = np.abs(pred_grid - true_grid)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    temp_im = None
    for ax, title, grid in zip(axes[:2], ("Pred", "True"), (pred_grid, true_grid)):
        temp_im = ax.imshow(
            grid,
            extent=[0.0, 1.0, 0.0, 1.0],
            origin="lower",
            aspect="equal",
            cmap=TEMP_CMAP_OBJECT,
            vmin=temp_vmin,
            vmax=temp_vmax,
        )
        ax.set_title(f"{title}{title_suffix}", fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel("normalized longitude", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("normalized latitude", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    err_im = axes[2].imshow(
        err_grid,
        extent=[0.0, 1.0, 0.0, 1.0],
        origin="lower",
        aspect="equal",
        cmap=ERROR_CMAP_OBJECT,
        vmin=0.0,
        vmax=err_vmax,
    )
    axes[2].set_title(f"Error{title_suffix}", fontsize=PANEL_TITLE_FONTSIZE)
    axes[2].set_xlabel("normalized longitude", fontsize=AXIS_LABEL_FONTSIZE)
    axes[2].set_ylabel("normalized latitude", fontsize=AXIS_LABEL_FONTSIZE)
    axes[2].set_box_aspect(1)
    axes[2].tick_params(labelsize=TICK_LABEL_FONTSIZE)

    temp_cbar = fig.colorbar(temp_im, ax=axes[:2], fraction=0.046, pad=0.04, ticks=ticks_for(temp_vmin, temp_vmax, 5.0))
    temp_cbar.set_label("Temperature ($^\\circ$C)", fontsize=AXIS_LABEL_FONTSIZE)
    temp_cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    err_cbar = fig.colorbar(err_im, ax=axes[2], fraction=0.046, pad=0.04, ticks=ticks_for(0.0, err_vmax, 2.0))
    err_cbar.set_label("Absolute error ($^\\circ$C)", fontsize=AXIS_LABEL_FONTSIZE)
    err_cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    fig.suptitle(f"Temperature field forecast, day {day}", fontsize=FIGURE_TITLE_FONTSIZE)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def save_day_csv(
    output_csv: Path,
    data: eq.TemperatureData,
    indices: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    day: int,
) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["day_index", "lat", "lon", "true_temp_C", "pred_temp_C", "abs_error_C"])
        for idx, p, y in zip(indices, pred, true):
            writer.writerow([day, f"{data.lat[idx]:.10f}", f"{data.lon[idx]:.10f}", f"{y:.10f}", f"{p:.10f}", f"{abs(p-y):.10f}"])


def metric_row(day: int, indices: np.ndarray, pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    mse = float(np.mean(err**2))
    return {
        "day": float(day),
        "points": float(indices.size),
        "mse_C2": mse,
        "rmse_C": float(np.sqrt(mse)),
        "mae_C": float(np.mean(np.abs(err))),
        "relative_l2": float(np.linalg.norm(err) / np.linalg.norm(true)),
    }


def save_metric_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["day", "points", "mse_C2", "rmse_C", "mae_C", "relative_l2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all_days(
    model: FourierDoubleGateNet,
    data: eq.TemperatureData,
    mean: float,
    std: float,
    device: torch.device,
    future_observed_idx: np.ndarray | None = None,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    observed_set = set() if future_observed_idx is None else {int(i) for i in future_observed_idx.tolist()}
    full_rows: list[dict[str, float]] = []
    unobserved_rows: list[dict[str, float]] = []

    for day in range(int(data.unique_days.min()), int(data.unique_days.max()) + 1):
        indices = eq.day_indices(data, day)
        if indices.size == 0:
            continue
        pred = predict_indices(model, data, indices, mean, std, device)
        true = data.temp_c[indices].astype(np.float64)
        full_rows.append(metric_row(day, indices, pred, true))
        if day >= FORECAST_DAY_START:
            keep_mask = np.array([int(i) not in observed_set for i in indices], dtype=bool)
            kept = indices[keep_mask]
            if kept.size:
                unobserved_rows.append(metric_row(day, kept, pred[keep_mask], true[keep_mask]))
            else:
                unobserved_rows.append(
                    {
                        "day": float(day),
                        "points": 0.0,
                        "mse_C2": float("nan"),
                        "rmse_C": float("nan"),
                        "mae_C": float("nan"),
                        "relative_l2": float("nan"),
                    }
                )
    return full_rows, unobserved_rows


def evaluate_and_plot(
    model: FourierDoubleGateNet,
    data: eq.TemperatureData,
    out_dir: Path,
    mean: float,
    std: float,
    device: torch.device,
) -> dict[str, float]:
    frames: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    day_results: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    per_day_metrics: list[dict[str, float]] = []

    for day in range(FORECAST_DAY_START, FORECAST_DAY_END + 1):
        indices = eq.day_indices(data, day)
        if indices.size == 0:
            print(f"[warn] day {day} has no rows; skipping", flush=True)
            continue
        pred = predict_indices(model, data, indices, mean, std, device)
        true = data.temp_c[indices].astype(np.float32)
        err = pred - true
        all_pred.append(pred)
        all_true.append(true)
        day_results.append((day, indices, pred, true))
        per_day_metrics.append(metric_row(day, indices, pred, true))

    if not all_pred:
        raise RuntimeError(f"No forecast rows found for day {FORECAST_DAY_START}--{FORECAST_DAY_END}")

    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)
    err_all = pred_all - true_all
    auto_temp_vmin, auto_temp_vmax = rounded_bounds(np.concatenate([pred_all, true_all]), 5.0)
    temp_vmin = TEMP_VMIN if TEMP_VMIN is not None else auto_temp_vmin
    temp_vmax = TEMP_VMAX if TEMP_VMAX is not None else auto_temp_vmax
    err_vmax = ERR_VMAX if ERR_VMAX is not None else rounded_upper(float(np.max(np.abs(err_all))), 2.0)

    for day, indices, pred, true in day_results:
        pred_grid = build_masked_grid(data, indices, pred)
        true_grid = build_masked_grid(data, indices, true)
        frame_png = out_dir / f"forecast_frame_day_{day:02d}.png"
        plot_triplet(pred_grid, true_grid, day, frame_png, temp_vmin, temp_vmax, err_vmax)
        frames.append(imageio.imread(frame_png))

        if day in FORECAST_SNAPSHOT_DAYS or day in SNAPSHOT_DAYS:
            plot_triplet(pred_grid, true_grid, day, out_dir / f"forecast_snapshot_day_{day:02d}.png", temp_vmin, temp_vmax, err_vmax)
            save_day_csv(out_dir / f"forecast_snapshot_day_{day:02d}.csv", data, indices, pred, true, day)

    gif_name = f"temperature_forecast_day{FORECAST_DAY_START:02d}_{FORECAST_DAY_END:02d}.gif"
    imageio.mimsave(out_dir / gif_name, frames, duration=0.65)
    with (out_dir / "forecast_metrics_by_day.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["day", "points", "mse_C2", "rmse_C", "mae_C", "relative_l2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_day_metrics)

    mse = float(np.mean(err_all**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err_all)))
    rel_l2 = float(np.linalg.norm(err_all) / np.linalg.norm(true_all))
    return {
        "forecast_mse_C2": mse,
        "forecast_rmse_C": rmse,
        "forecast_mae_C": mae,
        "forecast_relative_l2": rel_l2,
        "plot_temp_vmin_C": temp_vmin,
        "plot_temp_vmax_C": temp_vmax,
        "plot_error_vmax_C": err_vmax,
    }


def main() -> None:
    set_deterministic(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FORECAST_RESULTS_DIR / f"forecast_fdg_aw_landmask_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"dtype={DTYPE}, seed={SEED}")
    print(
        f"Training: day {eq.TRAIN_DAY_START}-{eq.TRAIN_DAY_END} with IC=all day-{eq.TRAIN_DAY_START} points, "
        f"BC={BC_FRACTION:.2f} per-day boundary points, DATA={DATA_FRACTION:.2f} internal points"
    )
    print(f"Forecast evaluation: day {FORECAST_DAY_START}-{FORECAST_DAY_END}")
    print(f"Output directory: {out_dir}")

    data = eq.load_temperature_data()
    print(
        f"Spatial mask: valid={int(data.valid_spatial_mask.sum())}, "
        f"sea={int((~data.valid_spatial_mask).sum())}, "
        f"four-neighbor boundary={int(data.boundary_spatial_mask.sum())}"
    )
    rng = np.random.default_rng(SEED)
    indices = eq.select_training_indices(data, rng, ic_all=True, bc_fraction=BC_FRACTION, data_fraction=DATA_FRACTION)
    np.savez_compressed(
        out_dir / "train_indices.npz",
        ic=indices["ic"],
        bc=indices["bc"],
        data=indices["data"],
        train_all=indices["train_all"],
        seed=np.asarray([SEED], dtype=np.int64),
        bc_fraction=np.asarray([BC_FRACTION], dtype=np.float32),
        data_fraction=np.asarray([DATA_FRACTION], dtype=np.float32),
    )
    train_all = indices["train_all"]
    temp_train = data.temp_c[train_all]
    mean = float(temp_train.mean())
    std = float(temp_train.std() if temp_train.std() > 0 else 1.0)

    print(f"Loaded rows={data.temp_c.size}, train rows day {eq.TRAIN_DAY_START}-{eq.TRAIN_DAY_END}={train_all.size}")
    print(f"Selected IC={indices['ic'].size}, BC={indices['bc'].size}, DATA={indices['data'].size}")
    print(f"Temperature normalization: mean={mean:.6f} C, std={std:.6f} C")

    tensors = make_train_tensors(data, indices, mean, std, device)
    model = FourierDoubleGateNet(
        input_dim=eq.INPUT_DIM,
        output_dim=eq.OUTPUT_DIM,
        hidden_width=HIDDEN_WIDTH,
        hidden_depth=HIDDEN_DEPTH,
        freqs=FOURIER_FREQS,
    ).to(device=device, dtype=DTYPE)
    aw = AdaptiveWeights().to(device=device, dtype=DTYPE)

    train_model(model, aw, tensors, out_dir)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "aw_state_dict": aw.state_dict(),
            "seed": SEED,
            "dtype": str(DTYPE),
            "hidden_width": HIDDEN_WIDTH,
            "hidden_depth": HIDDEN_DEPTH,
            "fourier_freqs": FOURIER_FREQS,
            "temp_mean_C": mean,
            "temp_std_C": std,
            "bc_fraction": BC_FRACTION,
            "data_fraction": DATA_FRACTION,
            "adam_steps": ADAM_STEPS,
            "lbfgs_steps": LBFGS_STEPS,
            "boundary_mode": "four_neighbor_land_mask",
        },
        out_dir / "fdg_temperature_forecast_model.pt",
    )
    metrics = evaluate_and_plot(model, data, out_dir, mean, std, device)
    full_rows, unobserved_rows = evaluate_all_days(model, data, mean, std, device)
    save_metric_rows(out_dir / "metrics_day01_50.csv", full_rows)
    save_metric_rows(out_dir / "forecast_unobserved_metrics_day21_50.csv", unobserved_rows)

    with (out_dir / "forecast_summary.txt").open("w", encoding="utf-8") as f:
        f.write("FDG temperature forecast summary\n")
        f.write(f"cublas_workspace_config={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}\n")
        f.write(f"pythonhashseed={os.environ.get('PYTHONHASHSEED')}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"dtype={DTYPE}\n")
        f.write(f"adam_steps={ADAM_STEPS}\n")
        f.write(f"lbfgs_steps={LBFGS_STEPS}\n")
        f.write(f"hidden_width={HIDDEN_WIDTH}\n")
        f.write(f"hidden_depth={HIDDEN_DEPTH}\n")
        f.write(f"fourier_freqs={FOURIER_FREQS}\n")
        f.write("boundary_mode=four_neighbor_land_mask\n")
        f.write(f"valid_spatial_count={int(data.valid_spatial_mask.sum())}\n")
        f.write(f"sea_spatial_count={int((~data.valid_spatial_mask).sum())}\n")
        f.write(f"boundary_spatial_count={int(data.boundary_spatial_mask.sum())}\n")
        f.write(f"train_day_start={eq.TRAIN_DAY_START}\n")
        f.write(f"train_day_end={eq.TRAIN_DAY_END}\n")
        f.write(f"forecast_day_start={FORECAST_DAY_START}\n")
        f.write(f"forecast_day_end={FORECAST_DAY_END}\n")
        f.write(f"ic_count={indices['ic'].size}\n")
        f.write(f"bc_count={indices['bc'].size}\n")
        f.write(f"data_count={indices['data'].size}\n")
        f.write(f"train_indices_file=train_indices.npz\n")
        f.write(f"model_checkpoint=fdg_temperature_forecast_model.pt\n")
        f.write(f"temp_cmap={TEMP_CMAP}\n")
        f.write(f"error_cmap={ERROR_CMAP}\n")
        f.write(f"plot_temp_vmin_override_C={TEMP_VMIN}\n")
        f.write(f"plot_temp_vmax_override_C={TEMP_VMAX}\n")
        f.write(f"plot_error_vmax_override_C={ERR_VMAX}\n")
        f.write(f"temp_mean_C={mean}\n")
        f.write(f"temp_std_C={std}\n")
        for key, value in metrics.items():
            f.write(f"{key}={value}\n")

    print(f"Final forecast metrics day {FORECAST_DAY_START}-{FORECAST_DAY_END}:")
    print(f"  MSE(C^2)    = {metrics['forecast_mse_C2']:.4f}")
    print(f"  RMSE(C)     = {metrics['forecast_rmse_C']:.4f}")
    print(f"  MAE(C)      = {metrics['forecast_mae_C']:.4f}")
    print(f"  Relative L2 = {metrics['forecast_relative_l2']:.4e}")
    print(f"GIF saved to: {out_dir / f'temperature_forecast_day{FORECAST_DAY_START:02d}_{FORECAST_DAY_END:02d}.gif'}")


if __name__ == "__main__":
    main()
