"""
FDG runner for GEE North China temperature field reconstruction.

Experiment 1 in this file:
    Train on day 1--20 using IC/BC/internal-data samples, then reconstruct
    the full temperature field for each day 1--20.

Outputs:
    - animated GIF with Pred / True / Absolute Error for days 1--20
    - PNG and CSV field snapshots for days 1, 6, 11, 16
    - training loss CSV
    - run summary text
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
# Hyperparameters can be overridden through environment variables. This makes
# the archived script reproducible by default while still allowing controlled
# sensitivity tests without editing the source file.
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
SNAPSHOT_DAYS = tuple(int(x) for x in os.environ.get("TEMP_SNAPSHOT_DAYS", "5,10,20").split(","))
COMPOSITE_DAYS = tuple(int(x) for x in os.environ.get("TEMP_COMPOSITE_DAYS", "5,10,20").split(","))
TEMP_VMIN = env_optional_float("TEMP_PLOT_VMIN")
TEMP_VMAX = env_optional_float("TEMP_PLOT_VMAX")
ERR_VMAX = env_optional_float("TEMP_ERR_VMAX")
TEMP_CMAP = os.environ.get("TEMP_CMAP", "viridis")
ERROR_CMAP = os.environ.get("TEMP_ERROR_CMAP", "viridis")
AXIS_LABEL_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 14
FIGURE_TITLE_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["ps.fonttype"] = 42
TEMP_CMAP_OBJECT = matplotlib.colormaps.get_cmap(TEMP_CMAP).copy()
TEMP_CMAP_OBJECT.set_bad("white")
ERROR_CMAP_OBJECT = matplotlib.colormaps.get_cmap(ERROR_CMAP).copy()
ERROR_CMAP_OBJECT.set_bad("white")


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
    # Data-constrained FDG backbone for temperature fields. The input is
    # normalized (longitude, latitude, time); deterministic Fourier features
    # enrich these coordinates before gated hidden layers select and recombine
    # the resulting basis functions.
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
        # Concatenate raw normalized coordinates with sin/cos features at the
        # prescribed frequencies. This mirrors the PDE experiments but without
        # an explicit atmospheric PDE residual.
        pieces = [x]
        for freq in self.freqs:
            z = 2.0 * np.pi * freq * x
            pieces.append(torch.sin(z))
            pieces.append(torch.cos(z))
        return torch.cat(pieces, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_layer(self.encode(x)))
        for value_layer, gate_layer, skip_layer in zip(self.value_layers, self.gate_layers, self.skip_layers):
            # Each layer blends a nonlinear value branch with a skip-like branch.
            # The sigmoid gate controls which representation is trusted at each
            # spatial-temporal sample.
            value = self.activation(value_layer(h))
            gate = torch.sigmoid(gate_layer(h))
            skip = self.activation(skip_layer(h))
            h = gate * value + (1.0 - gate) * skip
        return self.output_layer(h)


class AdaptiveWeights(nn.Module):
    # Learnable weights for the three data constraints: initial field (IC),
    # boundary observations (BC), and sparse interior observations (DATA).
    # This is the data-only analogue of adaptive PDE/IC/BC weighting.
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
    # Build normalized training tensors from the sampled row indices. Targets
    # are standardized with statistics computed only from the training window.
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
    # Two-stage optimization: Adam jointly updates the network and adaptive
    # weights, then L-BFGS refines only the network with the learned weights
    # frozen. The history saved here supports the reported loss diagnostics.
    mse = nn.MSELoss()
    theta_params = list(model.parameters())
    theta_optim = torch.optim.Adam(theta_params, lr=LR_THETA)
    q_optim = torch.optim.Adam(aw.parameters(), lr=LR_Q)
    history: list[dict[str, float]] = []

    def compute_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # IC uses all valid land cells on the first training day, BC uses sampled
        # four-neighbor land-boundary cells on each day, and DATA uses sparse
        # non-boundary interior observations.
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
    # Batched inference over arbitrary row indices. Predictions are converted
    # back from standardized units to degrees Celsius before evaluation/plotting.
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
    # Convert vector predictions back to the valid land grid; sea cells remain
    # masked and are therefore rendered as white and excluded from metrics.
    return eq.values_to_masked_grid(data, indices, values)


def plot_triplet(
    pred_grid: np.ndarray,
    true_grid: np.ndarray,
    day: int,
    output_path: Path,
    temp_vmin: float,
    temp_vmax: float,
    err_vmax: float,
    title_suffix: str = "",
    save_vector: bool = False,
) -> None:
    # Save one Pred/True/Absolute-Error panel for a single day. The same color
    # limits are reused across days to make temporal comparisons meaningful.
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
    fig.suptitle(f"Temperature field reconstruction, day {day}", fontsize=FIGURE_TITLE_FONTSIZE)
    fig.savefig(output_path, dpi=180)
    if save_vector:
        fig.savefig(output_path.with_suffix(".svg"), dpi=300)
        fig.savefig(output_path.with_suffix(".eps"), dpi=300)
    plt.close(fig)


def plot_composite_triplets(
    day_grids: dict[int, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    temp_vmin: float,
    temp_vmax: float,
    err_vmax: float,
) -> None:
    # Compact paper figure: stack selected days vertically, each with Pred,
    # True, and Error panels.
    missing_days = [day for day in COMPOSITE_DAYS if day not in day_grids]
    if missing_days:
        raise ValueError(f"Composite days are not available: {missing_days}")

    fig = plt.figure(figsize=(12.5, 3.6 * len(COMPOSITE_DAYS)), constrained_layout=True)
    subfigures = fig.subfigures(nrows=len(COMPOSITE_DAYS), ncols=1)
    if len(COMPOSITE_DAYS) == 1:
        subfigures = [subfigures]

    for row, day in enumerate(COMPOSITE_DAYS):
        pred_grid, true_grid = day_grids[day]
        err_grid = np.abs(pred_grid - true_grid)
        subfig = subfigures[row]
        subfig.suptitle(f"Temperature field reconstruction, day {day}", fontsize=FIGURE_TITLE_FONTSIZE)
        pred_ax, true_ax, err_ax = subfig.subplots(1, 3)

        temp_im = None
        for ax, title, grid in ((pred_ax, "Pred", pred_grid), (true_ax, "True", true_grid)):
            temp_im = ax.imshow(
                grid,
                extent=[0.0, 1.0, 0.0, 1.0],
                origin="lower",
                aspect="equal",
                cmap=TEMP_CMAP_OBJECT,
                vmin=temp_vmin,
                vmax=temp_vmax,
            )
            ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
            ax.set_xlabel("normalized longitude", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("normalized latitude", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_box_aspect(1)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        err_im = err_ax.imshow(
            err_grid,
            extent=[0.0, 1.0, 0.0, 1.0],
            origin="lower",
            aspect="equal",
            cmap=ERROR_CMAP_OBJECT,
            vmin=0.0,
            vmax=err_vmax,
        )
        err_ax.set_title("Error", fontsize=PANEL_TITLE_FONTSIZE)
        err_ax.set_xlabel("normalized longitude", fontsize=AXIS_LABEL_FONTSIZE)
        err_ax.set_ylabel("normalized latitude", fontsize=AXIS_LABEL_FONTSIZE)
        err_ax.set_box_aspect(1)
        err_ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        temp_cbar = subfig.colorbar(
            temp_im,
            ax=(pred_ax, true_ax),
            fraction=0.046,
            pad=0.025,
            ticks=ticks_for(temp_vmin, temp_vmax, 5.0),
        )
        temp_cbar.set_label("Temperature ($^\\circ$C)", fontsize=AXIS_LABEL_FONTSIZE)
        temp_cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        err_cbar = subfig.colorbar(
            err_im,
            ax=err_ax,
            fraction=0.046,
            pad=0.025,
            ticks=ticks_for(0.0, err_vmax, 2.0),
        )
        err_cbar.set_label("Absolute error ($^\\circ$C)", fontsize=AXIS_LABEL_FONTSIZE)
        err_cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    stem = "temperature_reconstruction_days_" + "_".join(f"{day:02d}" for day in COMPOSITE_DAYS)
    fig.savefig(output_dir / f"{stem}.png", dpi=220)
    fig.savefig(output_dir / f"{stem}.svg", dpi=300)
    fig.savefig(output_dir / f"{stem}.eps", dpi=300)
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


def evaluate_and_plot(
    model: FourierDoubleGateNet,
    data: eq.TemperatureData,
    out_dir: Path,
    mean: float,
    std: float,
    device: torch.device,
) -> dict[str, float]:
    # Reconstruct every training-window day, save per-day visual artifacts, and
    # compute the global RMSE/MAE/relative-L2 values reported in the manuscript.
    frames: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    day_results: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    composite_grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for day in range(eq.TRAIN_DAY_START, eq.TRAIN_DAY_END + 1):
        indices = eq.day_indices(data, day)
        pred = predict_indices(model, data, indices, mean, std, device)
        true = data.temp_c[indices].astype(np.float32)
        all_pred.append(pred)
        all_true.append(true)
        day_results.append((day, indices, pred, true))

    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)
    auto_temp_vmin, auto_temp_vmax = rounded_bounds(np.concatenate([pred_all, true_all]), 5.0)
    temp_vmin = TEMP_VMIN if TEMP_VMIN is not None else auto_temp_vmin
    temp_vmax = TEMP_VMAX if TEMP_VMAX is not None else auto_temp_vmax
    err_vmax = ERR_VMAX if ERR_VMAX is not None else rounded_upper(float(np.max(np.abs(pred_all - true_all))), 2.0)

    for day, indices, pred, true in day_results:
        pred_grid = build_masked_grid(data, indices, pred)
        true_grid = build_masked_grid(data, indices, true)
        frame_png = out_dir / f"frame_day_{day:02d}.png"
        plot_triplet(pred_grid, true_grid, day, frame_png, temp_vmin, temp_vmax, err_vmax)
        frames.append(imageio.imread(frame_png))

        if day in SNAPSHOT_DAYS:
            plot_triplet(
                pred_grid,
                true_grid,
                day,
                out_dir / f"snapshot_day_{day:02d}.png",
                temp_vmin,
                temp_vmax,
                err_vmax,
                save_vector=True,
            )
            save_day_csv(out_dir / f"snapshot_day_{day:02d}.csv", data, indices, pred, true, day)

        if day in COMPOSITE_DAYS:
            composite_grids[day] = (pred_grid, true_grid)

    gif_name = f"temperature_reconstruction_day{eq.TRAIN_DAY_START:02d}_{eq.TRAIN_DAY_END:02d}.gif"
    imageio.mimsave(out_dir / gif_name, frames, duration=0.65)
    plot_composite_triplets(composite_grids, out_dir, temp_vmin, temp_vmax, err_vmax)
    rmse = float(np.sqrt(np.mean((pred_all - true_all) ** 2)))
    mae = float(np.mean(np.abs(pred_all - true_all)))
    rel_l2 = float(np.linalg.norm(pred_all - true_all) / np.linalg.norm(true_all))
    return {
        "rmse_C": rmse,
        "mae_C": mae,
        "relative_l2": rel_l2,
        "plot_temp_vmin_C": temp_vmin,
        "plot_temp_vmax_C": temp_vmax,
        "plot_error_vmax_C": err_vmax,
    }


def main() -> None:
    # Full experiment driver: load the preprocessed land-mask dataset, sample
    # IC/BC/DATA constraints, train FDG-PINN, save the checkpoint, and export
    # metrics plus figures.
    set_deterministic(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = eq.RESULTS_DIR / f"reconstruction_fdg_aw_landmask_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"dtype={DTYPE}, seed={SEED}")
    print(
        f"Sampling: IC=all day-{eq.TRAIN_DAY_START} points, "
        f"BC={BC_FRACTION:.2f} per-day boundary points, DATA={DATA_FRACTION:.2f} "
        f"internal points day {eq.TRAIN_DAY_START}-{eq.TRAIN_DAY_END}"
    )
    print(f"Output directory: {out_dir}")

    data = eq.load_temperature_data()
    # The North China domain contains sea cells. Boundary points are therefore
    # defined by the four-neighbor land mask in temperature_fdg_eq.py rather
    # than by only x=0/1 or y=0/1 rectangular edges.
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
        out_dir / "fdg_temperature_model.pt",
    )
    metrics = evaluate_and_plot(model, data, out_dir, mean, std, device)

    with (out_dir / "run_summary.txt").open("w", encoding="utf-8") as f:
        f.write("FDG temperature reconstruction summary\n")
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
        f.write(f"ic_count={indices['ic'].size}\n")
        f.write(f"bc_count={indices['bc'].size}\n")
        f.write(f"data_count={indices['data'].size}\n")
        f.write(f"train_indices_file=train_indices.npz\n")
        f.write(f"model_checkpoint=fdg_temperature_model.pt\n")
        f.write(f"plot_temp_vmin_override_C={TEMP_VMIN}\n")
        f.write(f"plot_temp_vmax_override_C={TEMP_VMAX}\n")
        f.write(f"plot_error_vmax_override_C={ERR_VMAX}\n")
        f.write(f"temp_mean_C={mean}\n")
        f.write(f"temp_std_C={std}\n")
        for key, value in metrics.items():
            f.write(f"{key}={value}\n")

    print(f"Final reconstruction metrics day {eq.TRAIN_DAY_START}-{eq.TRAIN_DAY_END}:")
    print(f"  RMSE(C)     = {metrics['rmse_C']:.4f}")
    print(f"  MAE(C)      = {metrics['mae_C']:.4f}")
    print(f"  Relative L2 = {metrics['relative_l2']:.4e}")
    print(f"GIF saved to: {out_dir / f'temperature_reconstruction_day{eq.TRAIN_DAY_START:02d}_{eq.TRAIN_DAY_END:02d}.gif'}")


if __name__ == "__main__":
    main()
