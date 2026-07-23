"""
FDG runner for out-of-window temperature field forecasting with future-day
interior-point supervision.

Train on day 1--20 using IC/BC/internal-data samples, add partial interior
observations on the out-of-window days 21--50, then evaluate the resulting
model on all days 1--50 and save daily metrics.
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", os.environ.get("TEMP_SEED", "2026"))

import numpy as np
import torch
import torch.nn as nn

import temperature_fdg_eq as eq


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


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
FUTURE_DATA_FRACTION = env_float("TEMP_FUTURE_DATA_FRACTION", DATA_FRACTION)
HIDDEN_WIDTH = env_int("TEMP_HIDDEN_WIDTH", 128)
HIDDEN_DEPTH = env_int("TEMP_HIDDEN_DEPTH", 4)
FOURIER_FREQS = env_list_float("TEMP_FOURIER_FREQS", (1.0, 2.0, 4.0, 8.0))
DTYPE_NAME = os.environ.get("TEMP_DTYPE", "float32").lower()
DTYPE = torch.float64 if DTYPE_NAME == "float64" else torch.float32
RESULTS_DIR = Path(__file__).resolve().parent / "fdg_temperature_forecast_results"
FUTURE_DAY_START = env_int("TEMP_FORECAST_DAY_START", eq.FORECAST_DAY_START)
FUTURE_DAY_END = env_int("TEMP_FORECAST_DAY_END", eq.FORECAST_DAY_END)


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


def select_future_interior_indices(
    data: eq.TemperatureData,
    rng: np.random.Generator,
    fraction: float,
    day_start: int,
    day_end: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for day in range(day_start, day_end + 1):
        day_interior = np.flatnonzero((data.day == day) & (~data.is_boundary))
        if day_interior.size:
            n_pick = max(1, int(round(float(fraction) * day_interior.size)))
            parts.append(rng.choice(day_interior, size=n_pick, replace=False))
    return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def train_model(
    model: FourierDoubleGateNet,
    aw: AdaptiveWeights,
    tensors: dict[str, torch.Tensor],
    out_dir: Path,
) -> None:
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

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            total, _, _, _ = compute_losses()
            total.backward()
            return total

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

    with (out_dir / "training_loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "stage", "total", "ic", "bc", "data", "w_ic", "w_bc", "w_data"])
        writer.writeheader()
        writer.writerows(history)


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
    inputs = eq.make_inputs(data, indices)
    preds: list[np.ndarray] = []
    model.eval()
    for start in range(0, inputs.shape[0], batch_size):
        x = torch.as_tensor(inputs[start : start + batch_size], dtype=DTYPE, device=device)
        y = model(x).squeeze(1).detach().cpu().numpy()
        preds.append(y)
    return np.concatenate(preds).astype(np.float64) * std + mean


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    denom = float(np.linalg.norm(true))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(pred - true) / denom)


def metric_row(day: int, indices: np.ndarray, pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    mse = float(np.mean(err**2))
    return {
        "day": float(day),
        "points": float(indices.size),
        "mse_C2": mse,
        "rmse_C": float(np.sqrt(mse)),
        "mae_C": float(np.mean(np.abs(err))),
        "relative_l2": relative_l2(pred, true),
    }


def evaluate_model(
    model: FourierDoubleGateNet,
    data: eq.TemperatureData,
    mean: float,
    std: float,
    device: torch.device,
    future_observed_idx: np.ndarray,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    full_rows: list[dict[str, float]] = []
    unobserved_rows: list[dict[str, float]] = []
    observed_future_set = set(int(i) for i in future_observed_idx.tolist())

    for day in range(int(data.unique_days.min()), int(data.unique_days.max()) + 1):
        indices = eq.day_indices(data, day)
        if indices.size == 0:
            continue
        pred = predict_indices(model, data, indices, mean, std, device)
        true = data.temp_c[indices].astype(np.float64)
        full_rows.append(metric_row(day, indices, pred, true))

        if day >= FUTURE_DAY_START:
            keep_mask = np.array([int(i) not in observed_future_set for i in indices], dtype=bool)
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


def save_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["day", "points", "mse_C2", "rmse_C", "mae_C", "relative_l2"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_deterministic(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"forecast_data_fdg_aw_landmask_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = eq.load_temperature_data()
    rng = np.random.default_rng(SEED)
    indices = eq.select_training_indices(data, rng, ic_all=True, bc_fraction=BC_FRACTION, data_fraction=DATA_FRACTION)
    data_train = indices["data"]
    data_future = select_future_interior_indices(data, rng, FUTURE_DATA_FRACTION, FUTURE_DAY_START, FUTURE_DAY_END)
    indices["data_train"] = data_train
    indices["data_future"] = data_future
    indices["data"] = np.sort(np.concatenate([data_train, data_future]))

    train_all = indices["train_all"]
    temp_train = data.temp_c[train_all]
    mean = float(temp_train.mean())
    std = float(temp_train.std() if temp_train.std() > 0 else 1.0)

    np.savez_compressed(
        out_dir / "train_indices.npz",
        ic=indices["ic"],
        bc=indices["bc"],
        data=indices["data"],
        data_train=indices["data_train"],
        data_future=indices["data_future"],
        train_all=indices["train_all"],
    )

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
            "future_data_fraction": FUTURE_DATA_FRACTION,
            "adam_steps": ADAM_STEPS,
            "lbfgs_steps": LBFGS_STEPS,
            "boundary_mode": "four_neighbor_land_mask",
            "forecast_mode": "future_interior_data",
        },
        out_dir / "fdg_temperature_forecast_model.pt",
    )

    full_rows, unobserved_rows = evaluate_model(model, data, mean, std, device, data_future)
    save_rows(out_dir / "metrics_day01_50.csv", full_rows)
    save_rows(out_dir / "forecast_unobserved_metrics_day21_50.csv", unobserved_rows)

    forecast_rows = [row for row in full_rows if FUTURE_DAY_START <= row["day"] <= FUTURE_DAY_END]
    save_rows(out_dir / "forecast_metrics_by_day.csv", forecast_rows)

    with (out_dir / "forecast_summary.txt").open("w", encoding="utf-8") as f:
        f.write("FDG temperature forecast with future interior-point supervision\n")
        f.write(f"seed={SEED}\n")
        f.write(f"dtype={DTYPE}\n")
        f.write(f"adam_steps={ADAM_STEPS}\n")
        f.write(f"lbfgs_steps={LBFGS_STEPS}\n")
        f.write(f"hidden_width={HIDDEN_WIDTH}\n")
        f.write(f"hidden_depth={HIDDEN_DEPTH}\n")
        f.write(f"fourier_freqs={FOURIER_FREQS}\n")
        f.write(f"bc_fraction={BC_FRACTION}\n")
        f.write(f"data_fraction={DATA_FRACTION}\n")
        f.write(f"future_data_fraction={FUTURE_DATA_FRACTION}\n")
        f.write(f"ic_count={indices['ic'].size}\n")
        f.write(f"bc_count={indices['bc'].size}\n")
        f.write(f"data_train_count={indices['data_train'].size}\n")
        f.write(f"data_future_count={indices['data_future'].size}\n")
        f.write(f"data_total_count={indices['data'].size}\n")
        f.write(f"temp_mean_C={mean}\n")
        f.write(f"temp_std_C={std}\n")


if __name__ == "__main__":
    main()
