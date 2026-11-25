"""Five-seed, first-gate-only ablation using the published FDG training pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PUBLISHED_CODE = HERE.parent
sys.path.insert(0, str(PUBLISHED_CODE))
import FDG_runner as runner  # noqa: E402


CASES = (
    "fdg_transport1d_eq",
    "fdg_burgers_mixed1d_eq",
    "fdg_helmholtz2d_k10_eq",
    "fdg_helmholtz2d_k20_eq",
    "fdg_helmholtz2d_mixedfreq_eq",
    "fdg_reaction_diffusion_mixed1d_eq",
)
SEEDS = (1111, 2222, 3333, 4444, 5555)
METRICS = (
    "training_time",
    "final_pde_loss",
    "final_ic_loss",
    "final_bc_loss",
    "train_l2",
    "global_l2",
    "discrete_l2",
)
FIELDS = ("equation", "seed", "method", "model_kind", "weight_mode", "parameter_count", *METRICS)


class SingleGatePINN(nn.Module):
    """Modified-MLP-style reference interpolation without the fusion gate."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        self.act = nn.Tanh()
        self.in_layer = nn.Linear(in_dim, hidden)
        self.U_net = runner.MLP(in_dim, hidden, hidden=64, depth=2, act=nn.Tanh)
        self.V_net = runner.MLP(in_dim, hidden, hidden=64, depth=2, act=nn.Tanh)
        self.B_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth - 1))
        self.out_layer = nn.Linear(hidden, out_dim)

    def forward(self, x):
        u = self.U_net(x)
        v = self.V_net(x)
        z = self.act(self.in_layer(x))
        for gate in self.B_layers:
            b = torch.sigmoid(gate(z))
            z = (1.0 - b) * u + b * v
        return self.out_layer(z)


ORIGINAL_BUILD_MODEL = runner.build_model


def build_model(eq, model_kind):
    if model_kind != "fourier_single_gate":
        return ORIGINAL_BUILD_MODEL(eq, model_kind)
    lows, highs = runner.input_bounds_from_eq(eq)
    encoder = runner.FourierFeatureEncoder(lows, highs, runner.fourier_freqs)
    trunk = SingleGatePINN(encoder.out_dim, eq.OUTPUT_DIM, runner.hidden, runner.n_hidden_layers)
    return runner.EncodedPINN(encoder, trunk).to(device=runner.device, dtype=runner.TORCH_DTYPE)


runner.build_model = build_model
runner.ROOT_DIR = str(HERE)
runner.SAVE_ARTIFACTS = False
runner.SAVE_PREDICTION_DATA = False
runner.APPEND_TEX = False
runner.WRITE_LOG_XLSX = False


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(path: Path, rows):
    values = []
    for case in CASES:
        selected = [row for row in rows if row["equation"] == case]
        if not selected:
            continue
        result = {"equation": case, "n": len(selected), "parameter_count": selected[0]["parameter_count"]}
        for metric in METRICS:
            data = np.asarray([float(row[metric]) for row in selected], dtype=float)
            result[f"{metric}_mean"] = float(data.mean())
            result[f"{metric}_std"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        values.append(result)
    fields = ["equation", "n", "parameter_count"]
    for metric in METRICS:
        fields += [f"{metric}_mean", f"{metric}_std"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation", action="append", choices=CASES)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    cases = args.equation or CASES
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    raw_path = HERE / "raw_runs.csv"
    summary_path = HERE / "summary_mean_std.csv"
    rows = read_rows(raw_path)
    completed = {(row["equation"], int(row["seed"])) for row in rows}
    print(f"Single-gate ablation on {runner.device}; seeds={seeds}", flush=True)
    if runner.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    for case in cases:
        eq = importlib.import_module(case)
        for seed in seeds:
            if (case, seed) in completed:
                print(f"Skip existing: {case} seed={seed}", flush=True)
                continue
            runner.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = build_model(eq, "fourier_single_gate")
            parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
            del model
            print(f"Run: {case} seed={seed} parameters={parameters}", flush=True)
            result = runner.run_combo(case, "fourier_single_gate", "aw")
            row = {
                "equation": case,
                "seed": seed,
                "method": "SingleGate-PINN",
                "model_kind": "fourier_single_gate",
                "weight_mode": "aw",
                "parameter_count": parameters,
                **{metric: float(result["time_avg_l2"] if metric == "discrete_l2" else result[metric]) for metric in METRICS},
            }
            rows.append(row)
            completed.add((case, seed))
            write_rows(raw_path, rows)
            summarize(summary_path, rows)
            print(f"Saved: {case} seed={seed} L2={row['global_l2']:.6g} time={row['training_time']:.2f}s", flush=True)
    print(f"Complete: {len(rows)} runs in {raw_path}", flush=True)


if __name__ == "__main__":
    main()
