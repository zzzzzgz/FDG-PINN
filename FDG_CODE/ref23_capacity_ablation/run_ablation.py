"""Controlled Ref. [23] and capacity-matched gate ablations for the ND review."""

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
WORKSPACE = HERE.parents[1]
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
VARIANTS = ("ref23_equal_width", "ref23_matched_params", "first_gate_matched_params")
METRICS = (
    "training_time", "final_pde_loss", "final_ic_loss", "final_bc_loss",
    "train_l2", "global_l2", "discrete_l2",
)
FIELDS = ("equation", "seed", "variant", "hidden_width", "parameter_count",
          "fdg_parameter_count", "weight_mode", *METRICS)


class SingleGatePINN(nn.Module):
    """First-gate-only version of the FDG backbone (deeper U/V branches)."""

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, width)
        self.u_net = runner.MLP(in_dim, width, hidden=64, depth=2, act=nn.Tanh)
        self.v_net = runner.MLP(in_dim, width, hidden=64, depth=2, act=nn.Tanh)
        self.gates = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.out_layer = nn.Linear(width, out_dim)

    def forward(self, x):
        u = self.u_net(x)
        v = self.v_net(x)
        h = torch.tanh(self.in_layer(x))
        for gate in self.gates:
            b = torch.sigmoid(gate(h))
            h = (1.0 - b) * u + b * v
        return self.out_layer(h)


class Ref23ModifiedMLP(nn.Module):
    """Equations (43)--(47) of Wang, Teng, and Perdikaris (2021)."""

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        self.u_layer = nn.Linear(in_dim, width)
        self.v_layer = nn.Linear(in_dim, width)
        self.in_layer = nn.Linear(in_dim, width)
        self.gates = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.out_layer = nn.Linear(width, out_dim)

    def forward(self, x):
        u = torch.tanh(self.u_layer(x))
        v = torch.tanh(self.v_layer(x))
        h = torch.tanh(self.in_layer(x))
        for gate in self.gates:
            z = torch.tanh(gate(h))
            h = (1.0 - z) * u + z * v
        return self.out_layer(h)


ORIGINAL_BUILD_MODEL = runner.build_model


def trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_model(eq, variant, width):
    lows, highs = runner.input_bounds_from_eq(eq)
    encoder = runner.FourierFeatureEncoder(lows, highs, runner.fourier_freqs)
    if variant.startswith("ref23_"):
        trunk = Ref23ModifiedMLP(encoder.out_dim, eq.OUTPUT_DIM, width, runner.n_hidden_layers)
    elif variant == "first_gate_matched_params":
        trunk = SingleGatePINN(encoder.out_dim, eq.OUTPUT_DIM, width, runner.n_hidden_layers)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return runner.EncodedPINN(encoder, trunk).to(device=runner.device, dtype=runner.TORCH_DTYPE)


def fdg_count(eq):
    return trainable_parameters(ORIGINAL_BUILD_MODEL(eq, "fourier_double_gate"))


def matched_width(eq, variant, target):
    # Deterministic architecture-only selection; no outcome metric is used.
    choices = ((abs(trainable_parameters(make_model(eq, variant, width)) - target), width)
               for width in range(64, 321))
    return min(choices)[1]


def build_model(eq, model_kind):
    if model_kind not in VARIANTS:
        return ORIGINAL_BUILD_MODEL(eq, model_kind)
    target = fdg_count(eq)
    width = runner.hidden if model_kind == "ref23_equal_width" else matched_width(eq, model_kind, target)
    return make_model(eq, model_kind, width)


runner.build_model = build_model
runner.ROOT_DIR = str(HERE)
runner.SAVE_ARTIFACTS = False
runner.SAVE_PREDICTION_DATA = False
runner.APPEND_TEX = False
runner.WRITE_LOG_XLSX = False


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    output = []
    for equation in CASES:
        for variant in VARIANTS:
            subset = [row for row in rows if row["equation"] == equation and row["variant"] == variant]
            if not subset:
                continue
            item = {"equation": equation, "variant": variant, "n": len(subset),
                    "hidden_width": subset[0]["hidden_width"],
                    "parameter_count": subset[0]["parameter_count"],
                    "fdg_parameter_count": subset[0]["fdg_parameter_count"]}
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            output.append(item)
    fields = ["equation", "variant", "n", "hidden_width", "parameter_count", "fdg_parameter_count"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    with (HERE / "summary_mean_std.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation", action="append", choices=CASES)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    cases = args.equation or CASES
    variants = args.variant or VARIANTS
    seeds = tuple(int(value) for value in args.seeds.split(","))
    raw_path = HERE / "raw_runs.csv"
    rows = read_rows(raw_path)
    done = {(row["equation"], int(row["seed"]), row["variant"]) for row in rows}
    print(f"Device={runner.device}; dtype={runner.TORCH_DTYPE}; seeds={seeds}", flush=True)
    for equation in cases:
        eq = importlib.import_module(equation)
        target = fdg_count(eq)
        for variant in variants:
            width = runner.hidden if variant == "ref23_equal_width" else matched_width(eq, variant, target)
            count = trainable_parameters(make_model(eq, variant, width))
            print(f"Config {equation} {variant}: width={width} params={count} FDG={target}", flush=True)
            for seed in seeds:
                if (equation, seed, variant) in done:
                    print(f"Skip {equation} {variant} seed={seed}", flush=True)
                    continue
                runner.seed = seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                print(f"Run {equation} {variant} seed={seed}", flush=True)
                metrics = runner.run_combo(equation, variant, "aw")
                row = {"equation": equation, "seed": seed, "variant": variant,
                       "hidden_width": width, "parameter_count": count,
                       "fdg_parameter_count": target, "weight_mode": "aw"}
                row.update({key: float(metrics["time_avg_l2"] if key == "discrete_l2" else metrics[key])
                            for key in METRICS})
                rows.append(row)
                done.add((equation, seed, variant))
                write_rows(raw_path, rows)
                summarize(rows)
                print(f"Saved L2={row['global_l2']:.6g} time={row['training_time']:.2f}s", flush=True)
    print(f"Complete: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
