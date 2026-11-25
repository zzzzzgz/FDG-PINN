"""Paired India temperature reconstruction: Fourier MLP versus FDG backbone."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
INDIA = HERE.parent / "Temperature_forecast_india"
OUTPUT = HERE / "review_temperature_ff_mlp"
sys.path.insert(0, str(INDIA))


class FourierMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_width=128,
                 hidden_depth=4, freqs=(1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        self.freqs = tuple(float(freq) for freq in freqs)
        encoded_dim = input_dim * (1 + 2 * len(self.freqs))
        self.input_layer = nn.Linear(encoded_dim, hidden_width)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth))
        self.output_layer = nn.Linear(hidden_width, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        pieces = [x]
        for freq in self.freqs:
            phase = 2.0 * np.pi * freq * x
            pieces.extend((torch.sin(phase), torch.cos(phase)))
        h = self.activation(self.input_layer(torch.cat(pieces, dim=1)))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=("ff_mlp", "fdg"), required=True)
    args = parser.parse_args()

    import temperature_fdg_eq as eq
    import FDG_runner_temperature as runner

    eq.RESULTS_DIR = OUTPUT / f"seed_{args.seed}" / args.method
    runner.SEED = args.seed
    if args.method == "ff_mlp":
        runner.FourierDoubleGateNet = FourierMLP
    runner.main()

    completed = sorted(eq.RESULTS_DIR.glob("reconstruction_fdg_aw_*/run_summary.txt"),
                       key=lambda path: path.stat().st_mtime)
    summary_path = completed[-1]
    summary = dict(line.split("=", 1) for line in summary_path.read_text(encoding="utf-8").splitlines() if "=" in line)
    record = {"seed": args.seed, "method": args.method,
              "run_dir": str(summary_path.parent),
              "rmse_C": float(summary["rmse_C"]),
              "mae_C": float(summary["mae_C"]),
              "relative_l2": float(summary["relative_l2"])}
    result = eq.RESULTS_DIR / "reconstruction_result.json"
    result.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"RESULT_FILE={result}", flush=True)


if __name__ == "__main__":
    main()
