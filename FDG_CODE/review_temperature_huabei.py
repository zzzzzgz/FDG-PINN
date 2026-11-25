"""Run paired North China temperature reconstruction review experiments."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.parent
HUABEI = PUBLISHED / "Temperature_forecast_huabei_50day"
OUTPUT = HERE / "review_temperature_huabei"
sys.path.insert(0, str(HUABEI))
sys.path.insert(0, str(PUBLISHED / "FDG_CODE"))


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


class PhiNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_width=128,
                 hidden_depth=4, freqs=(1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        from FDG_runner import ImprovedPINN
        self.model = ImprovedPINN(input_dim, output_dim, hidden=hidden_width,
                                  n_hidden_layers=hidden_depth)

    def forward(self, x):
        return self.model(x)


def evaluate_without_artifacts(runner, eq):
    def evaluate(model, data, out_dir, mean, std, device):
        predictions = []
        targets = []
        for day in range(eq.TRAIN_DAY_START, eq.TRAIN_DAY_END + 1):
            indices = eq.day_indices(data, day)
            predictions.append(runner.predict_indices(model, data, indices, mean, std, device))
            targets.append(data.temp_c[indices].astype(np.float32))
        pred = np.concatenate(predictions)
        true = np.concatenate(targets)
        difference = pred - true
        return {"rmse_C": float(np.sqrt(np.mean(difference ** 2))),
                "mae_C": float(np.mean(np.abs(difference))),
                "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(true))}
    return evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=("pinn", "phi_nn", "ff_mlp", "fdg"), required=True)
    args = parser.parse_args()
    import temperature_fdg_eq as eq

    result = OUTPUT / f"seed_{args.seed}" / args.method / "reconstruction_result.json"
    if result.exists():
        raise FileExistsError(f"Preserving existing result: {result}")
    if args.method == "pinn":
        source = HUABEI / "PINN_runner_temperature_fixed.py"
        spec = importlib.util.spec_from_file_location("review_huabei_plain", source)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.__file__ = str(result.parent.parent / "PINN_runner_temperature_fixed.py")
        root = result.parent.parent / "pinn_temperature_results"
        pattern = "reconstruction_pinn_fixed_*/run_summary.txt"
    else:
        import FDG_runner_temperature as runner
        if args.method == "phi_nn":
            runner.FourierDoubleGateNet = PhiNN
        elif args.method == "ff_mlp":
            runner.FourierDoubleGateNet = FourierMLP
        eq.RESULTS_DIR = result.parent
        root = eq.RESULTS_DIR
        pattern = "reconstruction_fdg_aw_*/run_summary.txt"
    runner.SEED = args.seed
    runner.evaluate_and_plot = evaluate_without_artifacts(runner, eq)
    runner.main()

    completed = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not completed:
        raise FileNotFoundError(f"No completed run in {root}")
    summary_path = completed[-1]
    summary = dict(line.split("=", 1) for line in summary_path.read_text(encoding="utf-8").splitlines() if "=" in line)
    record = {"seed": args.seed, "method": args.method, "run_dir": str(summary_path.parent),
              "rmse_C": float(summary["rmse_C"]), "mae_C": float(summary["mae_C"]),
              "relative_l2": float(summary["relative_l2"])}
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"RESULT_FILE={result}", flush=True)


if __name__ == "__main__":
    main()
