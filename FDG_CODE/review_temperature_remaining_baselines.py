"""Run paired plain PINN and phi-NN India temperature reconstruction baselines."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.parent
INDIA = PUBLISHED / "Temperature_forecast_india"
OUTPUT = HERE / "review_temperature_remaining_baselines"
sys.path.insert(0, str(INDIA))
sys.path.insert(0, str(PUBLISHED / "FDG_CODE"))


class PhiNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_width=128,
                 hidden_depth=4, freqs=(1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        from FDG_runner import ImprovedPINN
        self.model = ImprovedPINN(input_dim, output_dim, hidden=hidden_width,
                                  n_hidden_layers=hidden_depth)

    def forward(self, x):
        return self.model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=("pinn", "phi_nn"), required=True)
    args = parser.parse_args()
    import temperature_fdg_eq as eq

    if args.method == "pinn":
        source = INDIA / "PINN_runner_temperature_fixed.py"
        spec = importlib.util.spec_from_file_location("review_plain_temperature", source)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.__file__ = str(OUTPUT / f"seed_{args.seed}" / "PINN_runner_temperature_fixed.py")
        root = Path(runner.__file__).parent / "pinn_temperature_results"
        pattern = "reconstruction_pinn_fixed_*/run_summary.txt"
    else:
        import FDG_runner_temperature as runner
        runner.FourierDoubleGateNet = PhiNN
        eq.RESULTS_DIR = OUTPUT / f"seed_{args.seed}" / "phi_nn"
        root = eq.RESULTS_DIR
        pattern = "reconstruction_fdg_aw_*/run_summary.txt"
    runner.SEED = args.seed
    runner.main()

    completed = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    summary_path = completed[-1]
    summary = dict(line.split("=", 1) for line in summary_path.read_text(encoding="utf-8").splitlines() if "=" in line)
    record = {"seed": args.seed, "method": args.method,
              "run_dir": str(summary_path.parent),
              "rmse_C": float(summary["rmse_C"]),
              "mae_C": float(summary["mae_C"]),
              "relative_l2": float(summary["relative_l2"])}
    result = OUTPUT / f"seed_{args.seed}" / args.method / "reconstruction_result.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"RESULT_FILE={result}", flush=True)


if __name__ == "__main__":
    main()
