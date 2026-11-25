"""Evaluate North China FDG checkpoints on supervised and held-out land cells."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
HUABEI = HERE.parent / "Temperature_forecast_huabei_50day"
ROOT = HERE / "review_temperature_huabei"
sys.path.insert(0, str(HUABEI))

import temperature_fdg_eq as eq
import FDG_runner_temperature as runner


def relative_l2(pred, true):
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def main():
    data = eq.load_temperature_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in (1111, 2222, 3333, 4444, 5555):
        record = json.loads((ROOT / f"seed_{seed}" / "fdg" / "reconstruction_result.json").read_text(encoding="utf-8"))
        run_dir = Path(record["run_dir"])
        with np.load(run_dir / "train_indices.npz") as saved:
            train_all = saved["train_all"]
            observed = np.unique(np.concatenate([saved[key] for key in ("ic", "bc", "data")]))
        unobserved = np.setdiff1d(train_all, observed, assume_unique=True)
        checkpoint = torch.load(run_dir / "fdg_temperature_model.pt", map_location=device, weights_only=True)
        model = runner.FourierDoubleGateNet(
            input_dim=eq.INPUT_DIM, output_dim=eq.OUTPUT_DIM,
            hidden_width=checkpoint["hidden_width"], hidden_depth=checkpoint["hidden_depth"],
            freqs=checkpoint["fourier_freqs"],
        ).to(device=device, dtype=runner.DTYPE)
        model.load_state_dict(checkpoint["model_state_dict"])
        mean, std = checkpoint["temp_mean_C"], checkpoint["temp_std_C"]
        observed_pred = runner.predict_indices(model, data, observed, mean, std, device)
        unobserved_pred = runner.predict_indices(model, data, unobserved, mean, std, device)
        observed_true = data.temp_c[observed]
        unobserved_true = data.temp_c[unobserved]
        full_l2 = float(np.sqrt(np.sum((observed_pred.astype(np.float64) - observed_true) ** 2)
                                + np.sum((unobserved_pred.astype(np.float64) - unobserved_true) ** 2)) /
                        np.sqrt(np.sum(observed_true.astype(np.float64) ** 2)
                                + np.sum(unobserved_true.astype(np.float64) ** 2)))
        if abs(full_l2 - record["relative_l2"]) > 2e-6:
            raise RuntimeError(f"Seed {seed}: full L2 mismatch ({full_l2} vs {record['relative_l2']})")
        rows.append({"seed": seed, "training_window_points": len(train_all),
                     "observed_points": len(observed), "unobserved_points": len(unobserved),
                     "observed_relative_l2": relative_l2(observed_pred, observed_true),
                     "unobserved_relative_l2": relative_l2(unobserved_pred, unobserved_true),
                     "full_relative_l2": full_l2})
    output = ROOT / "fdg_observed_unobserved_five_seeds.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for key in ("observed_points", "unobserved_points", "observed_relative_l2",
                "unobserved_relative_l2", "full_relative_l2"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        print(f"{key}: mean={values.mean():.8f}, sample_std={values.std(ddof=1):.8f}, "
              f"min={values.min():.8f}, max={values.max():.8f}")


if __name__ == "__main__":
    main()
