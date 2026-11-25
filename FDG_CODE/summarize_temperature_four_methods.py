"""Validate and summarize paired five-seed India reconstruction results."""

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SEEDS = (1111, 2222, 3333, 4444, 5555)
METHODS = ("pinn", "phi_nn", "ff_mlp", "fdg")
METRICS = ("ic", "bc", "data", "rmse_C", "mae_C", "relative_l2")
OUT = HERE / "review_temperature_ff_mlp"


def run_path(seed, method):
    root = (HERE / "review_temperature_remaining_baselines"
            if method in ("pinn", "phi_nn") else OUT)
    return root / f"seed_{seed}" / method / "reconstruction_result.json"


def main():
    rows = []
    for seed in SEEDS:
        reference = None
        for method in METHODS:
            record = json.loads(run_path(seed, method).read_text(encoding="utf-8"))
            run_dir = Path(record["run_dir"])
            with np.load(run_dir / "train_indices.npz") as indices:
                current = {key: indices[key].copy() for key in ("ic", "bc", "data", "train_all")}
            if reference is None:
                reference = current
            else:
                assert all(np.array_equal(reference[key], current[key]) for key in reference), (seed, method)
            with (run_dir / "training_loss.csv").open(newline="", encoding="utf-8") as stream:
                last = list(csv.DictReader(stream))[-1]
            rows.append({"seed": seed, "method": method,
                         "ic": float(last["ic"]), "bc": float(last["bc"]),
                         "data": float(last["data"]),
                         **{key: record[key] for key in ("rmse_C", "mae_C", "relative_l2")}})
    with (OUT / "four_method_paired_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result = {"method": method}
        for key in METRICS:
            values = np.array([row[key] for row in selected])
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_std"] = float(values.std(ddof=1))
        summary.append(result)
    with (OUT / "four_method_five_seed_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("Verified identical IC/BC/data/train indices for all four methods at every seed")
    for row in summary:
        print(row["method"], "RMSE", f"{row['rmse_C_mean']:.4f} +/- {row['rmse_C_std']:.4f}",
              "L2", f"{row['relative_l2_mean']:.5f} +/- {row['relative_l2_std']:.5f}")
    for method in METHODS[:-1]:
        wins = sum(next(row["relative_l2"] for row in rows if row["seed"] == seed and row["method"] == "fdg")
                   < next(row["relative_l2"] for row in rows if row["seed"] == seed and row["method"] == method)
                   for seed in SEEDS)
        print("FDG L2 wins vs", method, wins, "/ 5")


if __name__ == "__main__":
    main()
