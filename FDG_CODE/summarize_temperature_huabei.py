"""Check paired North China samples and summarize four methods over five seeds."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "review_temperature_huabei"
SEEDS = (1111, 2222, 3333, 4444, 5555)
METHODS = ("pinn", "phi_nn", "ff_mlp", "fdg")
METRICS = ("ic", "bc", "data", "rmse_C", "mae_C", "relative_l2")


def main():
    rows = []
    for seed in SEEDS:
        reference = None
        for method in METHODS:
            record = json.loads((ROOT / f"seed_{seed}" / method / "reconstruction_result.json").read_text(encoding="utf-8"))
            run_dir = Path(record["run_dir"])
            with np.load(run_dir / "train_indices.npz") as saved:
                current = {key: saved[key].copy() for key in ("ic", "bc", "data", "train_all")}
            if reference is None:
                reference = current
            elif not all(np.array_equal(reference[key], current[key]) for key in reference):
                raise RuntimeError(f"Sampling mismatch: seed={seed}, method={method}")
            with (run_dir / "training_loss.csv").open(newline="", encoding="utf-8") as stream:
                last = list(csv.DictReader(stream))[-1]
            rows.append({"seed": seed, "method": method,
                         "ic": float(last["ic"]), "bc": float(last["bc"]),
                         "data": float(last["data"]),
                         **{key: record[key] for key in ("rmse_C", "mae_C", "relative_l2")}})
    with (ROOT / "four_method_paired_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result = {"method": method}
        for key in METRICS:
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_std"] = float(values.std(ddof=1))
        summary.append(result)
    with (ROOT / "four_method_five_seed_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("Verified identical IC/BC/data/train indices for all methods at each seed")
    for row in summary:
        print(row)
    for method in METHODS[:-1]:
        wins = sum(next(row["relative_l2"] for row in rows if row["seed"] == seed and row["method"] == "fdg")
                   < next(row["relative_l2"] for row in rows if row["seed"] == seed and row["method"] == method)
                   for seed in SEEDS)
        print(f"FDG L2 wins vs {method}: {wins}/5")


if __name__ == "__main__":
    main()
