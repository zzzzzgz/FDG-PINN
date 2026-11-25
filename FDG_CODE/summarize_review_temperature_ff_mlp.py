"""Validate paired sampling and summarize the India FF-MLP comparison."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "review_temperature_ff_mlp"
SEEDS = (1111, 2222, 3333, 4444, 5555)
METHODS = ("ff_mlp", "fdg")
KEYS = ("ic", "bc", "data", "train_all")
METRICS = ("rmse_C", "mae_C", "relative_l2")


def main():
    rows = []
    for seed in SEEDS:
        records = {}
        indices = {}
        for method in METHODS:
            directory = ROOT / f"seed_{seed}" / method
            records[method] = json.loads((directory / "reconstruction_result.json").read_text(encoding="utf-8"))
            indices[method] = np.load(Path(records[method]["run_dir"]) / "train_indices.npz")
            rows.append({"seed": seed, "method": method,
                         **{key: records[method][key] for key in METRICS}})
        assert all(np.array_equal(indices["ff_mlp"][key], indices["fdg"][key]) for key in KEYS), seed
    with (ROOT / "paired_five_seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Paired indices verified for all five seeds")
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        print(method)
        for key in METRICS:
            values = [row[key] for row in selected]
            print(f"  {key}: {np.mean(values):.6f} +/- {np.std(values, ddof=1):.6f}")
    for key in METRICS:
        wins = sum(next(row[key] for row in rows if row["seed"] == seed and row["method"] == "fdg")
                   < next(row[key] for row in rows if row["seed"] == seed and row["method"] == "ff_mlp")
                   for seed in SEEDS)
        print(f"FDG wins for {key}: {wins}/5")


if __name__ == "__main__":
    main()
