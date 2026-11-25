"""Compare uniform and focused Burgers front experiments across five seeds."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_burgers_front_probe import relative_error, sine_components


HERE = Path(__file__).resolve().parent
DESIGNS = ("fdg_burgers_front1d_eq", "fdg_burgers_front1d_focused_eq")
METHODS = ("pinn", "ff_pinn", "fdg_pinn")
SEEDS = (1111, 2222, 3333, 4444, 5555)


def spectral_errors(field):
    pred = field["pred"][:, :, 0]
    true = field["true"][:, :, 0]
    pred_parts = [sine_components(slice_) for slice_ in pred]
    true_parts = [sine_components(slice_) for slice_ in true]
    low_pred = np.stack([part[0] for part in pred_parts])
    low_true = np.stack([part[0] for part in true_parts])
    high_pred = np.stack([part[1] for part in pred_parts])
    high_true = np.stack([part[1] for part in true_parts])
    return {
        "low_modes_1_to_4_l2": relative_error(low_pred, low_true),
        "high_modes_5_plus_l2": relative_error(high_pred, high_true),
        "high_modes_t1_l2": relative_error(high_pred[-1], high_true[-1]),
    }


def main():
    rows = []
    for design in DESIGNS:
        for seed in SEEDS:
            directory = HERE / design / f"probe_seed{seed}"
            for method in METHODS:
                metrics = json.loads((directory / f"{method}_seed{seed}.json").read_text(encoding="utf-8"))
                field_path = next((directory / method).rglob("*field_data.npz"))
                with np.load(field_path) as field:
                    spectral = spectral_errors(field)
                rows.append({
                    "design": "uniform" if design == DESIGNS[0] else "front_focused",
                    "seed": seed,
                    "method": method,
                    "global_l2": metrics["global_l2"],
                    "training_time_s": metrics["training_time_s"],
                    **spectral,
                })
    output = HERE / "fdg_burgers_front1d_focused_eq" / "focused_vs_uniform_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", output)
    for design in ("uniform", "front_focused"):
        print(design)
        subset = [row for row in rows if row["design"] == design]
        for method in METHODS:
            selected = [row for row in subset if row["method"] == method]
            print(method, "global", np.mean([r["global_l2"] for r in selected]),
                  "std", np.std([r["global_l2"] for r in selected], ddof=1),
                  "low", np.mean([r["low_modes_1_to_4_l2"] for r in selected]),
                  "high", np.mean([r["high_modes_5_plus_l2"] for r in selected]))
        wins = sum(next(r["global_l2"] for r in subset if r["seed"] == seed and r["method"] == "fdg_pinn")
                   < next(r["global_l2"] for r in subset if r["seed"] == seed and r["method"] == "ff_pinn")
                   for seed in SEEDS)
        print("FDG wins over FF", wins, "/", len(SEEDS))


if __name__ == "__main__":
    main()
