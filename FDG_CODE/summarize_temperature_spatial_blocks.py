"""Compare centered holdouts with the saved unblocked India controls."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from review_temperature_spatial_blocks import OUTPUT, block_mask, eq, runner, relative_l2
from review_temperature_ff_mlp import FourierMLP


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "review_temperature_ff_mlp"
SEEDS = (1111, 2222, 3333, 4444, 5555)
METHODS = ("ff_mlp", "fdg")
SIDES = (0.2, 0.4, 0.6)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_block_l2(data, side, method, seed, device):
    record = json.loads((BASELINE / f"seed_{seed}" / method / "reconstruction_result.json").read_text(encoding="utf-8"))
    run_dir = Path(record["run_dir"])
    state = torch.load(run_dir / "fdg_temperature_model.pt", map_location=device, weights_only=True)
    model_type = FourierMLP if method == "ff_mlp" else runner.FourierDoubleGateNet
    model = model_type(input_dim=eq.INPUT_DIM, output_dim=eq.OUTPUT_DIM,
                       hidden_width=state["hidden_width"], hidden_depth=state["hidden_depth"],
                       freqs=state["fourier_freqs"]).to(device=device, dtype=runner.DTYPE)
    model.load_state_dict(state["model_state_dict"])
    mask = block_mask(data, side) & (data.day > eq.TRAIN_DAY_START) & (data.day <= eq.TRAIN_DAY_END)
    indices = np.flatnonzero(mask)
    pred = runner.predict_indices(model, data, indices, state["temp_mean_C"],
                                  state["temp_std_C"], device)
    return relative_l2(pred, data.temp_c[indices]), record["relative_l2"], len(indices)


def main():
    data = eq.load_temperature_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for side in SIDES:
        for seed in SEEDS:
            paired_indices = None
            for method in METHODS:
                path = OUTPUT / f"side_{side:.1f}" / f"seed_{seed}" / method / "result.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                with np.load(Path(record["run_dir"]) / "train_indices.npz") as saved:
                    current = {key: saved[key].copy() for key in ("ic", "bc", "data")}
                if paired_indices is None:
                    paired_indices = current
                elif not all(np.array_equal(paired_indices[key], current[key]) for key in current):
                    raise RuntimeError(f"Unpaired observations: side={side}, seed={seed}")
                base_holdout, base_full, count = baseline_block_l2(data, side, method, seed, device)
                if count != record["heldout_points"]:
                    raise RuntimeError(f"Held-out count mismatch: side={side}, seed={seed}")
                rows.append({"side": side, "approx_area_fraction": side ** 2,
                             "seed": seed, "method": method, "heldout_points": count,
                             "unblocked_full_l2": base_full,
                             "blocked_full_l2": record["relative_l2"],
                             "unblocked_region_l2": base_holdout,
                             "blocked_region_l2": record["heldout_relative_l2"]})
    save_csv(OUTPUT / "paired_block_results.csv", rows)
    summary = []
    for side in SIDES:
        for method in METHODS:
            chosen = [row for row in rows if row["side"] == side and row["method"] == method]
            entry = {"side": side, "approx_area_fraction": side ** 2, "method": method,
                     "heldout_points": chosen[0]["heldout_points"]}
            for key in ("unblocked_full_l2", "blocked_full_l2",
                        "unblocked_region_l2", "blocked_region_l2"):
                values = np.asarray([row[key] for row in chosen], dtype=np.float64)
                entry[f"{key}_mean"] = float(values.mean())
                entry[f"{key}_std"] = float(values.std(ddof=1))
            entry["region_l2_increase_pct"] = 100 * (
                entry["blocked_region_l2_mean"] / entry["unblocked_region_l2_mean"] - 1)
            summary.append(entry)
    save_csv(OUTPUT / "five_seed_block_summary.csv", summary)
    print("Verified identical IC/BC/data indices for paired FF-MLP and FDG-PINN runs")
    for entry in summary:
        print(entry)


if __name__ == "__main__":
    main()
