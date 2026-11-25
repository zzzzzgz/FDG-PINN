"""Paired India reconstruction with three centered spatial observation holdouts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
INDIA = HERE.parent / "Temperature_forecast_india"
OUTPUT = HERE / "review_temperature_spatial_blocks"
sys.path.insert(0, str(INDIA))

import temperature_fdg_eq as eq
import FDG_runner_temperature as runner
from review_temperature_ff_mlp import FourierMLP


def block_mask(data, side):
    lon_mid = (float(data.lon.min()) + float(data.lon.max())) / 2
    lat_mid = (float(data.lat.min()) + float(data.lat.max())) / 2
    half_lon = side * (float(data.lon.max()) - float(data.lon.min())) / 2
    half_lat = side * (float(data.lat.max()) - float(data.lat.min())) / 2
    return ((np.abs(data.lon - lon_mid) <= half_lon)
            & (np.abs(data.lat - lat_mid) <= half_lat))


def relative_l2(pred, true):
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1111, 2222, 3333, 4444, 5555), required=True)
    parser.add_argument("--method", choices=("ff_mlp", "fdg"), required=True)
    parser.add_argument("--side", type=float, choices=(0.2, 0.4, 0.6), required=True)
    args = parser.parse_args()
    case = f"side_{args.side:.1f}"
    result_path = OUTPUT / case / f"seed_{args.seed}" / args.method / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Preserving completed run: {result_path}")
    eq.RESULTS_DIR = result_path.parent
    runner.SEED = args.seed
    if args.method == "ff_mlp":
        runner.FourierDoubleGateNet = FourierMLP

    original_select = eq.select_training_indices

    def select_outside_block(data, rng, **kwargs):
        indices = original_select(data, rng, **kwargs)
        excluded = block_mask(data, args.side) & (data.day > eq.TRAIN_DAY_START)
        candidates = np.flatnonzero((data.day >= eq.TRAIN_DAY_START)
                                    & (data.day <= eq.TRAIN_DAY_END)
                                    & (~data.is_boundary) & (~excluded))
        count = len(indices["data"])
        if len(candidates) < count:
            raise ValueError(f"Only {len(candidates)} eligible points for {count} observations")
        indices["data"] = np.sort(rng.choice(candidates, size=count, replace=False))
        if np.any(excluded[indices["data"]]):
            raise RuntimeError("Blocked point leaked into data supervision")
        return indices

    eq.select_training_indices = select_outside_block

    def evaluate(model, data, out_dir, mean, std, device):
        indices = np.flatnonzero((data.day >= eq.TRAIN_DAY_START)
                                  & (data.day <= eq.TRAIN_DAY_END))
        pred = runner.predict_indices(model, data, indices, mean, std, device)
        true = data.temp_c[indices].astype(np.float32)
        held_out = block_mask(data, args.side)[indices] & (data.day[indices] > eq.TRAIN_DAY_START)
        if not held_out.any():
            raise RuntimeError("No held-out rows")
        diff = pred - true
        return {"rmse_C": float(np.sqrt(np.mean(diff ** 2))),
                "mae_C": float(np.mean(np.abs(diff))),
                "relative_l2": relative_l2(pred, true),
                "heldout_relative_l2": relative_l2(pred[held_out], true[held_out]),
                "heldout_rmse_C": float(np.sqrt(np.mean(diff[held_out] ** 2))),
                "heldout_points": int(held_out.sum())}

    runner.evaluate_and_plot = evaluate
    runner.main()
    summaries = sorted(eq.RESULTS_DIR.glob("reconstruction_fdg_aw_*/run_summary.txt"),
                       key=lambda path: path.stat().st_mtime)
    summary_path = summaries[-1]
    summary = dict(line.split("=", 1) for line in summary_path.read_text(encoding="utf-8").splitlines() if "=" in line)
    record = {"seed": args.seed, "method": args.method, "side": args.side,
              "run_dir": str(summary_path.parent),
              **{key: float(summary[key]) for key in
                 ("relative_l2", "heldout_relative_l2", "heldout_rmse_C")},
              "heldout_points": int(summary["heldout_points"])}
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"RESULT_FILE={result_path}", flush=True)


if __name__ == "__main__":
    main()
