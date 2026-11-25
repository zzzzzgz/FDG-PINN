"""Re-evaluate saved rolling-origin FDG models through a chosen calendar day."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
INDIA = HERE.parent / "Temperature_forecast_india"
ROOT = HERE / "review_temperature_rolling_origins"
DATA_50 = HERE.parent / "Temperature_forecast_india_50day" / "FDG_PINN_India_50DAY_Temperature_preprocessed.csv"
ORIGINS = range(15, 21)
SEEDS = (1111, 2222, 3333, 4444, 5555)
sys.path.insert(0, str(INDIA))

import temperature_fdg_eq as eq
import FDG_runner_temperature_forecast as runner


def load_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = runner.FourierDoubleGateNet(
        input_dim=eq.INPUT_DIM, output_dim=eq.OUTPUT_DIM,
        hidden_width=state["hidden_width"], hidden_depth=state["hidden_depth"],
        freqs=state["fourier_freqs"],
    ).to(device=device, dtype=runner.DTYPE)
    model.load_state_dict(state["model_state_dict"])
    return model, state


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-day", type=int, default=25, choices=range(25, 51))
    parser.add_argument("--origins", type=int, nargs="+", default=list(ORIGINS))
    args = parser.parse_args()
    end_day = args.end_day
    origins = tuple(args.origins)
    if not origins or len(set(origins)) != len(origins) or min(origins) < 2 or max(origins) >= end_day:
        parser.error("Origins must be unique days from 2 through end-day minus one")
    suffix = "" if origins == tuple(ORIGINS) else "_origins_" + "_".join(f"{day:02d}" for day in origins)
    output = ROOT / f"forecast_to_day{end_day}{suffix}"
    data = eq.load_temperature_data(DATA_50 if end_day > 25 else eq.DATA_CSV)
    if data.day.max() < end_day:
        raise ValueError(f"Dataset ends at day {data.day.max()}, before requested day {end_day}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output.mkdir(exist_ok=True)
    per_seed = []
    make_inputs = eq.make_inputs
    for origin in origins:
        eq.make_inputs = partial(make_inputs, train_day_start=1, train_day_end=origin)
        for seed in SEEDS:
            record_path = ROOT / f"origin_{origin:02d}" / f"seed_{seed}" / "rolling_result.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            run_dir = Path(record["run_dir"])
            checkpoint_path = run_dir / "fdg_temperature_forecast_model.pt"
            model, state = load_model(checkpoint_path, device)
            for day in range(origin + 1, end_day + 1):
                indices = eq.day_indices(data, day)
                predicted = runner.predict_indices(model, data, indices,
                                                   state["temp_mean_C"], state["temp_std_C"], device)
                truth = data.temp_c[indices].astype(np.float32)
                value = float(np.linalg.norm(predicted - truth) / np.linalg.norm(truth))
                old = next((row for row in record["daily"] if int(float(row["day"])) == day), None)
                if old is not None and not np.isclose(value, float(old["relative_l2"]), rtol=1e-5, atol=1e-6):
                    raise RuntimeError(f"Saved-metric mismatch: origin={origin}, seed={seed}, day={day}: "
                                       f"{value} vs {old['relative_l2']}")
                per_seed.append({"origin": origin, "seed": seed, "day": day,
                                 "lead_days": day - origin, "relative_l2": value})
            print(f"Verified and evaluated origin {origin}, seed {seed} to day {end_day}", flush=True)
    write_csv(output / "per_seed_daily_relative_l2.csv", per_seed)
    summary = []
    for origin in origins:
        for day in range(origin + 1, end_day + 1):
            values = np.asarray([row["relative_l2"] for row in per_seed
                                 if row["origin"] == origin and row["day"] == day], dtype=np.float64)
            if values.size != len(SEEDS):
                raise RuntimeError(f"Missing seed: origin={origin}, day={day}")
            summary.append({"origin": origin, "day": day, "lead_days": day - origin,
                            "mean_relative_l2": float(values.mean()),
                            "sample_std_relative_l2": float(values.std(ddof=1))})
    write_csv(output / "five_seed_daily_summary.csv", summary)

    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(11.2 if end_day > 30 else 8.4, 5.0 if end_day > 30 else 4.5),
                           layout="constrained")
    for origin, color in zip(origins, colors):
        selected = [row for row in summary if row["origin"] == origin]
        days = np.asarray([row["day"] for row in selected])
        means = np.asarray([row["mean_relative_l2"] for row in selected])
        stds = np.asarray([row["sample_std_relative_l2"] for row in selected])
        ax.plot(days, means, color=color, marker="o", markersize=3.4,
                markevery=5 if end_day > 30 else 1,
                linewidth=1.7, label=f"Training ends: day {origin}")
        ax.fill_between(days, np.maximum(0, means - stds), means + stds,
                        color=color, alpha=0.13, linewidth=0)
    ax.set(xlim=(min(origins) + 0.7, end_day + 0.3), xlabel="Forecast day", ylabel=r"Relative $L_2$ error")
    tick_step = 5 if end_day > 30 else 1
    ax.set_xticks(range(min(origins) + 1, end_day + 1, tick_step))
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(ncol=min(4 if end_day > 30 else 3, len(origins)), fontsize=8.5,
              frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"rolling_origins_to_day{end_day}.{suffix}", dpi=250)
    plt.close(fig)
    print(f"Saved {len(per_seed)} per-seed daily values and {len(summary)} five-seed summaries to {output}")
    for origin in origins:
        final = next(row for row in summary if row["origin"] == origin and row["day"] == end_day)
        print(f"Origin {origin}, day {end_day}: {final['mean_relative_l2']:.5f} "
              f"+/- {final['sample_std_relative_l2']:.5f}")


if __name__ == "__main__":
    main()
