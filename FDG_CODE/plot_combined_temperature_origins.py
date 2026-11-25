"""Combine two rolling-origin summaries without duplicating their shared origin."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "review_temperature_rolling_origins"
EARLY = ROOT / "forecast_to_day30_origins_04_09_14_19" / "five_seed_daily_summary.csv"
LATE = ROOT / "forecast_to_day30" / "five_seed_daily_summary.csv"
OUTPUT = ROOT / "forecast_to_day30_combined"
ORIGINS = (4, 9, 14, 15, 16, 17, 18, 19)


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {(int(row["origin"]), int(row["day"])):
                (float(row["mean_relative_l2"]), float(row["sample_std_relative_l2"]))
                for row in csv.DictReader(stream)}


def main():
    early = read_summary(EARLY)
    late = read_summary(LATE)
    shared = set(early) & set(late)
    if len(shared) != 11 or any(early[key] != late[key] for key in shared):
        raise RuntimeError("The shared day-19 training origin differs between source summaries")
    combined = {**late, **early}
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "five_seed_daily_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("train_through_day", "forecast_start_day", "forecast_day",
                         "lead_days", "mean_relative_l2", "sample_std_relative_l2"))
        for origin in ORIGINS:
            for day in range(origin + 1, 31):
                mean, std = combined[(origin, day)]
                writer.writerow((origin, origin + 1, day, day - origin, mean, std))

    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(11.2, 5.0), layout="constrained")
    for origin, color in zip(ORIGINS, colors):
        days = np.arange(origin + 1, 31)
        values = np.asarray([combined[(origin, day)] for day in days])
        means, stds = values[:, 0], values[:, 1]
        ax.plot(days, means, color=color, linewidth=1.6, marker="o", markersize=2.5,
                label=f"Starts day {origin + 1}")
        ax.fill_between(days, np.maximum(0, means - stds), means + stds,
                        color=color, alpha=0.09, linewidth=0)
    ax.set(xlim=(4.7, 30.3), xlabel="Forecast day", ylabel=r"Relative $L_2$ error")
    ax.set_xticks(range(5, 31))
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / f"combined_rolling_origins_to_day30.{suffix}", dpi=250)
    plt.close(fig)
    print(f"Combined {len(ORIGINS)} distinct curves; {len(shared)} shared day-19 values agree exactly")
    print(OUTPUT)


if __name__ == "__main__":
    main()
