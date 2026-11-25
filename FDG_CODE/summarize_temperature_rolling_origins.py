"""Validate and summarize India FDG five-day rolling-origin forecast runs."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "review_temperature_rolling_origins"
SEEDS = (1111, 2222, 3333, 4444, 5555)


def main():
    detailed = []
    aggregate = []
    for origin in range(15, 21):
        origin_rows = []
        for seed in SEEDS:
            record_path = ROOT / f"origin_{origin:02d}" / f"seed_{seed}" / "rolling_result.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            assert record["origin"] == origin and record["seed"] == seed
            summary_path = Path(record["run_dir"]) / "forecast_summary.txt"
            summary = dict(line.split("=", 1) for line in summary_path.read_text(encoding="utf-8").splitlines() if "=" in line)
            assert int(summary["train_day_end"]) == origin
            assert int(summary["forecast_day_start"]) == origin + 1
            assert int(summary["forecast_day_end"]) == origin + 5
            global_row = {"origin": origin, "seed": seed,
                          "relative_l2": float(summary["forecast_relative_l2"]),
                          "rmse_C": float(summary["forecast_rmse_C"])}
            origin_rows.append(global_row)
            assert len(record["daily"]) == 5
            for lead, entry in enumerate(record["daily"], start=1):
                assert int(float(entry["day"])) == origin + lead
                detailed.append({"origin": origin, "seed": seed, "lead": lead,
                                 "day": origin + lead,
                                 "relative_l2": float(entry["relative_l2"]),
                                 "rmse_C": float(entry["rmse_C"]),
                                 "mae_C": float(entry["mae_C"])})
        l2 = np.array([row["relative_l2"] for row in origin_rows])
        rmse = np.array([row["rmse_C"] for row in origin_rows])
        aggregate.append({"origin": origin, "lead": "all_5_days",
                          "relative_l2_mean": l2.mean(), "relative_l2_std": l2.std(ddof=1),
                          "rmse_C_mean": rmse.mean(), "rmse_C_std": rmse.std(ddof=1)})
        for lead in range(1, 6):
            selected = [row for row in detailed if row["origin"] == origin and row["lead"] == lead]
            l2 = np.array([row["relative_l2"] for row in selected])
            rmse = np.array([row["rmse_C"] for row in selected])
            aggregate.append({"origin": origin, "lead": lead,
                              "relative_l2_mean": l2.mean(), "relative_l2_std": l2.std(ddof=1),
                              "rmse_C_mean": rmse.mean(), "rmse_C_std": rmse.std(ddof=1)})
    for filename, rows in (("daily_by_seed.csv", detailed),
                           ("rolling_origin_summary.csv", aggregate)):
        with (ROOT / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Validated", len(detailed), "forecast-day records from", len(SEEDS) * 6, "runs")
    for row in aggregate:
        if row["lead"] == "all_5_days":
            print(f"origin {row['origin']}: L2={row['relative_l2_mean']:.4f} +/- {row['relative_l2_std']:.4f}, "
                  f"RMSE={row['rmse_C_mean']:.3f} +/- {row['rmse_C_std']:.3f} C")


if __name__ == "__main__":
    main()
