"""Summarize five seed-1111..5555 origin-20 forecasts for manuscript Table 10."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "origin_20"
SEEDS = (1111, 2222, 3333, 4444, 5555)
FIELDS = ("mse_C2", "rmse_C", "mae_C", "relative_l2")
records = {}
for seed in SEEDS:
    files = list((ROOT / f"seed_{seed}").glob("*/forecast_metrics_by_day.csv"))
    if len(files) != 1:
        raise RuntimeError(f"seed {seed}: expected one daily CSV, found {len(files)}")
    with files[0].open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    records[seed] = {int(row["day"]): row for row in rows if 21 <= row["day"] <= 25}
    if set(records[seed]) != set(range(21, 26)):
        raise RuntimeError(f"seed {seed} lacks one or more forecast days")

combined = {}
for seed in SEEDS:
    rows = [records[seed][day] for day in range(21, 26)]
    weights = np.array([row["points"] for row in rows])
    mse = np.average([row["mse_C2"] for row in rows], weights=weights)
    mae = np.average([row["mae_C"] for row in rows], weights=weights)
    true_sq = sum(row["mse_C2"] * row["points"] / row["relative_l2"] ** 2 for row in rows)
    err_sq = sum(row["mse_C2"] * row["points"] for row in rows)
    combined[seed] = dict(mse_C2=mse, rmse_C=np.sqrt(mse), mae_C=mae,
                          relative_l2=np.sqrt(err_sq / true_sq))

output = Path(__file__).resolve().parent / "origin20_five_seed_table10.csv"
with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("period", *(f"{field}_{stat}" for field in FIELDS for stat in ("mean", "std"))))
    for day in (*range(21, 26), "Days 21--25"):
        values = combined if isinstance(day, str) else {seed: records[seed][day] for seed in SEEDS}
        cells = [day]
        for field in FIELDS:
            sample = np.array([values[seed][field] for seed in SEEDS])
            cells.extend((float(sample.mean()), float(sample.std(ddof=1))))
        writer.writerow(cells)
        print(day, " | ".join(f"{field}: {cells[1 + i*2]:.5f} ± {cells[2 + i*2]:.5f}" for i, field in enumerate(FIELDS)))
print(output)
