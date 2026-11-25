"""Refresh the two manuscript daily plots using saved seed-1111 CSV data."""
import csv
from pathlib import Path

from redraw_temperature_seed1111_figures import plot_daily

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "review_temperature_seed1111_figures"
DEST = HERE / "Figures_ND"

for region in ("india", "huabei"):
    with (SOURCE / f"{region}_daily_reconstruction_seed1111.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    target = DEST / f"{region}_mean_l2_combined_nd_seed1111"
    plot_daily(rows, target)
    print(target.with_suffix(".svg"))
