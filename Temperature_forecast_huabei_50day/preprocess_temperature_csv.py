"""
Preprocess GEE temperature samples for FDG-PINN experiments.

Input:
    FDG_PINN_HUABEI_50DAY_Temperature_trim_lat.csv

Output:
    FDG_PINN_HUABEI_50DAY_Temperature_preprocessed.csv
    FDG_PINN_HUABEI_50DAY_Temperature_preprocessed_meta.txt

The output CSV keeps only:
    system:index, day_index, lat, lon, temp_C

In the output, lat and lon are normalized to [0, 1]. The original coordinate
ranges are written to the metadata text file for later inverse transforms.
The raw GEE day_index is shifted from 0-based to 1-based numbering.
"""

from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "FDG_PINN_HUABEI_50DAY_Temperature_trim_lat.csv"
OUTPUT_CSV = BASE_DIR / "FDG_PINN_HUABEI_50DAY_Temperature_preprocessed.csv"
META_TXT = BASE_DIR / "FDG_PINN_HUABEI_50DAY_Temperature_preprocessed_meta.txt"

KEEP_COLUMNS = ["system:index", "day_index", "lat", "lon", "temp_C"]


def _to_float(value: str, column: str, row_number: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in column {column!r} at row {row_number}: {value!r}") from exc


def scan_ranges(input_csv: Path) -> dict[str, float | int]:
    lat_min = float("inf")
    lat_max = float("-inf")
    lon_min = float("inf")
    lon_max = float("-inf")
    temp_min = float("inf")
    temp_max = float("-inf")
    day_min = float("inf")
    day_max = float("-inf")
    rows = 0

    with input_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [col for col in KEEP_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            lat = _to_float(row["lat"], "lat", row_number)
            lon = _to_float(row["lon"], "lon", row_number)
            temp = _to_float(row["temp_C"], "temp_C", row_number)
            day = _to_float(row["day_index"], "day_index", row_number)

            lat_min = min(lat_min, lat)
            lat_max = max(lat_max, lat)
            lon_min = min(lon_min, lon)
            lon_max = max(lon_max, lon)
            temp_min = min(temp_min, temp)
            temp_max = max(temp_max, temp)
            day_min = min(day_min, day)
            day_max = max(day_max, day)
            rows += 1

    if rows == 0:
        raise ValueError(f"No data rows found in {input_csv}")
    if lat_max == lat_min or lon_max == lon_min:
        raise ValueError("Latitude or longitude range is zero; cannot normalize coordinates.")

    return {
        "rows": rows,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "day_min": day_min,
        "day_max": day_max,
    }


def preprocess(input_csv: Path, output_csv: Path, meta_txt: Path) -> dict[str, float | int]:
    stats = scan_ranges(input_csv)
    lat_min = float(stats["lat_min"])
    lat_max = float(stats["lat_max"])
    lon_min = float(stats["lon_min"])
    lon_max = float(stats["lon_max"])
    lat_scale = lat_max - lat_min
    lon_scale = lon_max - lon_min

    with input_csv.open("r", newline="", encoding="utf-8-sig") as src, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=KEEP_COLUMNS)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=2):
            lat = _to_float(row["lat"], "lat", row_number)
            lon = _to_float(row["lon"], "lon", row_number)
            day = _to_float(row["day_index"], "day_index", row_number)
            temp = _to_float(row["temp_C"], "temp_C", row_number)

            lat_norm = (lat - lat_min) / lat_scale
            lon_norm = (lon - lon_min) / lon_scale
            output_day = day + 1.0

            writer.writerow(
                {
                    "system:index": row["system:index"],
                    "day_index": f"{output_day:.0f}" if output_day.is_integer() else f"{output_day:.10g}",
                    "lat": f"{lat_norm:.10f}",
                    "lon": f"{lon_norm:.10f}",
                    "temp_C": f"{temp:.10f}",
                }
            )

    with meta_txt.open("w", encoding="utf-8") as f:
        f.write("FDG-PINN HUABEI temperature preprocessing metadata\n")
        f.write(f"input_csv={input_csv.name}\n")
        f.write(f"output_csv={output_csv.name}\n")
        for key in [
            "rows",
            "day_min",
            "day_max",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "temp_min",
            "temp_max",
        ]:
            f.write(f"{key}={stats[key]}\n")
        f.write(f"output_day_min={float(stats['day_min']) + 1.0}\n")
        f.write(f"output_day_max={float(stats['day_max']) + 1.0}\n")
        f.write("day_output=day_original+1\n")
        f.write("lat_output=(lat_original-lat_min)/(lat_max-lat_min)\n")
        f.write("lon_output=(lon_original-lon_min)/(lon_max-lon_min)\n")

    return stats


def main() -> None:
    stats = preprocess(INPUT_CSV, OUTPUT_CSV, META_TXT)
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Meta:   {META_TXT}")
    print(f"Rows:   {stats['rows']}")
    print(f"Input days:  {stats['day_min']} to {stats['day_max']}")
    print(f"Output days: {float(stats['day_min']) + 1.0} to {float(stats['day_max']) + 1.0}")
    print(f"Lat:    {stats['lat_min']} to {stats['lat_max']} -> [0, 1]")
    print(f"Lon:    {stats['lon_min']} to {stats['lon_max']} -> [0, 1]")
    print(f"Temp C: {stats['temp_min']} to {stats['temp_max']}")


if __name__ == "__main__":
    main()
