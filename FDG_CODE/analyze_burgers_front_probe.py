"""Diagnose time-resolved and sine-mode errors from saved front fields."""

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("pinn", "ff_pinn", "fdg_pinn")


def sine_components(values):
    interior = values[1:-1]
    odd = np.concatenate(([0.0], interior, [0.0], -interior[::-1]))
    spectrum = np.fft.fft(odd)
    low_spectrum = np.zeros_like(spectrum)
    low_spectrum[1:5] = spectrum[1:5]
    low_spectrum[-4:] = spectrum[-4:]
    low = np.fft.ifft(low_spectrum).real[1:len(values) - 1]
    return low, interior - low


def relative_error(pred, true):
    denominator = np.linalg.norm(true)
    return float(np.linalg.norm(pred - true) / denominator) if denominator > 1e-12 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--equation", default="fdg_burgers_front1d_eq",
                        choices=("fdg_burgers_front1d_eq", "fdg_burgers_front1d_focused_eq"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / args.equation / f"probe_seed{args.seed}"
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for method in METHODS:
        source = next((root / method).rglob("*field_data.npz"))
        with np.load(source) as data:
            times = data["t"]
            predicted = data["pred"][:, :, 0]
            truth = data["true"][:, :, 0]
        errors = []
        for t, pred, true in zip(times, predicted, truth):
            pred_low, pred_high = sine_components(pred)
            true_low, true_high = sine_components(true)
            row = {
                "method": method,
                "time": float(t),
                "relative_l2": relative_error(pred, true),
                "low_modes_1_to_4_l2": relative_error(pred_low, true_low),
                "high_modes_5_plus_l2": relative_error(pred_high, true_high),
                "true_high_energy_fraction": float(np.linalg.norm(true_high) / np.linalg.norm(true[1:-1])),
            }
            rows.append(row)
            errors.append(row["relative_l2"])
        ax.plot(times, errors, marker="o", markersize=3, label=method.replace("_", "-").upper())
    with (root / f"time_frequency_errors_seed{args.seed}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ax.set(xlabel="Time", ylabel="Relative $L_2$ error", yscale="log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / f"time_errors_seed{args.seed}.png", dpi=180)
    plt.close(fig)
    for row in rows:
        if abs(row["time"] - 0.5) < 0.06 or abs(row["time"] - 1.0) < 0.01:
            print(row)


if __name__ == "__main__":
    main()
