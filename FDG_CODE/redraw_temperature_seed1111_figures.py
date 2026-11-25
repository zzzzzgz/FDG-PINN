"""Rebuild the four manuscript temperature figures from paired seed-1111 runs."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.parent
MANUSCRIPT = HERE / "Figures_ND"
OUTPUT = HERE / "review_temperature_seed1111_figures"
SEED = 1111


def record(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(checkpoint_path, model_type, eq, dtype, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    kwargs = {"input_dim": eq.INPUT_DIM, "output_dim": eq.OUTPUT_DIM,
              "hidden_width": state["hidden_width"], "hidden_depth": state["hidden_depth"]}
    if "fourier_freqs" in state and model_type.__name__ == "FourierDoubleGateNet":
        kwargs["freqs"] = state["fourier_freqs"]
    model = model_type(**kwargs).to(device=device, dtype=dtype)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def plot_daily(rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "svg.fonttype": "none",
                         "axes.linewidth": 0.6, "lines.linewidth": 1.4})
    days = np.asarray([row["day"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), constrained_layout=True)
    styles = (("Data-driven PINN", "mlp", "#d55e00", "s"),
              ("FDG-PINN", "fdg", "#0072b2", "^"))
    axes[0].plot(days, [row["true_mean_C"] for row in rows],
                 color="#202124", marker="o", markersize=2.6, label="True")
    for label, key, color, marker in styles:
        axes[0].plot(days, [row[f"{key}_mean_C"] for row in rows], color=color,
                     marker=marker, markersize=2.4, label=label)
        axes[1].plot(days, [row[f"{key}_relative_l2"] for row in rows], color=color,
                     marker=marker, markersize=2.4, label=label)
    axes[0].set_ylabel("Mean temperature ($^\\circ$C)")
    axes[1].set_ylabel(r"Relative $L_2$ error")
    axes[1].set_ylim(bottom=0)
    for index, ax in enumerate(axes):
        ax.set(xlim=(1, 20), xlabel="Day")
        ax.set_xticks((1, 5, 10, 15, 20))
        ax.tick_params(length=2.5, width=0.6, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.01, 0.98, "(a)" if index == 0 else "(b)",
                transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        ax.legend(frameon=False, loc="best")
    for suffix in ("svg", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=("india", "huabei"), required=True)
    args = parser.parse_args()
    india = args.region == "india"
    region_dir = PUBLISHED / ("Temperature_forecast_india" if india else "Temperature_forecast_huabei_50day")
    sys.path.insert(0, str(region_dir))
    import temperature_fdg_eq as eq
    import FDG_runner_temperature as fdg_runner
    import PINN_runner_temperature_fixed as mlp_runner

    if india:
        fdg_record = record(HERE / "review_temperature_ff_mlp" / f"seed_{SEED}" / "fdg" / "reconstruction_result.json")
        mlp_record = record(HERE / "review_temperature_remaining_baselines" / f"seed_{SEED}" / "pinn" / "reconstruction_result.json")
    else:
        root = HERE / "review_temperature_huabei" / f"seed_{SEED}"
        fdg_record = record(root / "fdg" / "reconstruction_result.json")
        mlp_record = record(root / "pinn" / "reconstruction_result.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = eq.load_temperature_data()
    fdg_dir, mlp_dir = Path(fdg_record["run_dir"]), Path(mlp_record["run_dir"])
    fdg, fdg_state = load_model(fdg_dir / "fdg_temperature_model.pt",
                                fdg_runner.FourierDoubleGateNet, eq, fdg_runner.DTYPE, device)
    mlp, mlp_state = load_model(mlp_dir / "pinn_temperature_model.pt",
                                mlp_runner.PlainPINN, eq, mlp_runner.DTYPE, device)
    rows = []
    predictions = {"fdg": [], "mlp": []}
    truths = []
    for day in range(1, 21):
        indices = eq.day_indices(data, day)
        true = data.temp_c[indices].astype(np.float64)
        day_row = {"day": day, "true_mean_C": float(true.mean())}
        truths.append(true)
        for key, runner, model, state in (
            ("fdg", fdg_runner, fdg, fdg_state),
            ("mlp", mlp_runner, mlp, mlp_state),
        ):
            pred = runner.predict_indices(model, data, indices,
                                          state["temp_mean_C"], state["temp_std_C"], device).astype(np.float64)
            predictions[key].append(pred)
            day_row[f"{key}_mean_C"] = float(pred.mean())
            day_row[f"{key}_relative_l2"] = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
        rows.append(day_row)
    true_all = np.concatenate(truths)
    for key, saved in (("fdg", fdg_record), ("mlp", mlp_record)):
        pred_all = np.concatenate(predictions[key])
        relative_l2 = float(np.linalg.norm(pred_all - true_all) / np.linalg.norm(true_all))
        if abs(relative_l2 - saved["relative_l2"]) > 2e-6:
            raise RuntimeError(f"{args.region} {key}: checkpoint metric mismatch: {relative_l2} vs {saved['relative_l2']}")

    OUTPUT.mkdir(exist_ok=True)
    MANUSCRIPT.mkdir(exist_ok=True)
    stem = "india" if india else "huabei"
    daily_source = OUTPUT / f"{stem}_mean_l2_seed1111"
    with (OUTPUT / f"{stem}_daily_reconstruction_seed1111.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_daily(rows, daily_source)
    shutil.copyfile(daily_source.with_suffix(".svg"), MANUSCRIPT / f"{stem}_mean_l2_combined_nd_seed1111.svg")

    field_source = fdg_dir / "temperature_reconstruction_days_05_10_20.svg"
    if not field_source.exists():
        raise FileNotFoundError(field_source)
    shutil.copyfile(field_source, MANUSCRIPT / f"{stem}_temperature_reconstruction_days_05_10_20_seed1111.svg")
    print(f"Verified seed {SEED} checkpoint metrics and saved two {stem} manuscript figures")


if __name__ == "__main__":
    main()
