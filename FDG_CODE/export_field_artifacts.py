from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

PANEL_TITLE_FONTSIZE = 11
LINE_WIDTH = 1.4


METHOD_LABELS = {
    "base_fixed": "Standard PINN",
    "improved_nn_aw": r"$\phi$-NN + AW",
    "double_gate_aw": "DG-PINN + AW",
    "fourier_double_gate_aw": "FDG-PINN + AW",
    "hard_fourier_double_gate_aw": "HFDG-PINN + AW",
}


def scalar_text(value: np.ndarray | str) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def component(arr: np.ndarray, idx: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[-1] > 1:
        return arr[..., idx]
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def infer_method(stem: str) -> str:
    parts = stem.replace("_field_data", "").split("_LHS_", 1)
    method = parts[1].rsplit("_", 1)[0] if len(parts) == 2 else stem
    return METHOD_LABELS.get(method, method)


def figure_stem(npz_path: Path) -> str:
    return npz_path.stem.replace("_field_data", "")


def default_artifact_dir(npz_path: Path) -> Path:
    eq_dir = npz_path.parent.parent
    return eq_dir / "field_artifacts"


def resolve_time_index(n_time: int, time_index: int | None) -> int:
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    if time_index is None:
        return n_time // 2
    if time_index < 0:
        return max(0, n_time + time_index)
    return min(time_index, n_time - 1)


def apply_1d_x_axis(ax, x: np.ndarray) -> None:
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    ax.set_xlim(x_min, x_max)
    if np.isclose(x_min, 0.0) and np.isclose(x_max, 1.0):
        ticks = np.linspace(0.0, 1.0, 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:.1f}" for v in ticks])
    elif np.isclose(x_min, -1.0) and np.isclose(x_max, 1.0):
        ticks = np.linspace(-1.0, 1.0, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:.1f}" for v in ticks])
    elif np.isclose(x_min, -5.0) and np.isclose(x_max, 5.0):
        ax.set_xticks([-5, -2.5, 0, 2.5, 5])
        ax.set_xticklabels(["-5", "-2.5", "0", "2.5", "5"])
    elif np.isclose(x_min, 0.0) and np.isclose(x_max, np.pi, atol=1.0e-5):
        ticks = [0.0, 1.0, 2.0, float(np.pi)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0.00", "1.00", "2.00", "3.14"])
    else:
        ax.set_xticks(np.linspace(x_min, x_max, 5))


def plot_dual_component_1d_png(npz_path: Path, out_dir: Path, time_index: int | None = None, file_format: str = "png") -> Path:
    data = np.load(npz_path, allow_pickle=True)
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    idx = resolve_time_index(len(t), time_index)

    pred = np.asarray(data["pred"])[idx]
    true = np.asarray(data["true"])[idx]
    error = np.asarray(data["error"])[idx]
    if pred.ndim != 2 or pred.shape[-1] < 2:
        raise ValueError(f"{npz_path} does not contain at least two output components")

    real_series = [pred[:, 0], true[:, 0], error[:, 0]]
    imag_series = [pred[:, 1], true[:, 1], error[:, 1]]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), dpi=300)
    twin_axes = []
    for ax, real_arr, imag_arr, title in zip(
        axes,
        real_series,
        imag_series,
        ("Pred", "True", "Error"),
    ):
        ax_im = ax.twinx()
        twin_axes.append(ax_im)
        ax.plot(x, real_arr, color="#1f77b4", linewidth=LINE_WIDTH)
        ax_im.plot(x, imag_arr, color="#2ca02c", linewidth=LINE_WIDTH)
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel("x")
        apply_1d_x_axis(ax, x)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="y", colors="#1f77b4")
        ax_im.tick_params(axis="y", colors="#2ca02c")
        ax.spines["left"].set_color("#1f77b4")
        ax_im.spines["right"].set_color("#2ca02c")
        if title == "Error":
            ax.set_ylim(-5.0e-4, 5.0e-4)
            ax_im.set_ylim(-5.0e-4, 5.0e-4)
        else:
            ax.set_ylim(0.0, 1.0)
            ax_im.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Real", color="#1f77b4")
    twin_axes[-1].set_ylabel("Imag", color="#2ca02c")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{figure_stem(npz_path)}_mid_t_field_plot.{file_format}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_panel_png(
    npz_path: Path,
    out_dir: Path,
    component_idx: int = 0,
    time_index: int | None = None,
    file_format: str = "png",
) -> Path:
    data = np.load(npz_path, allow_pickle=True)
    plot_style = scalar_text(data["plot_style"]) if "plot_style" in data else ""
    pred = component(data["pred"], component_idx)
    true = component(data["true"], component_idx)
    error = component(data["error"], component_idx)

    if plot_style == "1d_xt":
        x = np.asarray(data["x"])
        t = np.asarray(data["t"])
        idx = resolve_time_index(len(t), time_index)
        pred = pred[idx]
        true = true[idx]
        error = error[idx]
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), dpi=300)
        y_min = float(np.nanmin([pred, true]))
        y_max = float(np.nanmax([pred, true]))
        err_abs = float(np.nanmax(np.abs(error)))
        for ax, arr, title, color in zip(
            axes,
            (pred, true, error),
            ("Pred", "True", "Error"),
            ("#1f77b4", "#2ca02c", "#d62728"),
        ):
            ax.plot(x, arr, color=color, linewidth=LINE_WIDTH)
            ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
            ax.set_xlabel("x")
            apply_1d_x_axis(ax, x)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        if err_abs > 0:
            axes[2].set_ylim(-err_abs, err_abs)
        fig.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{figure_stem(npz_path)}_mid_t_field_plot.{file_format}"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
    else:
        x = np.asarray(data["x"])
        y = np.asarray(data["y"])
        if pred.ndim == 3:
            idx = resolve_time_index(pred.shape[0], time_index)
            pred = pred[idx]
            true = true[idx]
            error = error[idx]
        extent = [float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))]
        x_label, y_label = "x", "y"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), dpi=300)
    for ax, arr, title in zip(axes, (pred, true, error), ("Pred", "True", "Error")):
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title == "Error":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{figure_stem(npz_path)}_field_plot.{file_format}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[..., :3].copy()


def make_gif(npz_path: Path, out_dir: Path, component_idx: int = 0, duration: float = 0.35) -> Path:
    data = np.load(npz_path, allow_pickle=True)
    plot_style = scalar_text(data["plot_style"]) if "plot_style" in data else ""
    pred = component(data["pred"], component_idx)
    true = component(data["true"], component_idx)
    error = component(data["error"], component_idx)
    frames: list[np.ndarray] = []

    if plot_style == "1d_xt":
        x = np.asarray(data["x"])
        t = np.asarray(data["t"])
        pred_min, pred_max = float(np.nanmin([pred, true])), float(np.nanmax([pred, true]))
        err_abs = float(np.nanmax(np.abs(error)))
        for idx, tt in enumerate(t):
            fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=120)
            axes[0].plot(x, pred[idx], color="#1f77b4", linewidth=2)
            axes[1].plot(x, true[idx], color="#2ca02c", linewidth=2)
            axes[2].plot(x, error[idx], color="#d62728", linewidth=2)
            axes[0].set_title(f"Pred t={float(tt):.2f}")
            axes[1].set_title(f"True t={float(tt):.2f}")
            axes[2].set_title("Error")
            axes[0].set_ylim(pred_min, pred_max)
            axes[1].set_ylim(pred_min, pred_max)
            if err_abs > 0:
                axes[2].set_ylim(-err_abs, err_abs)
            for ax in axes:
                ax.set_xlabel("x")
                ax.grid(True, alpha=0.25)
            fig.tight_layout()
            frames.append(save_fig_to_array(fig))
            plt.close(fig)
    else:
        x = np.asarray(data["x"])
        y = np.asarray(data["y"])
        if pred.ndim == 3:
            indices = range(pred.shape[0])
        else:
            indices = [None] * 8
            pred = np.repeat(pred[None, ...], 8, axis=0)
            true = np.repeat(true[None, ...], 8, axis=0)
            error = np.repeat(error[None, ...], 8, axis=0)
        extent = [float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))]
        for idx in indices:
            i = 0 if idx is None else idx
            fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=120)
            for ax, arr, title in zip(axes, (pred[i], true[i], error[i]), ("Pred", "True", "Error")):
                im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap="viridis")
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                if title == "Error":
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            frames.append(save_fig_to_array(fig))
            plt.close(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{figure_stem(npz_path)}_field.gif"
    imageio.mimsave(out_path, frames, duration=duration)
    return out_path


def find_field_data(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PNG/GIF artifacts from FDG *_field_data.npz files.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--pattern", default="*/new_lossplot/*_field_data.npz")
    parser.add_argument("--out", type=Path, default=None, help="Optional common output directory. Default: <eq>/field_artifacts.")
    parser.add_argument("--paper-out", type=Path, default=None, help="Optional path for one copied paper PNG.")
    parser.add_argument("--paper-match", default="", help="Substring used to select the PNG copied to --paper-out.")
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--dual-components", action="store_true", help="For 1D two-output fields, plot components 0 and 1 with twin y-axes.")
    parser.add_argument("--format", choices=("png", "svg"), default="png", help="Snapshot image format.")
    parser.add_argument("--time-index", type=int, default=None, help="Time index for snapshot PNGs; default is the middle saved time.")
    args = parser.parse_args()

    files = find_field_data(args.root, args.pattern)
    if not files:
        raise SystemExit(f"No field_data files found under {args.root} with pattern {args.pattern}")
    copied_paper = False
    for npz_path in files:
        out_dir = args.out if args.out is not None else default_artifact_dir(npz_path)
        if args.dual_components:
            png = plot_dual_component_1d_png(npz_path, out_dir, args.time_index, args.format)
        else:
            png = plot_panel_png(npz_path, out_dir, args.component, args.time_index, args.format)
        gif = make_gif(npz_path, out_dir, args.component)
        if args.paper_out is not None and not copied_paper and args.paper_match and args.paper_match in npz_path.name:
            args.paper_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, args.paper_out)
            copied_paper = True
            print(f"  Paper PNG: {args.paper_out}")
        print(f"{npz_path.name}\n  PNG: {png}\n  GIF: {gif}")


if __name__ == "__main__":
    main()
