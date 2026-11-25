"""Plot the unforced Burgers front and a representative FDG-PINN prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fdg_burgers_front1d_eq as equation


HERE = Path(__file__).resolve().parent
DATA = next((HERE / "fdg_burgers_front1d_focused_eq" / "probe_seed1111" / "fdg_pinn").rglob("*field_data.npz"))
OUTPUT = HERE / "Figures_ND" / "burgers_front_focused_nd.svg"

with np.load(DATA) as field:
    x = field["x"]
    true_end = field["true"][-1, :, 0]
    pred_end = field["pred"][-1, :, 0]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True, sharey=True)
for t, color in [(0.0, "#455a64"), (0.25, "#238b7b"),
                 (0.5, "#db7d30"), (1.0, "#b5233d")]:
    axes[0].plot(x, equation.cole_hopf_reference(x, t),
                 color=color, linewidth=1.3, label=rf"$t={t:g}$")
axes[0].set_title("Front evolution")
axes[0].legend(frameon=False, loc="upper right", ncol=2,
               columnspacing=0.6, handlelength=1.5)
axes[1].plot(x, true_end, color="#262626", linewidth=1.5, label="Reference")
axes[1].plot(x, pred_end, color="#1878b8", linewidth=1.2,
             linestyle="--", label="FDG-PINN")
axes[1].set_title(r"Prediction at $t=1$ (seed 1111)")
axes[1].legend(frameon=False, loc="upper right", handlelength=1.5)
for axis in axes:
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("$x$")
    axis.set_xticks([-1, -0.5, 0, 0.5, 1])
    axis.grid(color="#dddddd", linewidth=0.4)
axes[0].set_ylabel("$u(x,t)$")
fig.tight_layout(w_pad=1.1)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
plt.close(fig)
print(OUTPUT)
