import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import time
import copy
import importlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from openpyxl import Workbook, load_workbook
except Exception:
    Workbook = None
    load_workbook = None

# =========================================================
# 0) User switches
# =========================================================
# Main reproducibility runner for the manufactured PDE experiments in the
# paper. Equation modules provide residuals, IC/BC samplers, exact solutions,
# and plotting metadata; this file provides shared models, weighting schemes,
# optimizer stages, and artifact saving.
sample_mode = "LHS"   # "random" or "LHS"
RUN_ALL = True

# FDG/HFDG formal runner:
# One equation is run with the main comparison methods. Table data, loss-plot
# data, and prediction-field data are saved under <equation>/new_lossplot/.
#
# combos: (equation_module_name, model_kind, weight_mode)
combos = [
    ("fdg_transport1d_eq", "base", "fixed"),
    ("fdg_transport1d_eq", "improved_nn", "aw"),
    ("fdg_transport1d_eq", "double_gate", "aw"),
    ("fdg_transport1d_eq", "fourier_double_gate", "aw"),
]

# single-run fallback (only used when RUN_ALL=False)
run_equation = "fdg_transport1d_eq"
MODEL_KIND = "fourier_double_gate"     # "base", "improved_nn", "double_gate", "attn_res", "res_gate", "fourier_*", "adaptive_fourier_*", "hard_*"
WEIGHT_MODE = "aw"             # "fixed", "aw", "improved_aw1"

# =========================================================
# 1) Optimization
# =========================================================
steps_adam = 2000
steps_lbfgs = 1000
print_every = 1000
lr_theta = 1e-3
lr_q = 2e-4
weight_decay_theta = 1e-6
clip_grad_norm = 100.0

# Sampling sizes
N_f = 10000
N_ic = 2000
N_bc = 2000

# Model size
hidden = 96
n_hidden_layers = 4
fourier_freqs = (1.0, 2.0, 4.0, 8.0)

# Visualization / evaluation
Nx, Ny = 140, 140
n_frames = 20
gif_duration = 0.12
n_test = 10
LOSS_LOG_EVERY = 10
L2_LOG_EVERY = 10
SAVE_ARTIFACTS = True
WRITE_LOG_XLSX = False
RESTORE_ADAM_BY_L2 = True
USE_ANCHOR_LOSS = False
N_anchor = 1024
ANCHOR_WEIGHT_FIXED = 1.0
RESET_SEED_EACH_RUN = True
DETERMINISTIC_RUN = True
SAVE_PREDICTION_DATA = True
APPEND_TEX = False

# Fixed weights
W_PDE_FIXED = 1.0
W_IC_FIXED = 1.0
W_BC_FIXED = 1.0

# improved_aw1 params
IMB_GAP_LOG10 = 3.0
IMB_ALPHA = 0.6  # Strength of the imbalance compensation update.
COMP_GAMMA = 2  # Sensitivity of the compensation factor to loss imbalance.
COMP_CLAMP_MIN = 0.25
COMP_CLAMP_MAX = 8.0
COMP_EMA = 0.95

# L-BFGS options
LBFGS_LR = 0.5
LBFGS_HISTORY = 50
LBFGS_LINE_SEARCH = "strong_wolfe"

# =========================================================
# 2) Device / seeds / paths
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2026)
np.random.seed(2026)

print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_XLSX_PATH = os.path.join(ROOT_DIR, "log.xlsx")


def env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name, default):
    val = os.environ.get(name)
    return default if val is None or val == "" else int(val)


def env_float(name, default):
    val = os.environ.get(name)
    return default if val is None or val == "" else float(val)


def env_float_tuple(name, default):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return tuple(float(x.strip()) for x in val.split(",") if x.strip())


def env_torch_dtype(name, default="float64"):
    val = os.environ.get(name, default).strip().lower()
    if val in ("float64", "double", "64"):
        return torch.float64
    if val in ("float32", "single", "32"):
        return torch.float32
    raise ValueError(f"Unsupported {name}={val}; use float64 or float32")


RUN_ALL = env_bool("PINN_RUN_ALL", RUN_ALL)
TORCH_DTYPE = env_torch_dtype("PINN_DTYPE", "float64")
torch.set_default_dtype(TORCH_DTYPE)
print("Torch dtype:", TORCH_DTYPE)
DETERMINISTIC_RUN = env_bool("PINN_DETERMINISTIC", DETERMINISTIC_RUN)
if DETERMINISTIC_RUN:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
        print("Deterministic algorithms: enabled")
    except Exception as e:
        print(f"Deterministic algorithms could not be fully enabled: {e}")
seed = env_int("PINN_SEED", 2026)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
run_equation = os.environ.get("PINN_EQUATION", run_equation)
MODEL_KIND = os.environ.get("PINN_MODEL_KIND", MODEL_KIND)
WEIGHT_MODE = os.environ.get("PINN_WEIGHT_MODE", WEIGHT_MODE)
formal_equation = os.environ.get("PINN_FDG_EQUATION", os.environ.get("PINN_LOSSPLOT_EQUATION", run_equation))
METHOD_SET = os.environ.get("PINN_METHOD_SET", "auto").strip().lower()
steps_adam = env_int("PINN_STEPS_ADAM", steps_adam)
steps_lbfgs = env_int("PINN_STEPS_LBFGS", steps_lbfgs)
print_every = env_int("PINN_PRINT_EVERY", print_every)
LOSS_LOG_EVERY = env_int("PINN_LOSS_LOG_EVERY", LOSS_LOG_EVERY)
L2_LOG_EVERY = env_int("PINN_L2_LOG_EVERY", L2_LOG_EVERY)
N_f = env_int("PINN_N_F", N_f)
N_ic = env_int("PINN_N_IC", N_ic)
N_bc = env_int("PINN_N_BC", N_bc)
Nx = env_int("PINN_NX", Nx)
Ny = env_int("PINN_NY", Ny)
n_test = env_int("PINN_N_TEST", n_test)
n_frames = env_int("PINN_N_FRAMES", n_frames)
hidden = env_int("PINN_HIDDEN", hidden)
n_hidden_layers = env_int("PINN_DEPTH", n_hidden_layers)
fourier_freqs = env_float_tuple("PINN_FOURIER_FREQS", fourier_freqs)
lr_theta = env_float("PINN_LR_THETA", lr_theta)
lr_q = env_float("PINN_LR_Q", lr_q)
weight_decay_theta = env_float("PINN_WEIGHT_DECAY", weight_decay_theta)
clip_grad_norm = env_float("PINN_CLIP_GRAD_NORM", clip_grad_norm)
SAVE_ARTIFACTS = env_bool("PINN_SAVE_ARTIFACTS", SAVE_ARTIFACTS)
WRITE_LOG_XLSX = env_bool("PINN_WRITE_LOG", WRITE_LOG_XLSX)
RESTORE_ADAM_BY_L2 = env_bool("PINN_RESTORE_ADAM_BY_L2", RESTORE_ADAM_BY_L2)
USE_ANCHOR_LOSS = env_bool("PINN_USE_ANCHOR", USE_ANCHOR_LOSS)
N_anchor = env_int("PINN_N_ANCHOR", N_anchor)
ANCHOR_WEIGHT_FIXED = env_float("PINN_ANCHOR_WEIGHT", ANCHOR_WEIGHT_FIXED)
RESET_SEED_EACH_RUN = env_bool("PINN_RESET_SEED_EACH_RUN", RESET_SEED_EACH_RUN)
SAVE_PREDICTION_DATA = env_bool("PINN_SAVE_PREDICTION_DATA", SAVE_PREDICTION_DATA)
APPEND_TEX = env_bool("PINN_APPEND_TEX", APPEND_TEX)
W_PDE_FIXED = env_float("PINN_W_PDE_FIXED", W_PDE_FIXED)
W_IC_FIXED = env_float("PINN_W_IC_FIXED", W_IC_FIXED)
W_BC_FIXED = env_float("PINN_W_BC_FIXED", W_BC_FIXED)
IMB_GAP_LOG10 = env_float("PINN_IMB_GAP_LOG10", IMB_GAP_LOG10)
IMB_ALPHA = env_float("PINN_IMB_ALPHA", IMB_ALPHA)
COMP_GAMMA = env_float("PINN_COMP_GAMMA", COMP_GAMMA)
COMP_CLAMP_MIN = env_float("PINN_COMP_CLAMP_MIN", COMP_CLAMP_MIN)
COMP_CLAMP_MAX = env_float("PINN_COMP_CLAMP_MAX", COMP_CLAMP_MAX)
COMP_EMA = env_float("PINN_COMP_EMA", COMP_EMA)


def default_combos_for_equation(eq_name):
    eq = importlib.import_module(eq_name)
    methods = [
        (eq_name, "base", "fixed"),
        (eq_name, "improved_nn", "aw"),
        (eq_name, "double_gate", "aw"),
        (eq_name, "fourier_double_gate", "aw"),
    ]
    if METHOD_SET in ("hfdg", "auto") and hasattr(eq, "output_transform"):
        methods.append((eq_name, "hard_fourier_double_gate", "aw"))
    if METHOD_SET == "fdg":
        return [m for m in methods if not m[1].startswith("hard_")]
    if METHOD_SET == "hfdg":
        return methods
    return methods


combos = default_combos_for_equation(formal_equation)


def now_tag():
    return datetime.now().strftime("%m%d%H%M")


# =========================================================
# 3) Basic helpers
# =========================================================
def grad(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def relative_l2_error(pred, true, eps=1e-12):
    pred = np.asarray(pred)
    true = np.asarray(true)
    num = np.sqrt(np.mean((pred - true) ** 2))
    den = np.sqrt(np.mean(true ** 2)) + eps
    return float(num / den)


def mse_error(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float(np.mean((pred - true) ** 2))


def ensure_log_xlsx():
    if Workbook is None:
        return None
    if not os.path.exists(LOG_XLSX_PATH):
        wb = Workbook()
        ws = wb.active
        ws.title = "log"
        fields = [
            "equation",
            "model_kind",
            "weight_mode",
            "sample_mode",
            "active_terms",
            "anchor_points",
            "time(mmddHHMM)",
            "training_time_sec",
            "avg_time_avg_L2_error",
            "global_relative_L2_error",
            "MSE_error",
            "printed_loss_lines",
            "train_relative_L2_error",
            "train_MSE_error",
        ]
        for i, f in enumerate(fields, start=1):
            ws.cell(row=i, column=1).value = f
        wb.save(LOG_XLSX_PATH)
    return LOG_XLSX_PATH


def append_log_column(record: dict):
    if load_workbook is None:
        print("openpyxl not available, skip log.xlsx")
        return
    fp = ensure_log_xlsx()
    wb = load_workbook(fp)
    ws = wb["log"]
    col = ws.max_column + 1
    keys = [
        "equation",
        "model_kind",
        "weight_mode",
        "sample_mode",
        "active_terms",
        "anchor_points",
        "time_tag",
        "training_time",
        "avg_l2",
        "global_l2",
        "mse",
        "loss_lines",
        "train_l2",
        "train_mse",
    ]
    labels = [
        "equation",
        "model_kind",
        "weight_mode",
        "sample_mode",
        "active_terms",
        "anchor_points",
        "time(mmddHHMM)",
        "training_time_sec",
        "avg_time_avg_L2_error",
        "global_relative_L2_error",
        "MSE_error",
        "printed_loss_lines",
        "train_relative_L2_error",
        "train_MSE_error",
    ]
    for r, label in enumerate(labels, start=1):
        if ws.cell(row=r, column=1).value in (None, ""):
            ws.cell(row=r, column=1).value = label
    for r, k in enumerate(keys, start=1):
        ws.cell(row=r, column=col).value = record.get(k, "")
    wb.save(fp)


# =========================================================
# 4) Models
# =========================================================
class BaseMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=96, depth=5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, depth=2, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def input_bounds_from_eq(eq):
    if eq.INPUT_DIM == 2 and getattr(eq, "PLOT_STYLE", "") == "1d_xt":
        names = [("x_min", "x_max"), ("t_min", "t_max")]
    elif eq.INPUT_DIM == 2:
        names = [("x_min", "x_max"), ("y_min", "y_max")]
    elif eq.INPUT_DIM == 3:
        names = [("x_min", "x_max"), ("y_min", "y_max"), ("t_min", "t_max")]
    else:
        raise ValueError(f"No default input bounds for INPUT_DIM={eq.INPUT_DIM}")
    lows = [float(getattr(eq, lo)) for lo, _ in names]
    highs = [float(getattr(eq, hi)) for _, hi in names]
    return lows, highs


class FourierFeatureEncoder(nn.Module):
    # Deterministic multiscale Fourier features used by FF-PINN and FDG-PINN.
    # Coordinates are normalized to [-1, 1], then expanded with sin/cos bases
    # at the prescribed frequencies to expose oscillatory components directly.
    def __init__(self, lows, highs, freqs):
        super().__init__()
        lows_t = torch.tensor(lows, dtype=TORCH_DTYPE)
        highs_t = torch.tensor(highs, dtype=TORCH_DTYPE)
        self.register_buffer("lows", lows_t)
        self.register_buffer("scale", torch.clamp(highs_t - lows_t, min=1e-6))
        self.register_buffer("freqs", torch.tensor(freqs, dtype=TORCH_DTYPE))
        self.out_dim = len(lows) * (1 + 2 * len(freqs))

    def forward(self, x):
        z = 2.0 * (x - self.lows) / self.scale - 1.0
        angles = np.pi * z.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=1)
        return torch.cat([z, sincos], dim=1)


class AdaptiveFourierFeatureEncoder(nn.Module):
    # Optional experimental encoder with learnable frequency shifts, phases,
    # and gates. The final paper uses the deterministic Fourier encoder above
    # to keep the PDE comparisons controlled and reproducible.
    def __init__(self, lows, highs, freqs, gate_bias=3.0):
        super().__init__()
        lows_t = torch.tensor(lows, dtype=TORCH_DTYPE)
        highs_t = torch.tensor(highs, dtype=TORCH_DTYPE)
        base_freqs = torch.tensor(freqs, dtype=TORCH_DTYPE)
        self.register_buffer("lows", lows_t)
        self.register_buffer("scale", torch.clamp(highs_t - lows_t, min=1e-6))
        self.register_buffer("base_freqs", base_freqs)

        dim = len(lows)
        n_freq = len(freqs)
        self.log_freq_delta = nn.Parameter(torch.zeros(dim, n_freq))
        self.phase = nn.Parameter(torch.zeros(dim, n_freq))
        self.gate_net = nn.Linear(dim, dim * n_freq)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, float(gate_bias))
        self.out_dim = dim * (1 + 2 * n_freq)

    def forward(self, x):
        z = 2.0 * (x - self.lows) / self.scale - 1.0
        freq = self.base_freqs.view(1, -1) * torch.exp(torch.clamp(self.log_freq_delta, -2.0, 2.0))
        angles = np.pi * z.unsqueeze(-1) * freq.unsqueeze(0) + self.phase.unsqueeze(0)
        gates = torch.sigmoid(self.gate_net(z)).view(z.shape[0], z.shape[1], -1)
        sincos = torch.cat([gates * torch.sin(angles), gates * torch.cos(angles)], dim=-1).flatten(start_dim=1)
        return torch.cat([z, sincos], dim=1)


class EncodedPINN(nn.Module):
    def __init__(self, encoder, trunk):
        super().__init__()
        self.encoder = encoder
        self.trunk = trunk

    def forward(self, x):
        return self.trunk(self.encoder(x))


class EqOutputTransformPINN(nn.Module):
    def __init__(self, model, transform):
        super().__init__()
        self.model = model
        self.transform = transform
        self.uses_hard_output_transform = True

    def forward(self, x):
        raw = self.model(x)
        return self.transform(x, raw)


class EqInputTransformPINN(nn.Module):
    def __init__(self, transform, model):
        super().__init__()
        self.transform = transform
        self.model = model

    def forward(self, x):
        return self.model(self.transform(x))


class ImprovedPINN(nn.Module):
    # Phi-NN-style structural baseline. It mixes three auxiliary feature
    # streams through learned gates; Fourier encoding is added only when this
    # trunk is wrapped by build_model().
    def __init__(self, in_dim, out_dim, hidden=96, n_hidden_layers=5, gate_act="sigmoid"):
        super().__init__()
        assert gate_act in ("sigmoid", "tanh")
        self.hidden = hidden
        self.n_hidden_layers = n_hidden_layers
        self.gate_act = gate_act
        self.act = nn.Tanh()

        self.gamma1 = MLP(in_dim, hidden, hidden=64, depth=2, act=nn.Tanh)
        self.gamma2 = MLP(in_dim, hidden, hidden=64, depth=2, act=nn.Tanh)
        self.gamma3 = MLP(in_dim, hidden, hidden=64, depth=2, act=nn.Tanh)

        self.in_layer = nn.Linear(in_dim, hidden)
        self.y1_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers - 1))])
        self.y2_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers - 1))])
        self.out_layer = nn.Linear(hidden, out_dim)

    def _gate(self, a):
        return torch.sigmoid(a) if self.gate_act == "sigmoid" else torch.tanh(a)

    def forward(self, x):
        g1 = self.gamma1(x)
        g2 = self.gamma2(x)
        g3 = self.gamma3(x)

        h = self.act(self.in_layer(x))
        one = None
        for k in range(self.n_hidden_layers - 1):
            y1 = self._gate(self.y1_layers[k](h))
            y2 = self._gate(self.y2_layers[k](h))
            if one is None:
                one = torch.ones_like(y1)
            h = (one - y1 - y2) * g1 + y1 * g2 + y2 * g3
        return self.out_layer(h)


class DoubleGatePINN(nn.Module):
    # Double-Gate backbone used in DG-PINN and FDG-PINN. U and V are two
    # coordinate-dependent feature bases; Z interpolates between them, while G
    # controls how much the gated basis replaces the ordinary hidden update.
    def __init__(self, in_dim, out_dim, hidden=96, n_hidden_layers=5):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.act = nn.Tanh()

        self.in_layer = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers - 1))])

        self.U_net = MLP(in_dim, hidden, hidden=64, depth=2, act=nn.Tanh)
        self.V_net = MLP(in_dim, hidden, hidden=64, depth=2, act=nn.Tanh)

        self.Z_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers - 1))])
        self.G_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers - 1))])
        self.out_layer = nn.Linear(hidden, out_dim)

    def forward(self, x):
        U = self.U_net(x)
        V = self.V_net(x)
        z = self.act(self.in_layer(x))

        for k in range(self.n_hidden_layers - 1):
            hk = self.act(self.layers[k](z))
            Z = torch.sigmoid(self.Z_layers[k](z))
            h_star = (1.0 - Z) * U + Z * V
            G = torch.sigmoid(self.G_layers[k](z))
            z = (1.0 - G) * hk + G * h_star
        return self.out_layer(z)


class ResidualGatePINN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=96, n_hidden_layers=5):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers))])
        self.gates = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers))])
        self.skip_scales = nn.Parameter(torch.full((max(1, n_hidden_layers),), 0.1))
        self.out_layer = nn.Linear(hidden, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.in_layer(x))
        for k, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            cand = self.act(block(h))
            g = torch.sigmoid(gate(h))
            scale = torch.tanh(self.skip_scales[k])
            h = h + scale * g * cand
        return self.out_layer(h)


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class AttentionResidualPINN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=96, n_hidden_layers=5):
        super().__init__()
        self.hidden = hidden
        self.n_hidden_layers = n_hidden_layers
        self.temperature = float(np.sqrt(hidden))
        self.in_layer = nn.Linear(in_dim, hidden)
        self.act = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(1, n_hidden_layers - 1))])
        self.layer_queries = nn.Parameter(torch.zeros(max(1, n_hidden_layers - 1), hidden))
        self.final_query = nn.Parameter(torch.zeros(hidden))
        self.key_norm = SimpleRMSNorm(hidden)
        self.out_layer = nn.Linear(hidden, out_dim)

    def _attend_history(self, history, query):
        V = torch.stack(history, dim=1)
        K = self.key_norm(V)
        q = query.view(1, 1, -1)
        logits = torch.sum(K * q, dim=-1) / self.temperature
        alpha = torch.softmax(logits, dim=1)
        return torch.sum(alpha.unsqueeze(-1) * V, dim=1)

    def forward(self, x):
        z0 = self.act(self.in_layer(x))
        history = [z0]
        for k in range(self.n_hidden_layers - 1):
            h = self._attend_history(history, self.layer_queries[k])
            z = self.act(self.layers[k](h))
            history.append(z)
        h_final = self._attend_history(history, self.final_query)
        return self.out_layer(h_final)



def build_model(eq, model_kind):
    # Compose the final model from independent switches: optional
    # characteristic-coordinate transform, optional Fourier encoder, selected
    # network trunk, and optional hard output transform. This is what makes the
    # ablation variants differ by one module while sharing the same pipeline.
    in_dim = eq.INPUT_DIM
    out_dim = eq.OUTPUT_DIM
    use_char_transform = model_kind.startswith("char_")
    if use_char_transform:
        model_kind = model_kind[len("char_"):]
        in_dim = int(getattr(eq, "CHAR_INPUT_DIM"))
    use_hard_transform = model_kind.startswith("hard_")
    if use_hard_transform:
        model_kind = model_kind[len("hard_"):]
    use_adaptive_fourier = model_kind.startswith("adaptive_fourier_")
    if use_adaptive_fourier:
        model_kind = model_kind[len("adaptive_fourier_"):]
    use_fourier = model_kind.startswith("fourier_")
    trunk_kind = model_kind[len("fourier_"):] if use_fourier else model_kind
    trunk_in_dim = in_dim
    encoder = None
    if use_fourier or use_adaptive_fourier:
        if use_char_transform and hasattr(eq, "char_input_bounds"):
            lows, highs = eq.char_input_bounds()
        else:
            lows, highs = input_bounds_from_eq(eq)
        if use_adaptive_fourier:
            encoder = AdaptiveFourierFeatureEncoder(lows, highs, fourier_freqs)
        else:
            encoder = FourierFeatureEncoder(lows, highs, fourier_freqs)
        trunk_in_dim = encoder.out_dim

    def finish(trunk):
        model = EncodedPINN(encoder, trunk) if encoder is not None else trunk
        if use_char_transform:
            model = EqInputTransformPINN(getattr(eq, "char_input_transform"), model)
        transform = getattr(eq, "output_transform", None) if use_hard_transform else None
        if transform is not None:
            model = EqOutputTransformPINN(model, transform)
        elif use_hard_transform:
            raise ValueError(f"MODEL_KIND requested hard transform, but {eq.NAME} has no output_transform")
        return model.to(device=device, dtype=TORCH_DTYPE)

    if trunk_kind == "base":
        return finish(BaseMLP(trunk_in_dim, out_dim, hidden=hidden, depth=n_hidden_layers))
    if trunk_kind == "improved_nn":
        return finish(ImprovedPINN(trunk_in_dim, out_dim, hidden=hidden, n_hidden_layers=n_hidden_layers))
    if trunk_kind == "double_gate":
        return finish(DoubleGatePINN(trunk_in_dim, out_dim, hidden=hidden, n_hidden_layers=n_hidden_layers))
    if trunk_kind == "attn_res":
        return finish(AttentionResidualPINN(trunk_in_dim, out_dim, hidden=hidden, n_hidden_layers=n_hidden_layers))
    if trunk_kind == "res_gate":
        return finish(ResidualGatePINN(trunk_in_dim, out_dim, hidden=hidden, n_hidden_layers=n_hidden_layers))
    raise ValueError("Unknown MODEL_KIND")


# =========================================================
# 5) Active-term weighting (supports 2-term / 3-term automatically)
# =========================================================
class SigmaAWGeneric(nn.Module):
    # Uncertainty-style adaptive loss weighting. For each active term L_i, q_i
    # is learned and contributes 0.5 * exp(-q_i) * L_i + 0.5 * q_i. The clamp
    # prevents the learned weights from collapsing or exploding.
    def __init__(self, n_terms, init_q=0.0, q_min=-10.0, q_max=10.0):
        super().__init__()
        self.q = nn.Parameter(torch.full((n_terms,), float(init_q), dtype=TORCH_DTYPE))
        self.q_min = q_min
        self.q_max = q_max

    def clamp_(self):
        with torch.no_grad():
            self.q.clamp_(self.q_min, self.q_max)

    def weights(self):
        return torch.exp(-self.q)

    def total_loss(self, losses_active):
        w = self.weights()
        total = 0.5 * torch.sum(w * losses_active) + 0.5 * torch.sum(self.q)
        return total, w


class LossCompensatorGeneric:
    # Extra imbalance compensator used only by the experimental improved_aw1
    # mode. It tracks relative loss scales and applies a smoothed multiplier
    # when one active constraint term dominates the others.
    def __init__(self,
                 n_terms,
                 gap_log10=2.0, alpha=0.6,
                 gamma=2.5,
                 cmin=0.25, cmax=8.0,
                 ema=0.95, eps=1e-30):
        self.n_terms = n_terms
        self.gap_log10 = gap_log10
        self.alpha = alpha
        self.gamma = gamma
        self.cmin = cmin
        self.cmax = cmax
        self.ema = ema
        self.eps = eps
        self.inited = False
        self.c_ema = np.ones(n_terms, dtype=np.float64)

    @torch.no_grad()
    def update(self, losses_active):
        vals = np.array([float(v.detach().item()) for v in losses_active], dtype=np.float64) + self.eps
        ratios = vals / max(float(np.sum(vals)), self.eps)
        c = np.exp(self.gamma * (ratios - 1.0 / self.n_terms))

        gap = float(np.log10(float(np.max(vals)) / max(float(np.min(vals)), self.eps)))
        if gap > self.gap_log10:
            boost = 10.0 ** (self.alpha * (gap - self.gap_log10))
            c[int(np.argmax(vals))] *= boost

        c = np.clip(c, self.cmin, self.cmax)
        c = c / max(float(np.mean(c)), self.eps)

        if not self.inited:
            self.c_ema = c
            self.inited = True
        else:
            a = self.ema
            self.c_ema = a * self.c_ema + (1.0 - a) * c

        out = self.c_ema / max(float(np.mean(self.c_ema)), self.eps)
        out = np.clip(out, self.cmin, self.cmax)
        info = {
            "ratios": tuple(float(x) for x in ratios),
            "gap_log10": float(gap),
            "comp": tuple(float(x) for x in out),
        }
        return out, info


# =========================================================
# 6) Loss parts and active-term packing
# =========================================================
mse = nn.MSELoss()


def make_samples(eq):
    # Sample the training constraints once per run. PDE collocation points are
    # always used; IC and BC samples are requested only when the equation module
    # declares the corresponding constraints.
    colloc = eq.sample_collocation(N_f, device, sample_mode)
    ic_data = eq.sample_ic(N_ic, device, sample_mode) if getattr(eq, "HAS_IC", False) else None
    bc_data = eq.sample_bc(N_bc, device, sample_mode) if getattr(eq, "HAS_BC", True) else None
    anchor_data = sample_anchor(eq, N_anchor) if USE_ANCHOR_LOSS and getattr(eq, "TRUE_AVAILABLE", False) else None
    return tuple(to_run_dtype(x) for x in (colloc, ic_data, bc_data, anchor_data))


def to_run_dtype(obj):
    if torch.is_tensor(obj):
        return obj.to(dtype=TORCH_DTYPE)
    if isinstance(obj, dict):
        return {k: to_run_dtype(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_run_dtype(v) for v in obj)
    return obj


def _lhs_np(n, dim):
    seg = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
    pts = np.zeros((n, dim), dtype=np.float32)
    for j in range(dim):
        u = np.random.rand(n).astype(np.float32)
        pts[:, j] = seg[:-1] + u * (seg[1:] - seg[:-1])
        np.random.shuffle(pts[:, j])
    return pts


def _sample_box_np(n, bounds):
    s = _lhs_np(n, len(bounds)) if sample_mode == "LHS" else np.random.rand(n, len(bounds)).astype(np.float32)
    out = np.zeros_like(s, dtype=np.float32)
    for j, (a, b) in enumerate(bounds):
        out[:, j] = float(a) + (float(b) - float(a)) * s[:, j]
    return out


def sample_anchor(eq, n):
    if eq.INPUT_DIM == 2 and getattr(eq, "PLOT_STYLE", "") == "1d_xt":
        arr = _sample_box_np(n, [(eq.x_min, eq.x_max), (eq.t_min, eq.t_max)])
        true = eq.true_solution_np(arr[:, 0], arr[:, 1])
    elif eq.INPUT_DIM == 2:
        arr = _sample_box_np(n, [(eq.x_min, eq.x_max), (eq.y_min, eq.y_max)])
        true = eq.true_solution_np(arr[:, 0], arr[:, 1])
    elif eq.INPUT_DIM == 3:
        arr = _sample_box_np(n, [(eq.x_min, eq.x_max), (eq.y_min, eq.y_max), (eq.t_min, eq.t_max)])
        true = eq.true_solution_np(arr[:, 0], arr[:, 1], arr[:, 2])
    else:
        raise ValueError(f"No default anchor sampler for INPUT_DIM={eq.INPUT_DIM}")
    return {
        "x": torch.tensor(arr, dtype=TORCH_DTYPE, device=device),
        "target": torch.tensor(np.asarray(true).reshape(n, eq.OUTPUT_DIM), dtype=TORCH_DTYPE, device=device),
    }


def zero_scalar():
    return torch.tensor(0.0, dtype=TORCH_DTYPE, device=device)


def loss_parts(eq, model, samples):
    # Shared loss assembly for all equation modules. The optional anchor loss is
    # kept for diagnostics but is disabled in the paper experiments.
    colloc, ic_data, bc_data, anchor_data = samples
    f = eq.pde_residual(model, colloc, grad)
    L_pde = mse(f, torch.zeros_like(f))
    L_ic = eq.ic_loss(model, ic_data, mse) if getattr(eq, "HAS_IC", False) else zero_scalar()
    L_bc = eq.bc_loss(model, bc_data, mse) if getattr(eq, "HAS_BC", True) else zero_scalar()
    L_anchor = mse(model(anchor_data["x"]), anchor_data["target"]) if anchor_data is not None else zero_scalar()
    return {
        "pde": L_pde,
        "ic": L_ic,
        "bc": L_bc,
        "anchor": L_anchor,
    }


def active_term_names(eq, model=None):
    # Remove IC/BC terms that are already enforced by a hard output transform,
    # so constraints satisfied analytically are not double-counted in training.
    names = ["pde"]
    if getattr(eq, "HAS_IC", False):
        hard_ic_active = (
            model is not None
            and getattr(model, "uses_hard_output_transform", False)
            and getattr(eq, "HARD_IC_OUTPUT", False)
        )
        if not hard_ic_active:
            names.append("ic")
    hard_bc_active = (
        model is not None
        and getattr(model, "uses_hard_output_transform", False)
        and getattr(eq, "HARD_BC_OUTPUT", False)
    )
    if getattr(eq, "HAS_BC", True) and not hard_bc_active:
        names.append("bc")
    if USE_ANCHOR_LOSS and getattr(eq, "TRUE_AVAILABLE", False):
        names.append("anchor")
    return names


def pack_active_losses(loss_dict, active_names):
    return torch.stack([loss_dict[name] for name in active_names])


def unpack_full_weights(active_names, w_active):
    out = {"pde": 0.0, "ic": 0.0, "bc": 0.0, "anchor": 0.0}
    for name, w in zip(active_names, w_active):
        out[name] = float(w)
    return out


# =========================================================
# 7) Evaluation and plotting
# =========================================================
def _to_numpy_true(true):
    true = np.asarray(true)
    return true.astype(np.float32)


def true_at_points(eq, pts):
    arr = np.asarray(pts, dtype=np.float32)
    if eq.INPUT_DIM == 2 and getattr(eq, "PLOT_STYLE", "") == "1d_xt":
        return _to_numpy_true(eq.true_solution_np(arr[:, 0], arr[:, 1])).reshape(-1, eq.OUTPUT_DIM)
    if eq.INPUT_DIM == 2:
        return _to_numpy_true(eq.true_solution_np(arr[:, 0], arr[:, 1])).reshape(-1, eq.OUTPUT_DIM)
    if eq.INPUT_DIM == 3:
        return _to_numpy_true(eq.true_solution_np(arr[:, 0], arr[:, 1], arr[:, 2])).reshape(-1, eq.OUTPUT_DIM)
    raise ValueError(f"No default true-solution evaluator for INPUT_DIM={eq.INPUT_DIM}")


def collect_training_points(samples, input_dim):
    colloc, ic_data, bc_data, _ = samples
    tensors = []
    if torch.is_tensor(colloc) and colloc.ndim == 2 and colloc.shape[1] == input_dim:
        tensors.append(colloc.detach())
    for data in (ic_data, bc_data):
        if isinstance(data, dict):
            for value in data.values():
                if torch.is_tensor(value) and value.ndim == 2 and value.shape[1] == input_dim:
                    tensors.append(value.detach())
    if not tensors:
        return None
    pts = torch.cat(tensors, dim=0)
    # Remove repeated BC/IC points only for evaluation cleanliness; this is not
    # used for optimization, so CPU de-duplication is acceptable.
    pts_np = pts.detach().cpu().numpy().astype(np.float32)
    return np.unique(pts_np, axis=0)


def evaluate_training_error_metrics(eq, model, samples):
    if not getattr(eq, "TRUE_AVAILABLE", False):
        return {"train_l2": float("nan"), "train_mse": float("nan")}
    pts_np = collect_training_points(samples, eq.INPUT_DIM)
    if pts_np is None or len(pts_np) == 0:
        return {"train_l2": float("nan"), "train_mse": float("nan")}
    preds = []
    batch = 65536
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pts_np), batch):
            pts = torch.tensor(pts_np[i:i + batch], dtype=TORCH_DTYPE, device=device)
            preds.append(model(pts).detach().cpu().numpy().reshape(-1, eq.OUTPUT_DIM))
    pred = np.concatenate(preds, axis=0)
    true = true_at_points(eq, pts_np)
    return {
        "train_l2": relative_l2_error(pred, true),
        "train_mse": mse_error(pred, true),
    }


def evaluate_error_metrics(eq, model):
    """
    Return three metrics on a separate evaluation grid:
      1) time_avg_l2: mean of slice-wise relative L2 errors
      2) global_l2: one relative L2 over the whole evaluation grid
      3) mse: mean squared error over the whole evaluation grid

    For 1D time-dependent PDEs, the grid has 600 x n_test points.
    For 2D steady PDEs, the grid has Nx x Ny points.
    For 2D time-dependent PDEs, the grid has Nx x Ny x n_test points.
    """
    if not getattr(eq, "TRUE_AVAILABLE", False):
        return {"time_avg_l2": float("nan"), "global_l2": float("nan"), "mse": float("nan")}

    preds_all = []
    trues_all = []
    slice_l2s = []

    if eq.PLOT_STYLE == "1d_xt":
        xs = eq.make_plot_grid(Nx=600)
        ts = np.linspace(getattr(eq, "t_min", 0.0), getattr(eq, "t_max", 1.0), n_test)
        with torch.no_grad():
            for tt in ts:
                x = torch.tensor(xs.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
                t = torch.full_like(x, float(tt))
                pred = model(torch.cat([x, t], dim=1)).cpu().numpy()
                true = _to_numpy_true(eq.true_solution_np(xs, tt))
                slice_l2s.append(relative_l2_error(pred, true))
                preds_all.append(pred.reshape(-1, eq.OUTPUT_DIM))
                trues_all.append(true.reshape(-1, eq.OUTPUT_DIM))

        pred_all = np.concatenate(preds_all, axis=0)
        true_all = np.concatenate(trues_all, axis=0)
        return {
            "time_avg_l2": float(np.mean(slice_l2s)),
            "global_l2": relative_l2_error(pred_all, true_all),
            "mse": mse_error(pred_all, true_all),
        }

    X, Y = eq.make_plot_grid(Nx=Nx, Ny=Ny)
    x = torch.tensor(X.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
    y = torch.tensor(Y.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)

    if eq.INPUT_DIM == 2:
        with torch.no_grad():
            pred = model(torch.cat([x, y], dim=1)).cpu().numpy().reshape(-1, eq.OUTPUT_DIM)
        true = _to_numpy_true(eq.true_solution_np(X, Y)).reshape(-1, eq.OUTPUT_DIM)
        return {
            "time_avg_l2": relative_l2_error(pred, true),
            "global_l2": relative_l2_error(pred, true),
            "mse": mse_error(pred, true),
        }

    ts = np.linspace(getattr(eq, "t_min", 0.0), getattr(eq, "t_max", 1.0), n_test)
    with torch.no_grad():
        for tt in ts:
            t = torch.full_like(x, float(tt))
            pred = model(torch.cat([x, y, t], dim=1)).cpu().numpy().reshape(-1, eq.OUTPUT_DIM)
            true = _to_numpy_true(eq.true_solution_np(X, Y, tt)).reshape(-1, eq.OUTPUT_DIM)
            slice_l2s.append(relative_l2_error(pred, true))
            preds_all.append(pred)
            trues_all.append(true)

    pred_all = np.concatenate(preds_all, axis=0)
    true_all = np.concatenate(trues_all, axis=0)
    return {
        "time_avg_l2": float(np.mean(slice_l2s)),
        "global_l2": relative_l2_error(pred_all, true_all),
        "mse": mse_error(pred_all, true_all),
    }


def evaluate_avg_l2(eq, model):
    # Backward-compatible wrapper: old name returns the time-averaged relative L2.
    return evaluate_error_metrics(eq, model)["time_avg_l2"]


def plot_loss_curves(history, save_path, title):
    if plt is None:
        print("matplotlib not available, skip loss curve.")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    ax.plot(history["iters_loss"], history["pde_loss"], label="PDE loss")
    if len(history["ic_loss"]) > 0:
        ax.plot(history["iters_loss"], history["ic_loss"], label="IC loss")
    if len(history["bc_loss"]) > 0:
        ax.plot(history["iters_loss"], history["bc_loss"], label="BC loss")
    if len(history["iters_l2"]) > 0:
        ax.plot(history["iters_l2"], history["l2_loss"], label="time-avg rel L2")
    if len(history.get("global_l2_loss", [])) > 0:
        ax.plot(history["iters_l2"], history["global_l2_loss"], label="global rel L2")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss / L2")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


METHOD_STYLES = {
    ("base", "fixed"): {
        "label": "Standard PINN",
        "color": "#1f77b4",
        "pde": "#6baed6",
        "ic": "#08519c",
        "bc": "#756bb1",
    },
    ("improved_nn", "aw"): {
        "label": r"$\phi$-NN + AW",
        "color": "#d99a00",
        "pde": "#f4c542",
        "ic": "#b66d00",
        "bc": "#8c6d31",
    },
    ("double_gate", "aw"): {
        "label": "DG-PINN + AW",
        "color": "#2ca02c",
        "pde": "#74c476",
        "ic": "#006d2c",
        "bc": "#31a354",
    },
    ("fourier_double_gate", "aw"): {
        "label": "FDG-PINN + AW",
        "color": "#d62728",
        "pde": "#fb6a4a",
        "ic": "#a50f15",
        "bc": "#e6550d",
    },
    ("hard_fourier_double_gate", "aw"): {
        "label": "HFDG-PINN + AW",
        "color": "#9467bd",
        "pde": "#bcbddc",
        "ic": "#6a51a3",
        "bc": "#807dba",
    },
}


def method_label(model_kind, weight_mode):
    return METHOD_STYLES.get((model_kind, weight_mode), {}).get(
        "label", f"{model_kind} + {weight_mode}"
    )


def method_colors(model_kind, weight_mode):
    default = {"color": "#333333", "pde": "#777777", "ic": "#111111", "bc": "#999999"}
    out = default.copy()
    out.update(METHOD_STYLES.get((model_kind, weight_mode), {}))
    return out


def plot_combined_loss_curves(runs, save_path, title):
    if plt is None:
        print("matplotlib not available, skip combined loss curve.")
        return

    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=180)
    for run in runs:
        history = run["history"]
        model_kind = run["model_kind"]
        weight_mode = run["weight_mode"]
        label = method_label(model_kind, weight_mode)
        colors = method_colors(model_kind, weight_mode)

        if len(history.get("iters_l2", [])) > 0:
            ax.plot(
                history["iters_l2"],
                history["l2_loss"],
                color=colors["color"],
                linewidth=2.4,
                label=f"{label}: Discrete L2",
            )

        if len(history.get("iters_loss", [])) > 0:
            ax.plot(
                history["iters_loss"],
                history["pde_loss"],
                color=colors["pde"],
                linestyle="--",
                linewidth=1.35,
                alpha=0.82,
                label=f"{label}: PDE",
            )
            if any(np.isfinite(v) and abs(v) > 0 for v in history.get("ic_loss", [])):
                ax.plot(
                    history["iters_loss"],
                    history["ic_loss"],
                    color=colors["ic"],
                    linestyle=":",
                    linewidth=1.35,
                    alpha=0.78,
                    label=f"{label}: IC",
                )
            if any(np.isfinite(v) and abs(v) > 0 for v in history.get("bc_loss", [])):
                ax.plot(
                    history["iters_loss"],
                    history["bc_loss"],
                    color=colors["bc"],
                    linestyle="-.",
                    linewidth=1.35,
                    alpha=0.78,
                    label=f"{label}: BC",
                )

    ax.axvline(steps_adam, color="#666666", linestyle=":", linewidth=1.0)
    ax.text(steps_adam, 0.98, "Adam / L-BFGS", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=9, color="#555555")
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Loss / relative L2")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.2, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_loss_history_data(history, prefix_path):
    loss_csv = prefix_path + "_loss_terms.csv"
    l2_csv = prefix_path + "_l2_curve.csv"
    npz_path = prefix_path + "_plot_data.npz"

    loss_header = "iteration,pde_loss,ic_loss,bc_loss"
    loss_arr = np.column_stack([
        np.asarray(history.get("iters_loss", []), dtype=np.float64),
        np.asarray(history.get("pde_loss", []), dtype=np.float64),
        np.asarray(history.get("ic_loss", []), dtype=np.float64),
        np.asarray(history.get("bc_loss", []), dtype=np.float64),
    ])
    if loss_arr.size > 0:
        np.savetxt(loss_csv, loss_arr, delimiter=",", header=loss_header, comments="")

    l2_header = "iteration,time_avg_l2,global_l2,mse"
    l2_arr = np.column_stack([
        np.asarray(history.get("iters_l2", []), dtype=np.float64),
        np.asarray(history.get("l2_loss", []), dtype=np.float64),
        np.asarray(history.get("global_l2_loss", []), dtype=np.float64),
        np.asarray(history.get("mse_loss", []), dtype=np.float64),
    ])
    if l2_arr.size > 0:
        np.savetxt(l2_csv, l2_arr, delimiter=",", header=l2_header, comments="")

    np.savez(
        npz_path,
        iters_loss=np.asarray(history.get("iters_loss", []), dtype=np.float64),
        pde_loss=np.asarray(history.get("pde_loss", []), dtype=np.float64),
        ic_loss=np.asarray(history.get("ic_loss", []), dtype=np.float64),
        bc_loss=np.asarray(history.get("bc_loss", []), dtype=np.float64),
        iters_l2=np.asarray(history.get("iters_l2", []), dtype=np.float64),
        time_avg_l2=np.asarray(history.get("l2_loss", []), dtype=np.float64),
        global_l2=np.asarray(history.get("global_l2_loss", []), dtype=np.float64),
        mse=np.asarray(history.get("mse_loss", []), dtype=np.float64),
    )
    return loss_csv, l2_csv, npz_path


def save_prediction_field_data(eq, model, save_path):
    if not SAVE_PREDICTION_DATA:
        return None
    model.eval()
    with torch.no_grad():
        if eq.PLOT_STYLE == "1d_xt" or eq.INPUT_DIM == 2 and hasattr(eq, "t_min"):
            xs = eq.make_plot_grid(Nx=max(Nx, 600))
            ts = np.linspace(getattr(eq, "t_min", 0.0), getattr(eq, "t_max", 1.0), n_test, dtype=np.float32)
            pred_all, true_all = [], []
            for tt in ts:
                x = torch.tensor(xs.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
                t = torch.full_like(x, float(tt))
                xt = torch.cat([x, t], dim=1)
                pred = model(xt).detach().cpu().numpy().reshape(len(xs), eq.OUTPUT_DIM)
                pred_all.append(pred)
                if getattr(eq, "TRUE_AVAILABLE", False):
                    true_all.append(_to_numpy_true(eq.true_solution_np(xs, tt)).reshape(len(xs), eq.OUTPUT_DIM))
            pred_arr = np.stack(pred_all, axis=0)
            true_arr = np.stack(true_all, axis=0) if true_all else np.full_like(pred_arr, np.nan)
            np.savez(
                save_path,
                plot_style="1d_xt",
                x=xs,
                t=ts,
                pred=pred_arr,
                true=true_arr,
                error=pred_arr - true_arr,
                output_dim=eq.OUTPUT_DIM,
            )
            return save_path

        if eq.INPUT_DIM == 2:
            X, Y = eq.make_plot_grid(Nx=Nx, Ny=Ny)
            x = torch.tensor(X.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
            y = torch.tensor(Y.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
            pred = model(torch.cat([x, y], dim=1)).detach().cpu().numpy().reshape(*X.shape, eq.OUTPUT_DIM)
            if getattr(eq, "TRUE_AVAILABLE", False):
                true = _to_numpy_true(eq.true_solution_np(X, Y)).reshape(*X.shape, eq.OUTPUT_DIM)
            else:
                true = np.full_like(pred, np.nan)
            np.savez(
                save_path,
                plot_style="2d_xy",
                x=X,
                y=Y,
                pred=pred,
                true=true,
                error=pred - true,
                output_dim=eq.OUTPUT_DIM,
            )
            return save_path

        if eq.INPUT_DIM == 3:
            X, Y = eq.make_plot_grid(Nx=Nx, Ny=Ny)
            tt = float(getattr(eq, "t_max", 1.0))
            x = torch.tensor(X.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
            y = torch.tensor(Y.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
            t = torch.full_like(x, tt)
            pred = model(torch.cat([x, y, t], dim=1)).detach().cpu().numpy().reshape(*X.shape, eq.OUTPUT_DIM)
            if getattr(eq, "TRUE_AVAILABLE", False):
                true = _to_numpy_true(eq.true_solution_np(X, Y, tt)).reshape(*X.shape, eq.OUTPUT_DIM)
            else:
                true = np.full_like(pred, np.nan)
            np.savez(
                save_path,
                plot_style="2d_xyt_snapshot",
                x=X,
                y=Y,
                t=np.array(tt),
                pred=pred,
                true=true,
                error=pred - true,
                output_dim=eq.OUTPUT_DIM,
            )
            return save_path

    return None


def _fmt_sci(v):
    try:
        v = float(v)
    except Exception:
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.6e}"


def _latex_sci(v):
    try:
        v = float(v)
    except Exception:
        return "--"
    if not np.isfinite(v):
        return "--"
    s = f"{v:.3e}"
    mant, exp = s.split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def _latex_texttt(s):
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return r"\texttt{" + "".join(repl.get(ch, ch) for ch in s) + "}"


def save_summary_tables(eq_name, results, out_dir):
    eq = importlib.import_module(eq_name)
    tag = now_tag()
    csv_path = os.path.join(out_dir, f"{eq.NAME}_summary_{tag}.csv")
    tex_path = os.path.join(out_dir, f"{eq.NAME}_summary_{tag}.tex")

    header = [
        "Method",
        "ModelKind",
        "WeightMode",
        "Time(s)",
        "FinalPDE",
        "FinalIC",
        "FinalBC",
        "L2(training)",
        "TrainMSE",
        "EvalMSE",
        "GlobalL2",
        "DiscreteL2",
    ]
    rows = []
    for _, mk, wm, metrics in results:
        if not isinstance(metrics, dict):
            continue
        rows.append([
            method_label(mk, wm).replace("$", ""),
            mk,
            wm,
            _fmt_sci(metrics.get("training_time", np.nan)),
            _fmt_sci(metrics.get("final_pde_loss", np.nan)),
            _fmt_sci(metrics.get("final_ic_loss", np.nan)),
            _fmt_sci(metrics.get("final_bc_loss", np.nan)),
            _fmt_sci(metrics.get("train_l2", np.nan)),
            _fmt_sci(metrics.get("train_mse", np.nan)),
            _fmt_sci(metrics.get("mse", np.nan)),
            _fmt_sci(metrics.get("global_l2", np.nan)),
            _fmt_sci(metrics.get("time_avg_l2", np.nan)),
        ])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(f"\\caption{{{_latex_texttt(eq.NAME)} formal run summary}}\n")
        f.write("\\begin{adjustbox}{max width=\\textwidth}\n\\scriptsize\n")
        f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write("Method & Time(s) & Final PDE & Final IC & Final BC & $L_2$(training) & Global $L_2$ & Discrete $L_2$ \\\\\n")
        f.write("\\midrule\n")
        for _, mk, wm, metrics in results:
            if not isinstance(metrics, dict):
                continue
            f.write(
                f"{method_label(mk, wm)} & "
                f"{float(metrics.get('training_time', np.nan)):.2f} & "
                f"{_latex_sci(metrics.get('final_pde_loss', np.nan))} & "
                f"{_latex_sci(metrics.get('final_ic_loss', np.nan))} & "
                f"{_latex_sci(metrics.get('final_bc_loss', np.nan))} & "
                f"{_latex_sci(metrics.get('train_l2', np.nan))} & "
                f"{_latex_sci(metrics.get('global_l2', np.nan))} & "
                f"{_latex_sci(metrics.get('time_avg_l2', np.nan))} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")

    print(f"Summary CSV saved to: {csv_path}")
    print(f"Summary TeX saved to: {tex_path}")
    return csv_path, tex_path


def append_summary_to_main_tex(eq_name, summary_tex_path):
    tex_path = os.path.join(ROOT_DIR, "HFDG_PINN.tex")
    if not os.path.exists(tex_path):
        print(f"Main tex not found, skip append: {tex_path}")
        return
    snippet = open(summary_tex_path, "r", encoding="utf-8").read()
    eq = importlib.import_module(eq_name)
    block = (
        "\n\n"
        f"% Auto-appended by FDG_runner.py for {eq.NAME} at {datetime.now().isoformat(timespec='seconds')}\n"
        f"\\subsection{{{_latex_texttt(eq.NAME)} 实验结果}}\n"
        f"{snippet}\n"
    )
    content = open(tex_path, "r", encoding="utf-8").read()
    preferred_markers = [
        "\\section{Discussion}",
        "\\bibliographystyle",
        "\\end{document}",
    ]
    for marker in preferred_markers:
        if marker in content:
            content = content.replace(marker, block + "\n" + marker, 1)
            break
    else:
        content = content + block
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Summary appended to: {tex_path}")



def make_snapshot(eq, model, save_path, t_snap=None):
    if plt is None:
        print("matplotlib not available, skip snapshot.")
        return
    if eq.PLOT_STYLE == "1d_xt":
        xs = eq.make_plot_grid(Nx=600)
        tt = float(getattr(eq, "t_max", 1.0)) if t_snap is None else float(t_snap)
        x = torch.tensor(xs.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
        t = torch.full_like(x, tt)
        with torch.no_grad():
            pred = model(torch.cat([x, t], dim=1)).cpu().numpy().reshape(-1)
        true = eq.true_solution_np(xs, tt) if getattr(eq, "TRUE_AVAILABLE", False) else None
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=140)
        axes[0].plot(xs, pred)
        axes[0].set_title("Pred")
        if true is not None:
            axes[1].plot(xs, true)
            axes[1].set_title("True")
            axes[2].plot(xs, pred - true)
            axes[2].set_title("Error")
        else:
            axes[1].axis("off")
            axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return

    X, Y = eq.make_plot_grid(Nx=Nx, Ny=Ny)
    x = torch.tensor(X.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
    y = torch.tensor(Y.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)

    if eq.INPUT_DIM == 2:
        with torch.no_grad():
            pred = model(torch.cat([x, y], dim=1)).cpu().numpy().reshape(Ny, Nx, eq.OUTPUT_DIM)
        true = eq.true_solution_np(X, Y) if getattr(eq, "TRUE_AVAILABLE", False) else None
    else:
        tt = float(getattr(eq, "t_max", 1.0)) if t_snap is None else float(t_snap)
        t = torch.full_like(x, tt)
        with torch.no_grad():
            pred = model(torch.cat([x, y, t], dim=1)).cpu().numpy().reshape(Ny, Nx, eq.OUTPUT_DIM)
        true = eq.true_solution_np(X, Y, tt) if getattr(eq, "TRUE_AVAILABLE", False) else None

    if eq.OUTPUT_DIM == 1:
        P = pred[..., 0]
        T = true[..., 0] if true is not None and true.ndim == 3 else (true if true is not None else None)
        E = None if T is None else P - T
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=140)
        axes[0].imshow(P, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
        axes[0].set_title("Pred")
        if T is not None:
            axes[1].imshow(T, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
            axes[1].set_title("True")
            im = axes[2].imshow(E, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
            axes[2].set_title("Error")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            axes[1].axis("off")
            axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return

    names = getattr(eq, "COMPONENT_NAMES", [f"c{i}" for i in range(eq.OUTPUT_DIM)])
    nrow = eq.OUTPUT_DIM
    fig, axes = plt.subplots(nrow, 3, figsize=(10.5, 3.0 * nrow), dpi=140)
    if nrow == 1:
        axes = np.array([axes])
    for i in range(eq.OUTPUT_DIM):
        P = pred[..., i]
        T = true[..., i] if true is not None else None
        E = None if T is None else (P - T)
        axes[i, 0].imshow(P, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
        axes[i, 0].set_title(f"{names[i]} pred")
        if T is not None:
            axes[i, 1].imshow(T, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
            axes[i, 1].set_title(f"{names[i]} true")
            im = axes[i, 2].imshow(E, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
            axes[i, 2].set_title(f"{names[i]} error")
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        else:
            axes[i, 1].axis("off")
            axes[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)



def make_gif(eq, model, gif_path):
    if imageio is None or plt is None:
        print("imageio/matplotlib not available, skip GIF.")
        return
    if eq.PLOT_STYLE == "1d_xt":
        xs = eq.make_plot_grid(Nx=600)
        ts = np.linspace(getattr(eq, "t_min", 0.0), getattr(eq, "t_max", 1.0), n_frames)
        tmp_dir = os.path.join(os.path.dirname(gif_path), "_tmp_frames")
        os.makedirs(tmp_dir, exist_ok=True)
        frame_paths = []
        with torch.no_grad():
            for k, tt in enumerate(ts):
                x = torch.tensor(xs.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
                t = torch.full_like(x, float(tt))
                pred = model(torch.cat([x, t], dim=1)).cpu().numpy().reshape(-1)
                true = eq.true_solution_np(xs, tt) if getattr(eq, "TRUE_AVAILABLE", False) else None
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=120)
                axes[0].plot(xs, pred)
                axes[0].set_title(f"Pred t={tt:.2f}")
                if true is not None:
                    axes[1].plot(xs, true)
                    axes[1].set_title(f"True t={tt:.2f}")
                    axes[2].plot(xs, pred - true)
                    axes[2].set_title("Error")
                else:
                    axes[1].axis("off")
                    axes[2].axis("off")
                fig.tight_layout()
                fp = os.path.join(tmp_dir, f"frame_{k:03d}.png")
                fig.savefig(fp)
                plt.close(fig)
                frame_paths.append(fp)
        images = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(gif_path, images, duration=gif_duration)
        return

    X, Y = eq.make_plot_grid(Nx=Nx, Ny=Ny)
    x = torch.tensor(X.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
    y = torch.tensor(Y.reshape(-1, 1), dtype=TORCH_DTYPE, device=device)
    tmp_dir = os.path.join(os.path.dirname(gif_path), "_tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    frame_paths = []

    if eq.INPUT_DIM == 2:
        ts = [None]
    else:
        ts = np.linspace(getattr(eq, "t_min", 0.0), getattr(eq, "t_max", 1.0), n_frames)

    with torch.no_grad():
        for k, tt in enumerate(ts):
            if eq.INPUT_DIM == 2:
                pred = model(torch.cat([x, y], dim=1)).cpu().numpy().reshape(Ny, Nx, eq.OUTPUT_DIM)
                true = eq.true_solution_np(X, Y) if getattr(eq, "TRUE_AVAILABLE", False) else None
            else:
                t = torch.full_like(x, float(tt))
                pred = model(torch.cat([x, y, t], dim=1)).cpu().numpy().reshape(Ny, Nx, eq.OUTPUT_DIM)
                true = eq.true_solution_np(X, Y, float(tt)) if getattr(eq, "TRUE_AVAILABLE", False) else None

            P = pred[..., 0]
            T = true[..., 0] if true is not None and true.ndim == 3 else (true if true is not None else None)
            E = None if T is None else P - T

            fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=120)
            axes[0].imshow(T if T is not None else P, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
            axes[0].set_title("True" if T is not None else "Pred")
            axes[1].imshow(P, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
            axes[1].set_title("Pred")
            if E is not None:
                im = axes[2].imshow(E, origin="lower", extent=[eq.x_min, eq.x_max, eq.y_min, eq.y_max], aspect="auto")
                axes[2].set_title("Error")
                fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            else:
                axes[2].axis("off")
            fig.tight_layout()
            fp = os.path.join(tmp_dir, f"frame_{k:03d}.png")
            fig.savefig(fp)
            plt.close(fig)
            frame_paths.append(fp)
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=gif_duration)


# =========================================================
# 8) Train one experiment
# =========================================================
def train_one(eq, model_kind, weight_mode, out_dir):
    # Train one equation/method/weighting combination and save all artifacts
    # needed for the paper tables, loss curves, field plots, and GIFs.
    model = build_model(eq, model_kind)
    samples = make_samples(eq)
    active_names = active_term_names(eq, model)
    n_active = len(active_names)

    history = {
        "iters_loss": [],
        "pde_loss": [],
        "ic_loss": [],
        "bc_loss": [],
        "iters_l2": [],
        "l2_loss": [],
        "global_l2_loss": [],
        "mse_loss": [],
    }
    printed_lines = []

    sigma_aw = SigmaAWGeneric(n_active).to(device) if weight_mode in ("aw", "improved_aw1") else None
    loss_comp = LossCompensatorGeneric(
        n_terms=n_active,
        gap_log10=IMB_GAP_LOG10,
        alpha=IMB_ALPHA,
        gamma=COMP_GAMMA,
        cmin=COMP_CLAMP_MIN,
        cmax=COMP_CLAMP_MAX,
        ema=COMP_EMA,
    ) if weight_mode == "improved_aw1" else None

    opt_theta = torch.optim.Adam(model.parameters(), lr=lr_theta, weight_decay=weight_decay_theta)
    opt_q = torch.optim.Adam(sigma_aw.parameters(), lr=lr_q) if sigma_aw is not None else None

    print("=" * 90)
    print(f"Running: EQ={eq.NAME}, MODEL={model_kind}, WEIGHT={weight_mode}, SAMPLE={sample_mode}")
    print(f"Counts: N_f={N_f}, N_ic={N_ic if getattr(eq, 'HAS_IC', False) else 0}, N_bc={N_bc if getattr(eq, 'HAS_BC', True) else 0}")
    print(f"Active terms: {active_names}")
    print("Stage 1: Adam (theta / q alternating)" if sigma_aw is not None else "Stage 1: Adam (theta only)")
    print("=" * 90)

    best_state = None
    best_loss_value = np.inf
    best_comp_active = None
    best_l2_state = None
    best_l2_value = np.inf
    best_l2_comp_active = None
    t0 = time.time()
    profile_trace = []
    lbfgs_closure_calls = 0
    lbfgs_state = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def current_values(update_comp=False):
        # Centralized objective evaluation used by Adam, L-BFGS, logging, and
        # final metrics. This keeps fixed-weight and adaptive-weight cases from
        # drifting into subtly different loss definitions.
        loss_dict = loss_parts(eq, model, samples)
        losses_active = pack_active_losses(loss_dict, active_names)

        if weight_mode == "fixed":
            fixed_map = {"pde": W_PDE_FIXED, "ic": W_IC_FIXED, "bc": W_BC_FIXED, "anchor": ANCHOR_WEIGHT_FIXED}
            total = zero_scalar()
            for name in active_names:
                total = total + fixed_map[name] * loss_dict[name]
            full_w = fixed_map.copy()
            info = None
            comp_active = np.ones(n_active, dtype=np.float64)
            return total, loss_dict, full_w, info, comp_active

        total_aw, w0 = sigma_aw.total_loss(losses_active)

        if weight_mode == "aw":
            full_w = unpack_full_weights(active_names, [float(v.item()) for v in w0])
            return total_aw, loss_dict, full_w, None, np.ones(n_active, dtype=np.float64)

        # improved_aw1
        if update_comp:
            comp_active, info = loss_comp.update(losses_active)
        else:
            comp_active = best_comp_active if best_comp_active is not None else np.ones(n_active, dtype=np.float64)
            info = {
                "ratios": tuple(float(v.detach().item()) / max(float(torch.sum(losses_active).detach().item()), 1e-30) for v in losses_active),
                "gap_log10": float("nan"),
                "comp": tuple(float(x) for x in comp_active),
            }
        comp_t = torch.tensor(comp_active, dtype=TORCH_DTYPE, device=device)
        w = w0 * comp_t
        total = 0.5 * torch.sum(w * losses_active) + 0.5 * torch.sum(sigma_aw.q)
        full_w = unpack_full_weights(active_names, [float(v.item()) for v in w.detach()])
        return total, loss_dict, full_w, info, comp_active

    # ---------------- Adam stage ----------------
    for it in range(1, steps_adam + 1):
        if sigma_aw is not None:
            # Update only adaptive loss parameters q. The solution network is
            # frozen so q learns the current IC/BC/PDE scale balance without
            # changing the represented solution.
            for p in model.parameters():
                p.requires_grad_(False)
            for p in sigma_aw.parameters():
                p.requires_grad_(True)
            opt_q.zero_grad()
            total_q, _, _, _, _ = current_values(update_comp=(weight_mode == "improved_aw1"))
            total_q.backward()
            opt_q.step()
            sigma_aw.clamp_()

        # Update only the neural-network parameters theta with the current
        # fixed loss weights. Alternating q and theta improves numerical
        # stability compared with updating both parameter sets at once.
        for p in model.parameters():
            p.requires_grad_(True)
        if sigma_aw is not None:
            for p in sigma_aw.parameters():
                p.requires_grad_(False)

        opt_theta.zero_grad()
        total, loss_dict, full_w, info, comp_active = current_values(update_comp=(weight_mode == "improved_aw1"))
        total.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        opt_theta.step()

        loss_scalar = float(total.item())
        if np.isfinite(loss_scalar) and loss_scalar < best_loss_value:
            best_loss_value = loss_scalar
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "sigma_aw": copy.deepcopy(sigma_aw.state_dict()) if sigma_aw is not None else None,
            }
            best_comp_active = np.array(comp_active, dtype=np.float64)

        if it == 1 or it % LOSS_LOG_EVERY == 0:
            history["iters_loss"].append(it)
            history["pde_loss"].append(float(loss_dict["pde"].item()))
            history["ic_loss"].append(float(loss_dict["ic"].item()))
            history["bc_loss"].append(float(loss_dict["bc"].item()))

        if getattr(eq, "TRUE_AVAILABLE", False) and (it == 1 or it % L2_LOG_EVERY == 0):
            model.eval()
            with torch.no_grad():
                cur_metrics = evaluate_error_metrics(eq, model)
            model.train()
            history["iters_l2"].append(it)
            history["l2_loss"].append(float(cur_metrics["time_avg_l2"]))
            history["global_l2_loss"].append(float(cur_metrics["global_l2"]))
            history["mse_loss"].append(float(cur_metrics["mse"]))
            profile_trace.append((time.time() - t0, float(cur_metrics["global_l2"]), "Adam", 0))
            cur_l2 = float(cur_metrics["time_avg_l2"])
            # For lossplot we may record L2 every 10 steps, but the original
            # HFDG_PINN_5 runs selected the Adam restore state at print/eval
            # checkpoints. Keep that behavior so the plot does not change the
            # optimization trajectory.
            can_update_restore_l2 = (it == 1 or it % print_every == 0)
            if RESTORE_ADAM_BY_L2 and can_update_restore_l2 and np.isfinite(cur_l2) and cur_l2 < best_l2_value:
                best_l2_value = cur_l2
                best_l2_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "sigma_aw": copy.deepcopy(sigma_aw.state_dict()) if sigma_aw is not None else None,
                }
                best_l2_comp_active = np.array(comp_active, dtype=np.float64)

        if it == 1 or it % print_every == 0:
            lp = float(loss_dict["pde"].item())
            li = float(loss_dict["ic"].item())
            lb = float(loss_dict["bc"].item())
            line = (
                f"[Adam {it:5d}/{steps_adam}] loss={loss_scalar:.3e} | "
                f"pde={lp:.3e}, ic={li:.3e}, bc={lb:.3e} | "
                f"w=({full_w['pde']:.2f},{full_w['ic']:.2f},{full_w['bc']:.2f})"
            )
            if info is not None:
                rtxt = ",".join(f"{x:.2f}" for x in info["ratios"])
                ctxt = ",".join(f"{x:.2f}" for x in info["comp"])
                line += f" | r=({rtxt}) gap={info['gap_log10']:.2f} comp=({ctxt})"
            print(line)
            printed_lines.append(line)

        if not np.isfinite(loss_scalar):
            print("Adam stage produced NaN/Inf, stop early and revert to best state.")
            break

    # Restore the best Adam checkpoint before L-BFGS. When an exact solution is
    # available, the restore criterion is the L2 metric reported in the paper;
    # otherwise the scalar objective is used.
    restore_state = best_l2_state if RESTORE_ADAM_BY_L2 and best_l2_state is not None else best_state
    restore_comp_active = best_l2_comp_active if restore_state is best_l2_state else best_comp_active
    if restore_state is not None:
        model.load_state_dict(restore_state["model"])
        if sigma_aw is not None and restore_state["sigma_aw"] is not None:
            sigma_aw.load_state_dict(restore_state["sigma_aw"])
            sigma_aw.clamp_()
        if restore_comp_active is not None:
            best_comp_active = np.array(restore_comp_active, dtype=np.float64)

    # ---------------- L-BFGS stage (theta only) ----------------
    if steps_lbfgs > 0:
        print("=" * 90)
        print("Stage 2: L-BFGS (theta only; q frozen, improved_aw1 compensation frozen)")
        print("=" * 90)
        for p in model.parameters():
            p.requires_grad_(True)
        if sigma_aw is not None:
            for p in sigma_aw.parameters():
                p.requires_grad_(False)

        # Freeze the compensation factors at the selected Adam state. L-BFGS
        # then refines theta under a fixed objective, which is more stable for
        # quasi-Newton optimization.
        frozen_comp_active = np.ones(n_active, dtype=np.float64) if best_comp_active is None else best_comp_active.copy()

        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=LBFGS_LR,
            max_iter=steps_lbfgs,
            history_size=LBFGS_HISTORY,
            line_search_fn=LBFGS_LINE_SEARCH,
        )

        lbfgs_closure_calls = 0

        def closure():
            nonlocal lbfgs_closure_calls
            lbfgs_closure_calls += 1
            opt_lbfgs.zero_grad()
            loss_dict = loss_parts(eq, model, samples)
            losses_active = pack_active_losses(loss_dict, active_names)
            if weight_mode == "fixed":
                fixed_map = {"pde": W_PDE_FIXED, "ic": W_IC_FIXED, "bc": W_BC_FIXED, "anchor": ANCHOR_WEIGHT_FIXED}
                total = zero_scalar()
                for name in active_names:
                    total = total + fixed_map[name] * loss_dict[name]
            elif weight_mode == "aw":
                total, _ = sigma_aw.total_loss(losses_active)
            else:
                w0 = sigma_aw.weights()
                comp_t = torch.tensor(frozen_comp_active, dtype=TORCH_DTYPE, device=device)
                w = w0 * comp_t
                total = 0.5 * torch.sum(w * losses_active) + 0.5 * torch.sum(sigma_aw.q)
            total.backward()

            if lbfgs_closure_calls == 1 or lbfgs_closure_calls % LOSS_LOG_EVERY == 0:
                x_iter = steps_adam + min(lbfgs_closure_calls, steps_lbfgs)
                history["iters_loss"].append(x_iter)
                history["pde_loss"].append(float(loss_dict["pde"].detach().item()))
                history["ic_loss"].append(float(loss_dict["ic"].detach().item()))
                history["bc_loss"].append(float(loss_dict["bc"].detach().item()))

            if (
                getattr(eq, "TRUE_AVAILABLE", False)
                and (lbfgs_closure_calls == 1 or lbfgs_closure_calls % L2_LOG_EVERY == 0)
            ):
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    cur_metrics = evaluate_error_metrics(eq, model)
                if was_training:
                    model.train()
                x_iter = steps_adam + min(lbfgs_closure_calls, steps_lbfgs)
                history["iters_l2"].append(x_iter)
                history["l2_loss"].append(float(cur_metrics["time_avg_l2"]))
                history["global_l2_loss"].append(float(cur_metrics["global_l2"]))
                history["mse_loss"].append(float(cur_metrics["mse"]))
                profile_trace.append((time.time() - t0, float(cur_metrics["global_l2"]), "L-BFGS trial", lbfgs_closure_calls))
            return total

        try:
            opt_lbfgs.step(closure)
        except Exception as e:
            print(f"L-BFGS failed: {e}")
            if best_state is not None:
                model.load_state_dict(best_state["model"])
                if sigma_aw is not None and best_state["sigma_aw"] is not None:
                    sigma_aw.load_state_dict(best_state["sigma_aw"])
                    sigma_aw.clamp_()
        lbfgs_state = opt_lbfgs.state.get(opt_lbfgs._params[0], {})

        if RESTORE_ADAM_BY_L2 and best_l2_state is not None and getattr(eq, "TRUE_AVAILABLE", False):
            model.eval()
            cur_after_lbfgs_l2 = evaluate_error_metrics(eq, model)["time_avg_l2"]
            model.train()
            if np.isfinite(cur_after_lbfgs_l2) and cur_after_lbfgs_l2 > best_l2_value:
                print(
                    f"L-BFGS worsened time-avg L2 from {best_l2_value:.3e} "
                    f"to {cur_after_lbfgs_l2:.3e}; restore best Adam-L2 state."
                )
                model.load_state_dict(best_l2_state["model"])
                if sigma_aw is not None and best_l2_state["sigma_aw"] is not None:
                    sigma_aw.load_state_dict(best_l2_state["sigma_aw"])
                    sigma_aw.clamp_()
                if best_l2_comp_active is not None:
                    best_comp_active = np.array(best_l2_comp_active, dtype=np.float64)

    # Final evaluation and artifact saving. These CSV/NPZ/image files are the
    # data source for the manuscript tables and visualizations.
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
    else:
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
    train_time = time.time() - t0

    total, loss_dict, full_w, _, _ = current_values(update_comp=False)
    total_val = float(total.item())
    lp = float(loss_dict["pde"].item())
    li = float(loss_dict["ic"].item())
    lb = float(loss_dict["bc"].item())

    if not np.isfinite(total_val) or any(not np.isfinite(v) for v in [lp, li, lb]):
        print("Final values not finite, revert to best Adam state.")
        if best_state is not None:
            model.load_state_dict(best_state["model"])
            if sigma_aw is not None and best_state["sigma_aw"] is not None:
                sigma_aw.load_state_dict(best_state["sigma_aw"])
                sigma_aw.clamp_()
            total, loss_dict, full_w, _, _ = current_values(update_comp=False)
            total_val = float(total.item())
            lp = float(loss_dict["pde"].item())
            li = float(loss_dict["ic"].item())
            lb = float(loss_dict["bc"].item())

    print(f"[Final] loss={total_val:.3e} | pde={lp:.3e}, ic={li:.3e}, bc={lb:.3e} | w=({full_w['pde']:.2f},{full_w['ic']:.2f},{full_w['bc']:.2f})")
    print(f"Training time (s): {train_time:.2f}")

    train_metrics = evaluate_training_error_metrics(eq, model, samples)
    train_l2 = train_metrics["train_l2"]
    train_mse = train_metrics["train_mse"]
    if np.isfinite(train_l2):
        print(f"Training-sample supervised relative L2-error = {train_l2:.3e}")
        print(f"Training-sample supervised MSE = {train_mse:.3e}")
    else:
        print("Training-sample supervised relative L2-error: N/A")
        print("Training-sample supervised MSE: N/A")

    final_metrics = evaluate_error_metrics(eq, model)
    avg_l2 = final_metrics["time_avg_l2"]
    global_l2 = final_metrics["global_l2"]
    mse_val = final_metrics["mse"]
    if np.isfinite(avg_l2):
        print(f"Time-averaged relative discrete L2-error = {avg_l2:.3e}")
        print(f"Global spatio-temporal relative L2-error = {global_l2:.3e}")
        print(f"Global spatio-temporal MSE = {mse_val:.3e}")
    else:
        print("Time-averaged relative discrete L2-error: N/A")
        print("Global spatio-temporal relative L2-error: N/A")
        print("Global spatio-temporal MSE: N/A")

    tag = now_tag()
    prefix = f"{eq.NAME}_{sample_mode}_{model_kind}_{weight_mode}_{tag}"
    loss_plot_path = os.path.join(out_dir, f"{prefix}_loss_curve.png")
    snapshot_path = os.path.join(out_dir, f"{prefix}_snapshot.png")
    gif_path = os.path.join(out_dir, f"{prefix}.gif")
    data_prefix_path = os.path.join(out_dir, prefix)
    pred_data_path = os.path.join(out_dir, f"{prefix}_field_data.npz")

    if SAVE_ARTIFACTS:
        final_iter = steps_adam + (steps_lbfgs if steps_lbfgs > 0 else 0)
        if len(history["iters_loss"]) == 0 or history["iters_loss"][-1] != final_iter:
            history["iters_loss"].append(final_iter)
            history["pde_loss"].append(float(lp))
            history["ic_loss"].append(float(li))
            history["bc_loss"].append(float(lb))
        if np.isfinite(avg_l2) and (len(history["iters_l2"]) == 0 or history["iters_l2"][-1] != final_iter):
            history["iters_l2"].append(final_iter)
            history["l2_loss"].append(float(avg_l2))
            history["global_l2_loss"].append(float(global_l2))
            history["mse_loss"].append(float(mse_val))
        plot_loss_curves(history, loss_plot_path, f"{eq.NAME}: {model_kind} + {weight_mode}")
        loss_csv, l2_csv, plot_npz = save_loss_history_data(history, data_prefix_path)
        saved_pred_path = save_prediction_field_data(eq, model, pred_data_path)

        print(f"Loss curve saved to: {loss_plot_path}")
        print(f"Loss data saved to:  {loss_csv}")
        print(f"L2 data saved to:    {l2_csv}")
        print(f"Plot npz saved to:   {plot_npz}")
        if saved_pred_path is not None:
            print(f"Field data saved to: {saved_pred_path}")

    record = {
        "equation": eq.NAME,
        "model_kind": model_kind,
        "weight_mode": weight_mode,
        "sample_mode": sample_mode,
        "active_terms": "+".join(active_names),
        "anchor_points": int(N_anchor) if USE_ANCHOR_LOSS else 0,
        "time_tag": tag,
        "training_time": float(train_time),
        "train_l2": float(train_l2) if np.isfinite(train_l2) else "N/A",
        "train_mse": float(train_mse) if np.isfinite(train_mse) else "N/A",
        "final_pde_loss": float(lp),
        "final_ic_loss": float(li),
        "final_bc_loss": float(lb),
        "avg_l2": float(avg_l2) if np.isfinite(avg_l2) else "N/A",
        "global_l2": float(global_l2) if np.isfinite(global_l2) else "N/A",
        "mse": float(mse_val) if np.isfinite(mse_val) else "N/A",
        "loss_lines": "\n".join(printed_lines),
    }
    if WRITE_LOG_XLSX:
        append_log_column(record)
    return {
        "train_l2": train_l2,
        "train_mse": train_mse,
        "time_avg_l2": avg_l2,
        "global_l2": global_l2,
        "mse": mse_val,
        "final_pde_loss": lp,
        "final_ic_loss": li,
        "final_bc_loss": lb,
        "training_time": train_time,
        "history": history,
        "active_names": active_names,
        "profile_trace": profile_trace + [(train_time, float(global_l2), "Final selected model", lbfgs_closure_calls)],
        "peak_allocated_mib": peak_allocated_bytes / 2**20,
        "peak_reserved_mib": peak_reserved_bytes / 2**20,
        "lbfgs_closure_calls": lbfgs_closure_calls,
        "lbfgs_optimizer_func_evals": int(lbfgs_state.get("func_evals", 0)),
        "lbfgs_iterations": int(lbfgs_state.get("n_iter", 0)),
    }


# =========================================================
# 9) Main
# =========================================================
def run_combo(eq_name, model_kind, weight_mode):
    if RESET_SEED_EACH_RUN:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    eq = importlib.import_module(eq_name)
    out_dir = os.path.join(ROOT_DIR, eq.NAME, "new_lossplot")
    os.makedirs(out_dir, exist_ok=True)
    print("-" * 90)
    print(
        "Lossplot run params | "
        f"equation={eq_name}, model_kind={model_kind}, weight_mode={weight_mode}, "
        f"N_f={N_f}, N_ic={N_ic if getattr(eq, 'HAS_IC', False) else 0}, "
        f"N_bc={N_bc if getattr(eq, 'HAS_BC', True) else 0}, "
        f"steps_adam={steps_adam}, steps_lbfgs={steps_lbfgs}, "
        f"fourier_freqs={fourier_freqs}, seed={seed}, dtype={TORCH_DTYPE}, "
        f"use_anchor={USE_ANCHOR_LOSS}"
    )
    print("-" * 90)
    return train_one(eq, model_kind, weight_mode, out_dir)



def main():
    if RUN_ALL:
        results = []
        for eq_name, model_kind, weight_mode in combos:
            metrics = run_combo(eq_name, model_kind, weight_mode)
            results.append((eq_name, model_kind, weight_mode, metrics))

        grouped = {}
        for eq_name, mk, wm, metrics in results:
            if not isinstance(metrics, dict) or "history" not in metrics:
                continue
            grouped.setdefault(eq_name, []).append({
                "model_kind": mk,
                "weight_mode": wm,
                "history": metrics["history"],
            })
        for eq_name, runs in grouped.items():
            eq = importlib.import_module(eq_name)
            out_dir = os.path.join(ROOT_DIR, eq.NAME, "new_lossplot")
            os.makedirs(out_dir, exist_ok=True)
            tag = now_tag()
            save_path = os.path.join(out_dir, f"{eq.NAME}_combined_lossplot_{tag}.png")
            plot_combined_loss_curves(runs, save_path, f"{eq.NAME}: Adam + L-BFGS loss comparison")
            print(f"Combined loss plot saved to: {save_path}")
            eq_results = [r for r in results if r[0] == eq_name]
            _, summary_tex_path = save_summary_tables(eq_name, eq_results, out_dir)
            if APPEND_TEX:
                append_summary_to_main_tex(eq_name, summary_tex_path)

        print("=" * 90)
        print("FINAL SUMMARY")
        print("=" * 90)
        for eq_name, mk, wm, metrics in results:
            if isinstance(metrics, dict) and np.isfinite(metrics.get("time_avg_l2", np.nan)):
                print(
                    f"{eq_name:22s} | {mk:12s} | {wm:12s} | "
                    f"time-avg L2 = {metrics['time_avg_l2']:.3e} | "
                    f"global L2 = {metrics['global_l2']:.3e} | "
                    f"MSE = {metrics['mse']:.3e}"
                )
            else:
                print(f"{eq_name:22s} | {mk:12s} | {wm:12s} | metrics = N/A")
    else:
        metrics = run_combo(run_equation, MODEL_KIND, WEIGHT_MODE)
        if isinstance(metrics, dict):
            print(
                f"Single run done. time-avg L2 = {metrics['time_avg_l2']:.3e}, "
                f"global L2 = {metrics['global_l2']:.3e}, MSE = {metrics['mse']:.3e}"
            )
        else:
            print(f"Single run done. metrics = {metrics}")


if __name__ == "__main__":
    main()
