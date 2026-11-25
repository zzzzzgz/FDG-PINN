"""Unforced viscous Burgers front-steepening benchmark.

The reference is evaluated by a positive Cole--Hopf heat-kernel quadrature;
only initial and boundary values, not reference interior values, train the PINN.
"""

from functools import lru_cache

import numpy as np
import torch


NAME = "fdg_burgers_front1d_eq"
INPUT_DIM = 2
OUTPUT_DIM = 1
PLOT_STYLE = "1d_xt"
TRUE_AVAILABLE = True
HAS_IC = True
HAS_BC = True
HARD_IC_OUTPUT = False

x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0
nu = 0.01 / np.pi

_Z = np.linspace(-15.0, 15.0, 1201, dtype=np.float64)
_GAUSSIAN_LOG = -0.5 * _Z * _Z


def _lhs(n, dim):
    seg = np.linspace(0.0, 1.0, n + 1)
    pts = np.zeros((n, dim), dtype=np.float32)
    for j in range(dim):
        u = np.random.rand(n)
        pts[:, j] = seg[:-1] + u * (seg[1:] - seg[:-1])
        np.random.shuffle(pts[:, j])
    return pts


def _sample_box(n, bounds, sample_mode):
    s = _lhs(n, len(bounds)) if sample_mode == "LHS" else np.random.rand(n, len(bounds)).astype(np.float32)
    out = np.zeros_like(s, dtype=np.float32)
    for j, (a, b) in enumerate(bounds):
        out[:, j] = a + (b - a) * s[:, j]
    return out


def cole_hopf_reference(x, t, z=None):
    """Evaluate u=-2 nu (log phi)_x using Gaussian heat-kernel quadrature."""
    x, t = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                               np.asarray(t, dtype=np.float64))
    if np.any(t < 0.0):
        raise ValueError("Time must be nonnegative")
    flat_x, flat_t = x.ravel(), t.ravel()
    values = np.empty(flat_x.size, dtype=np.float64)
    nodes = _Z if z is None else np.asarray(z, dtype=np.float64)
    gaussian_log = _GAUSSIAN_LOG if z is None else -0.5 * nodes * nodes
    initial = flat_t == 0.0
    values[initial] = -np.sin(np.pi * flat_x[initial])
    pending = np.flatnonzero(~initial)
    for start in range(0, len(pending), 256):
        take = pending[start:start + 256]
        y = flat_x[take, None] + np.sqrt(2.0 * nu * flat_t[take, None]) * nodes[None, :]
        phase = np.pi * y
        log_terms = gaussian_log[None, :] - 50.0 * np.cos(phase)
        scaled = np.exp(log_terms - np.max(log_terms, axis=1, keepdims=True))
        values[take] = -np.sum(scaled * np.sin(phase), axis=1) / np.sum(scaled, axis=1)
    return values.reshape(x.shape)


@lru_cache(maxsize=32)
def _evaluation_slice(t, n):
    xs = np.linspace(x_min, x_max, n, dtype=np.float32)
    return cole_hopf_reference(xs, float(t)).astype(np.float32).reshape(-1, 1)


def true_solution_np(x, t):
    x_array = np.asarray(x)
    t_array = np.asarray(t)
    if t_array.ndim == 0 and x_array.ndim == 1 and x_array.size == 600:
        reference_grid = np.linspace(x_min, x_max, 600, dtype=np.float32)
        if np.array_equal(x_array, reference_grid):
            return _evaluation_slice(float(t_array), 600).copy()
    return cole_hopf_reference(x_array, t_array).astype(np.float32).reshape(-1, 1)


def sample_collocation(n, device, sample_mode):
    arr = _sample_box(n, [(x_min, x_max), (t_min, t_max)], sample_mode)
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=True)


def sample_ic(n, device, sample_mode):
    x = _sample_box(n, [(x_min, x_max)], sample_mode)
    xt = torch.tensor(np.concatenate([x, np.zeros_like(x)], axis=1),
                      dtype=torch.float32, device=device)
    return {"xt": xt, "target": -torch.sin(np.pi * xt[:, 0:1]).detach()}


def sample_bc(n, device, sample_mode):
    count = max(1, n // 2)
    t = _sample_box(count, [(t_min, t_max)], sample_mode)[:, 0:1]
    left = torch.tensor(np.concatenate([np.full_like(t, x_min), t], axis=1),
                        dtype=torch.float32, device=device)
    right = torch.tensor(np.concatenate([np.full_like(t, x_max), t], axis=1),
                         dtype=torch.float32, device=device)
    return {"left": left, "right": right,
            "target_left": torch.zeros_like(left[:, 0:1]),
            "target_right": torch.zeros_like(right[:, 0:1])}


def pde_residual(model, xt, grad):
    u = model(xt)
    derivatives = grad(u, xt)
    u_x, u_t = derivatives[:, 0:1], derivatives[:, 1:2]
    u_xx = grad(u_x, xt)[:, 0:1]
    return u_t + u * u_x - nu * u_xx


def ic_loss(model, ic_data, mse):
    return mse(model(ic_data["xt"]), ic_data["target"])


def bc_loss(model, bc_data, mse):
    return mse(model(bc_data["left"]), bc_data["target_left"]) + mse(
        model(bc_data["right"]), bc_data["target_right"])


def make_plot_grid(Nx=600):
    return np.linspace(x_min, x_max, Nx, dtype=np.float32)
