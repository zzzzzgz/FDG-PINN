"""Burgers front benchmark with shared front-aware collocation sampling."""

import numpy as np
import torch

import fdg_burgers_front1d_eq as base


NAME = "fdg_burgers_front1d_focused_eq"
INPUT_DIM = base.INPUT_DIM
OUTPUT_DIM = base.OUTPUT_DIM
PLOT_STYLE = base.PLOT_STYLE
TRUE_AVAILABLE = base.TRUE_AVAILABLE
HAS_IC = base.HAS_IC
HAS_BC = base.HAS_BC
HARD_IC_OUTPUT = base.HARD_IC_OUTPUT
x_min, x_max = base.x_min, base.x_max
t_min, t_max = base.t_min, base.t_max
nu = base.nu

true_solution_np = base.true_solution_np
sample_ic = base.sample_ic
sample_bc = base.sample_bc
pde_residual = base.pde_residual
ic_loss = base.ic_loss
bc_loss = base.bc_loss
make_plot_grid = base.make_plot_grid


def sample_collocation(n, device, sample_mode):
    n_focused = n // 2
    uniform = base._sample_box(n - n_focused,
                               [(x_min, x_max), (t_min, t_max)], sample_mode)
    focused_x = np.clip(np.random.normal(0.0, 0.12, n_focused),
                        x_min, x_max).astype(np.float32)
    focused_t = (0.2 + 0.8 * np.random.rand(n_focused)).astype(np.float32)
    focused = np.column_stack((focused_x, focused_t))
    points = np.concatenate((uniform, focused), axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)
