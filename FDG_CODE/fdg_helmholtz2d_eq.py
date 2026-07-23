import numpy as np
import torch

NAME = "fdg_helmholtz2d_eq"
INPUT_DIM = 2
OUTPUT_DIM = 1
PLOT_STYLE = "2d_xy"
TRUE_AVAILABLE = True
HAS_IC = False
HAS_BC = True

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
k = 5.0
residual_scale = 1000.0


def _lhs(n, dim):
    seg = np.linspace(0.0, 1.0, n + 1)
    pts = np.zeros((n, dim), dtype=np.float32)
    for j in range(dim):
        u = np.random.rand(n)
        pts[:, j] = seg[:-1] + u * (seg[1:] - seg[:-1])
        np.random.shuffle(pts[:, j])
    return pts


def _sample_box(n, bounds, sample_mode):
    dim = len(bounds)
    s = _lhs(n, dim) if sample_mode == "LHS" else np.random.rand(n, dim).astype(np.float32)
    out = np.zeros_like(s, dtype=np.float32)
    for j, (a, b) in enumerate(bounds):
        out[:, j] = a + (b - a) * s[:, j]
    return out


def true_solution_np(X, Y):
    u = (
        np.sin(2.0 * np.pi * 4.0 * X) * np.cos(2.0 * np.pi * 3.0 * Y)
        + 0.35 * np.cos(2.0 * np.pi * 8.0 * X) * np.sin(2.0 * np.pi * 2.0 * Y)
    )
    return u.astype(np.float32)[..., None]


def exact_torch(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return (
        torch.sin(2.0 * np.pi * 4.0 * x) * torch.cos(2.0 * np.pi * 3.0 * y)
        + 0.35 * torch.cos(2.0 * np.pi * 8.0 * x) * torch.sin(2.0 * np.pi * 2.0 * y)
    )


def source_torch(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    two_pi = 2.0 * np.pi

    u1 = torch.sin(two_pi * 4.0 * x) * torch.cos(two_pi * 3.0 * y)
    u2 = 0.35 * torch.cos(two_pi * 8.0 * x) * torch.sin(two_pi * 2.0 * y)
    lap_u1 = -((two_pi * 4.0) ** 2 + (two_pi * 3.0) ** 2) * u1
    lap_u2 = -((two_pi * 8.0) ** 2 + (two_pi * 2.0) ** 2) * u2
    return lap_u1 + lap_u2 + (k ** 2) * (u1 + u2)


def sample_collocation(n, device, sample_mode):
    arr = _sample_box(n, [(x_min, x_max), (y_min, y_max)], sample_mode)
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=True)


def sample_ic(n, device, sample_mode):
    return None


def sample_bc(n, device, sample_mode):
    nb = max(1, n // 4)
    x = _sample_box(nb, [(x_min, x_max)], sample_mode)[:, 0:1]
    y = _sample_box(nb, [(y_min, y_max)], sample_mode)[:, 0:1]
    left = np.concatenate([np.full_like(y, x_min), y], axis=1)
    right = np.concatenate([np.full_like(y, x_max), y], axis=1)
    bottom = np.concatenate([x, np.full_like(x, y_min)], axis=1)
    top = np.concatenate([x, np.full_like(x, y_max)], axis=1)
    pts = np.concatenate([left, right, bottom, top], axis=0)
    xy = torch.tensor(pts, dtype=torch.float32, device=device)
    return {"xy": xy, "target": exact_torch(xy).detach()}


def pde_residual(model, xy, grad):
    u = model(xy)
    gu = grad(u, xy)
    u_x = gu[:, 0:1]
    u_y = gu[:, 1:2]
    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]
    return (u_xx + u_yy + (k ** 2) * u - source_torch(xy)) / residual_scale


def ic_loss(model, ic_data, mse):
    return torch.tensor(0.0, dtype=torch.float32, device=next(model.parameters()).device)


def bc_loss(model, bc_data, mse):
    return mse(model(bc_data["xy"]), bc_data["target"])


def make_plot_grid(Nx=140, Ny=140):
    xs = np.linspace(x_min, x_max, Nx, dtype=np.float32)
    ys = np.linspace(y_min, y_max, Ny, dtype=np.float32)
    return np.meshgrid(xs, ys)
