import numpy as np
import torch

NAME = "fdg_transport1d_eq"
INPUT_DIM = 2
OUTPUT_DIM = 1
PLOT_STYLE = "1d_xt"
TRUE_AVAILABLE = True
HAS_IC = True
HAS_BC = True

x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0
c = 0.7


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


def true_solution_np(x, t):
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    u = (
        np.sin(2.0 * np.pi * (4.0 * x - 1.5 * t))
        + 0.35 * np.cos(2.0 * np.pi * (8.0 * x + 0.5 * t))
        + 0.20 * np.sin(2.0 * np.pi * (2.0 * x + 3.0 * t))
    )
    return u.astype(np.float32).reshape(-1, 1)


def exact_torch(xt):
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return (
        torch.sin(2.0 * np.pi * (4.0 * x - 1.5 * t))
        + 0.35 * torch.cos(2.0 * np.pi * (8.0 * x + 0.5 * t))
        + 0.20 * torch.sin(2.0 * np.pi * (2.0 * x + 3.0 * t))
    )


def source_torch(xt):
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    two_pi = 2.0 * np.pi

    a1 = two_pi * (4.0 * x - 1.5 * t)
    a2 = two_pi * (8.0 * x + 0.5 * t)
    a3 = two_pi * (2.0 * x + 3.0 * t)

    u_t = (
        -1.5 * two_pi * torch.cos(a1)
        -0.35 * 0.5 * two_pi * torch.sin(a2)
        +0.20 * 3.0 * two_pi * torch.cos(a3)
    )
    u_x = (
        4.0 * two_pi * torch.cos(a1)
        -0.35 * 8.0 * two_pi * torch.sin(a2)
        +0.20 * 2.0 * two_pi * torch.cos(a3)
    )
    return u_t + c * u_x


def sample_collocation(n, device, sample_mode):
    arr = _sample_box(n, [(x_min, x_max), (t_min, t_max)], sample_mode)
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=True)


def sample_ic(n, device, sample_mode):
    x = _sample_box(n, [(x_min, x_max)], sample_mode)
    t = np.zeros((n, 1), dtype=np.float32)
    xt = torch.tensor(np.concatenate([x, t], axis=1), dtype=torch.float32, device=device)
    return {"xt": xt, "target": exact_torch(xt).detach()}


def sample_bc(n, device, sample_mode):
    nb = max(1, n // 2)
    t = _sample_box(nb, [(t_min, t_max)], sample_mode)[:, 0:1]
    left = torch.tensor(np.concatenate([np.full_like(t, x_min), t], axis=1), dtype=torch.float32, device=device)
    right = torch.tensor(np.concatenate([np.full_like(t, x_max), t], axis=1), dtype=torch.float32, device=device)
    return {
        "left": left,
        "right": right,
        "target_left": exact_torch(left).detach(),
        "target_right": exact_torch(right).detach(),
    }


def pde_residual(model, xt, grad):
    u = model(xt)
    gu = grad(u, xt)
    u_x = gu[:, 0:1]
    u_t = gu[:, 1:2]
    return u_t + c * u_x - source_torch(xt)


def ic_loss(model, ic_data, mse):
    return mse(model(ic_data["xt"]), ic_data["target"])


def bc_loss(model, bc_data, mse):
    return mse(model(bc_data["left"]), bc_data["target_left"]) + mse(model(bc_data["right"]), bc_data["target_right"])


def make_plot_grid(Nx=600):
    return np.linspace(x_min, x_max, Nx, dtype=np.float32)
