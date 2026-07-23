import numpy as np
import torch

NAME = "fdg_reaction_diffusion_mixed1d_eq"
INPUT_DIM = 2
OUTPUT_DIM = 1
PLOT_STYLE = "1d_xt"
TRUE_AVAILABLE = True
HAS_IC = True
HAS_BC = True

x_min, x_max = 0.0, float(np.pi)
t_min, t_max = 0.0, 1.0
nu = 0.01
lam = 0.5


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
    u = np.exp(-t) * (np.sin(x) + 0.10 * np.sin(20.0 * x))
    return u.astype(np.float32).reshape(-1, 1)


def exact_torch(xt):
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return torch.exp(-t) * (torch.sin(x) + 0.10 * torch.sin(20.0 * x))


def source_torch(xt):
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    exp_t = torch.exp(-t)
    low = torch.sin(x)
    high = 0.10 * torch.sin(20.0 * x)
    u = exp_t * (low + high)
    u_t = -u
    u_xx = exp_t * (-low - (20.0 ** 2) * high)
    return u_t - nu * u_xx + lam * u


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
    u_xx = grad(u_x, xt)[:, 0:1]
    return u_t - nu * u_xx + lam * u - source_torch(xt)


def ic_loss(model, ic_data, mse):
    return mse(model(ic_data["xt"]), ic_data["target"])


def bc_loss(model, bc_data, mse):
    return mse(model(bc_data["left"]), bc_data["target_left"]) + mse(model(bc_data["right"]), bc_data["target_right"])


def make_plot_grid(Nx=600):
    return np.linspace(x_min, x_max, Nx, dtype=np.float32)
