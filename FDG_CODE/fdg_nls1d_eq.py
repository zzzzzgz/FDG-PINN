import numpy as np
import torch

NAME = "fdg_nls1d_eq"
INPUT_DIM = 2
OUTPUT_DIM = 2
PLOT_STYLE = "1d_xt"
TRUE_AVAILABLE = True
HAS_IC = True
HAS_BC = True

x_min, x_max = -5.0, 5.0
t_min, t_max = 0.0, 1.0


def _sech_np(z):
    z = np.asarray(z, dtype=np.float32)
    ez = np.exp(-np.abs(z))
    return 2.0 * ez / (1.0 + ez * ez)


def _sech_torch(z):
    ez = torch.exp(-torch.abs(z))
    return 2.0 * ez / (1.0 + ez * ez)


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
    amp = _sech_np(x)
    phase = 0.5 * t
    u = amp * np.cos(phase)
    v = amp * np.sin(phase)
    return np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)


def exact_torch(xt):
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    amp = _sech_torch(x)
    phase = 0.5 * t
    return torch.cat([amp * torch.cos(phase), amp * torch.sin(phase)], dim=1)


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
    uv = model(xt)
    u = uv[:, 0:1]
    v = uv[:, 1:2]

    gu = grad(u, xt)
    gv = grad(v, xt)
    u_x = gu[:, 0:1]
    u_t = gu[:, 1:2]
    v_x = gv[:, 0:1]
    v_t = gv[:, 1:2]
    u_xx = grad(u_x, xt)[:, 0:1]
    v_xx = grad(v_x, xt)[:, 0:1]
    r2 = u * u + v * v

    real_res = -v_t + 0.5 * u_xx + r2 * u
    imag_res = u_t + 0.5 * v_xx + r2 * v
    return torch.cat([real_res, imag_res], dim=1)


def ic_loss(model, ic_data, mse):
    return mse(model(ic_data["xt"]), ic_data["target"])


def bc_loss(model, bc_data, mse):
    left_loss = mse(model(bc_data["left"]), bc_data["target_left"])
    right_loss = mse(model(bc_data["right"]), bc_data["target_right"])
    return left_loss + right_loss


def make_plot_grid(Nx=600):
    return np.linspace(x_min, x_max, Nx, dtype=np.float32)
