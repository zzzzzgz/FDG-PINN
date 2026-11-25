"""Illustrative heat-equation test on the real line; no finite-x BCs."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import FDG_runner as runner


class MappedFourier(torch.nn.Module):
    def __init__(self, length=2.0):
        super().__init__()
        self.length = length
        self.encoder = runner.FourierFeatureEncoder([-1.0, 0.0], [1.0, 1.0], (1., 2., 4., 8.))

    def forward(self, xt):
        mapped = torch.cat((torch.tanh(xt[:, :1] / self.length), xt[:, 1:]), dim=1)
        return self.encoder(mapped)


def exact(x, t):
    return torch.exp(-x.square() / (1.0 + 4.0 * t)) / torch.sqrt(1.0 + 4.0 * t)


def sample(n, device, dtype):
    x = torch.randn(n, 1, device=device, dtype=dtype) * 2.0
    x[: n // 4] = torch.randn(n // 4, 1, device=device, dtype=dtype) * 4.0
    t = torch.rand(n, 1, device=device, dtype=dtype)
    return torch.cat((x, t), dim=1)


def metrics(model, device, dtype):
    out = {}
    with torch.no_grad():
        for name, a, b in [('center', -2., 2.), ('tails', 2., 6.), ('outer', 6., 10.)]:
            x = torch.linspace(a, b, 400, device=device, dtype=dtype)
            if name != 'center':
                x = torch.cat((-x.flip(0), x))
            t = torch.linspace(0, 1, 11, device=device, dtype=dtype)
            xx, tt = torch.meshgrid(x, t, indexing='ij')
            xt = torch.stack((xx.flatten(), tt.flatten()), dim=1)
            pred = model(xt).flatten()
            true = exact(xt[:, 0], xt[:, 1])
            out[name] = float(torch.linalg.vector_norm(pred - true) / torch.linalg.vector_norm(true).clamp_min(1e-12))
            out[name + '_rmse'] = float(torch.sqrt(torch.mean((pred - true).square())))
    return out


def run(kind, steps, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    encoder = MappedFourier().to(device=device, dtype=dtype)
    if kind == 'fdg':
        trunk = runner.DoubleGatePINN(encoder.encoder.out_dim, 1, hidden=96, n_hidden_layers=4)
    else:
        trunk = runner.BaseMLP(encoder.encoder.out_dim, 1, hidden=96, depth=4)
    model = runner.EncodedPINN(encoder, trunk).to(device=device, dtype=dtype)
    aw = runner.SigmaAWGeneric(2).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(list(model.parameters()) + list(aw.parameters()), lr=1e-3)
    colloc = sample(3000, device, dtype).detach().requires_grad_(True)
    ic = sample(500, device, dtype)
    ic[:, 1] = 0
    history = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        u = model(colloc)
        g = torch.autograd.grad(u, colloc, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(g[:, :1], colloc, torch.ones_like(g[:, :1]), create_graph=True)[0][:, :1]
        pde = (g[:, 1:] - uxx).square().mean()
        initial = (model(ic) - exact(ic[:, :1], ic[:, 1:])).square().mean()
        loss, _ = aw.total_loss(torch.stack((pde, initial)))
        loss.backward()
        opt.step()
        aw.clamp_()
        if step == 1 or step % 100 == 0 or step == steps:
            errors = metrics(model, device, dtype)
            history.append({'method': kind, 'step': step, 'pde': float(pde.detach()), 'ic': float(initial.detach()), **errors})
            print(history[-1], flush=True)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1111)
    args = parser.parse_args()
    rows = run('ff', args.steps, args.seed) + run('fdg', args.steps, args.seed)
    output = Path(__file__).resolve().parent
    with (output / 'pilot_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / 'pilot_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
