"""Localized high-frequency transport on x in R, without finite-x BCs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from pilot import MappedFourier, sample, runner


class ScaledFourier(MappedFourier):
    def forward(self, xt):
        mapped = torch.cat((xt[:, :1] / self.length, xt[:, 1:]), dim=1)
        return self.encoder(mapped)


def exact(x, t):
    xi = x - 0.7 * t
    envelope = torch.exp(-xi.square() / 4.0)
    return envelope * (
        torch.sin(8 * torch.pi * xi)
        + 0.35 * torch.cos(16 * torch.pi * xi)
        + 0.20 * torch.sin(4 * torch.pi * xi)
    )


def metrics(model, device, dtype):
    out = {}
    with torch.no_grad():
        for name, a, b in [('center', -2., 2.), ('tails', 2., 6.), ('outer', 6., 10.)]:
            x = torch.linspace(a, b, 500, device=device, dtype=dtype)
            if name != 'center':
                x = torch.cat((-x.flip(0), x))
            t = torch.linspace(0, 1, 11, device=device, dtype=dtype)
            xx, tt = torch.meshgrid(x, t, indexing='ij')
            xt = torch.stack((xx.flatten(), tt.flatten()), dim=1)
            pred = model(xt).flatten()
            true = exact(xt[:, 0], xt[:, 1])
            out[name + '_l2'] = float(torch.linalg.vector_norm(pred - true) / torch.linalg.vector_norm(true).clamp_min(1e-12))
            out[name + '_rmse'] = float(torch.sqrt(torch.mean((pred - true).square())))
    return out


def save_field_plot(model, device, dtype, output):
    x = torch.linspace(-6.0, 6.0, 1201, device=device, dtype=dtype)
    t = torch.full_like(x, 0.5)
    xt = torch.stack((x, t), dim=1)
    with torch.no_grad():
        pred = model(xt).flatten().cpu().numpy()
        true = exact(x, t).cpu().numpy()
    x_np = x.cpu().numpy()
    error = pred - true
    np.savetxt(output / 'transport_linear_fdg_t05_field.csv',
               np.column_stack((x_np, pred, true, error)), delimiter=',',
               header='x,pred,true,pred_minus_true', comments='')
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), sharex=True, layout='constrained')
    limits = (min(pred.min(), true.min()), max(pred.max(), true.max()))
    for ax, title, values, color in zip(
            axes, ('Pred', 'True', 'Error'), (pred, true, error),
            ('#2678b6', '#238b45', '#cb3c33')):
        ax.plot(x_np, values, color=color, lw=1.4)
        ax.set(title=title, xlabel='x', xlim=(-6, 6))
        ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        ax.grid(alpha=0.2)
        if title != 'Error':
            ax.set_ylim(*limits)
    for suffix in ('svg', 'png'):
        fig.savefig(output / f'transport_linear_fdg_t05_field.{suffix}', dpi=220)
    plt.close(fig)


def run(kind, steps, seed, mapping, plot=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    encoder = (MappedFourier(length=2.0) if mapping == 'tanh' else ScaledFourier(length=1.0)).to(device=device, dtype=dtype)
    if kind == 'fdg':
        trunk = runner.DoubleGatePINN(encoder.encoder.out_dim, 1, hidden=96, n_hidden_layers=4)
    else:
        trunk = runner.BaseMLP(encoder.encoder.out_dim, 1, hidden=96, depth=4)
    model = runner.EncodedPINN(encoder, trunk).to(device=device, dtype=dtype)
    aw = runner.SigmaAWGeneric(2).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aw.parameters()), lr=1e-3)
    colloc = sample(10000, device, dtype).detach().requires_grad_(True)
    ic = sample(2000, device, dtype)
    ic[:, 1] = 0
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        u = model(colloc)
        deriv = torch.autograd.grad(u, colloc, torch.ones_like(u), create_graph=True)[0]
        pde = (deriv[:, 1:] + 0.7 * deriv[:, :1]).square().mean()
        initial = (model(ic) - exact(ic[:, :1], ic[:, 1:])).square().mean()
        loss, _ = aw.total_loss(torch.stack((pde, initial)))
        loss.backward()
        optimizer.step()
        aw.clamp_()
        if step == 1 or step % 200 == 0 or step == steps:
            row = {'method': kind, 'step': step, 'pde': float(pde.detach()),
                   'ic': float(initial.detach()), **metrics(model, device, dtype)}
            history.append(row)
            print(row, flush=True)
    if plot and kind == 'fdg' and mapping == 'linear':
        save_field_plot(model, device, dtype, Path(__file__).resolve().parent)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--mapping', choices=('tanh', 'linear'), default='tanh')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.plot:
        if args.mapping != 'linear':
            parser.error('--plot currently supports only --mapping linear')
        rows = run('fdg', args.steps, args.seed, args.mapping, plot=True)
    else:
        rows = run('ff', args.steps, args.seed, args.mapping) + run('fdg', args.steps, args.seed, args.mapping)
    output = Path(__file__).resolve().parent
    suffix = '_plot_run' if args.plot else ''
    with (output / f'transport_{args.mapping}_pilot_metrics{suffix}.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / f'transport_{args.mapping}_pilot_config{suffix}.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
