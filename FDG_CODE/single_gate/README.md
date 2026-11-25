# Single-gate ablation

This directory contains a first-gate-only comparison for the six PDE benchmarks
in the submitted ND manuscript. It does not modify the manuscript or the
published FDG-PINN results.

## Definition and protocol

`run_single_gate.py` reuses the equation modules, Fourier encoder, adaptive
weighting, optimizer, sampling, evaluation, and training-time measurement in
`../../FDG-PINN/FDG_CODE/FDG_runner.py`. The new trunk keeps the same `U` and
`V` branch definitions and the first, state-dependent interpolation gate:

```text
B_k = sigmoid(W_B,k z_k + b_B,k)
z_(k+1) = (1 - B_k) * U + B_k * V
```

The second reference-fusion gate and its ordinary candidate path are removed.
This is a first-gate-only ablation, not a parameter-matched comparison or an
implementation claimed to be identical to Ref. [23]. It has 53,089 trainable
parameters for the 2-input/1-output benchmarks, versus 108,961 for the
corresponding FDG-PINN. Both use the same Fourier set `{1,2,4,8}`, adaptive
weighting, `float64`, LHS sampling, 2,000 Adam steps, up to 1,000 L-BFGS steps,
and seeds `1111,2222,3333,4444,5555`. Runs used an RTX 4070 Ti SUPER.

Training time is the existing runner's wall-clock measure from the beginning
of training through the end of optimization, with CUDA synchronization before
the timer stops. It excludes model construction, sample generation, final
evaluation, and artifact generation.

## Results

All entries are means over five seeds. `raw_runs.csv` has every seed and
`summary_mean_std.csv` includes sample standard deviations and all recorded
loss/L2 metrics. FF-PINN and FDG-PINN comparison values below come from their
existing `average_seed1111_5555` folders; k=10 uses its
`average_seed1111_5555_probe` folder.

| Equation | Single-gate Global L2 | FF-PINN Global L2 | FDG-PINN Global L2 | Single-gate time (s) | Single-gate wins vs FDG (paired seeds) |
|---|---:|---:|---:|---:|---:|
| Transport | 3.225e-3 | 4.036e-3 | 2.640e-3 | 160.5 | 0/5 |
| Burgers | 5.799e-3 | 8.676e-4 | 4.963e-4 | 200.1 | 0/5 |
| Helmholtz k=10 | 8.814e-3 | 1.362e-2 | 1.360e-2 | 329.8 | 4/5 |
| Helmholtz k=20 | 5.674e-3 | 7.916e-3 | 7.716e-3 | 318.0 | 5/5 |
| Mixed-frequency Helmholtz | 1.929e-2 | 1.951e-2 | 1.476e-2 | 305.3 | 1/5 |
| Reaction-diffusion | 1.167e-3 | 6.211e-4 | 3.329e-4 | 243.2 | 0/5 |

FDG-PINN has a lower mean error on four of six benchmarks. The single-gate
model has a lower mean error on both k=10 and k=20 Helmholtz benchmarks, and
is faster on all six. These results do **not** support a universal claim that
the second gate improves accuracy. They also cannot isolate a pure gate effect
from parameter count or other structural differences. A capacity-matched and
Ref. [23] comparison remains necessary for the stronger reviewer question.

## Reproduction

Run from any directory with the project's PINN environment:

```powershell
& 'F:\Anaconda\envs\PINN\python.exe' 'E:\workpackage\PINN\FDG_CODE\single_gate\run_single_gate.py'
```

The script skips already recorded `(equation, seed)` pairs and writes each new
run immediately. It does not overwrite the existing baselines or manuscript.
