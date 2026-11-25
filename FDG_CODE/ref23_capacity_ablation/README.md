# Ref. [23] / capacity-matched gate ablation

This folder is isolated from the submitted manuscript and existing six-equation
single-gate results. It reuses the equation modules and training/evaluation
pipeline in `../../FDG-PINN/FDG_CODE/FDG_runner.py`.

## Predefined variants

- `ref23_equal_width`: Eq. (43)--(47) of Wang, Teng, and Perdikaris (2021),
  *Understanding and mitigating gradient flow pathologies in physics-informed
  neural networks*, with width 96. Its `U`/`V` projections and gate use `tanh`.
- `ref23_matched_params`: same Ref. [23] architecture, with the width selected
  solely by the closest trainable-parameter count to FDG-PINN.
- `first_gate_matched_params`: the first-gate-only FDG ablation (deeper `U`/`V`
  branches and sigmoid gate), also selected by closest parameter count.

All variants use the same Fourier encoder `{1,2,4,8}`, adaptive loss weighting,
float64 precision, sampling, Adam/L-BFGS settings, and the predefined seeds
1111, 2222, 3333, 4444, and 5555. The original FDG-PINN runs are reused as
the comparator. For the current 2-input/1-output cases, FDG-PINN has 108,961
trainable parameters. The three variants have 33,505 (width 96), 109,325
(width 181), and 108,721 (width 157), respectively. Matching is within 0.4%
for the latter two. The equal-width comparison is not capacity matched.

## Completed observations

`raw_runs.csv` contains one row per completed run. `summary_mean_std.csv`
uses sample standard deviations; rows with `n=1` are pilot observations, not
multi-seed statistics. All runs below are on the RTX 4070 Ti SUPER.

| Equation | Variant | Completed seeds | Mean Global L2 | Sample SD | Mean training time (s) |
|---|---|---:|---:|---:|---:|
| Transport | Ref. [23], equal width | 5 | 1.929e-3 | 2.042e-4 | 141.4 |
| Transport | Ref. [23], matched parameters | 5 | 3.057e-3 | 2.534e-3 | 236.6 |
| Transport | First gate, matched parameters | 5 | 3.003e-3 | 2.255e-4 | 234.9 |
| Helmholtz k=20 | Ref. [23], equal width | 5 | 1.509e-2 | 4.709e-3 | 210.3 |
| Burgers | Ref. [23], matched parameters | 5 | 1.561e-3 | 6.183e-4 | 390.6 |
| Burgers | First gate, matched parameters | 5 | 1.915e-3 | 2.413e-3 | 400.7 |
| Reaction-diffusion | Ref. [23], matched parameters | 5 | 1.647e-3 | 4.381e-4 | 369.8 |
| Reaction-diffusion | First gate, matched parameters | 5 | 9.262e-4 | 7.412e-4 | 384.2 |
| Helmholtz k=10 | Ref. [23], matched parameters | 5 | 6.595e-2 | 2.255e-2 | 504.5 |
| Helmholtz k=10 | First gate, matched parameters | 5 | 1.670e-2 | 5.624e-3 | 479.4 |
| Mixed-frequency Helmholtz | Ref. [23], matched parameters | 5 | 2.091e-2 | 3.832e-3 | 505.4 |
| Mixed-frequency Helmholtz | First gate, matched parameters | 5 | 2.126e-2 | 1.037e-2 | 499.1 |
| Helmholtz k=20 | Ref. [23], matched parameters | 5 | 1.938e-2 | 7.728e-3 | 513.5 |
| Helmholtz k=20 | First gate, matched parameters | 5 | 7.784e-3 | 1.850e-3 | 538.5 |

The Burgers and reaction-diffusion capacity-matched runs use all five seeds.
Their existing FDG-PINN five-seed Global L2 results are 4.963e-4 +/-
9.854e-5 and 3.329e-4 +/- 3.641e-5, respectively. Paired by seed,
FDG-PINN is lower than the matched Ref. [23] variant on 5/5 seeds for each
equation, and lower than the matched first-gate variant on 4/5 Burgers seeds
and 5/5 reaction-diffusion seeds. Burgers first-gate seed 2222 is an
outlier (Global L2 = 6.219e-3); consult `raw_runs.csv` rather than only
the mean. The FDG-PINN baseline records are in
`../../FDG-PINN/FDG_CODE/<equation>/average_seed1111_5555/raw_runs.csv`.
`raw_runs_before_burgers_rd.csv` and `summary_before_burgers_rd.csv`
preserve the prior 22 completed runs independently of this extension.

For context, the existing FDG-PINN five-seed mean Global L2 is 2.640e-3
for Transport, 1.360e-2 for Helmholtz k=10, 1.476e-2 for mixed-frequency
Helmholtz, and 7.716e-3 for Helmholtz k=20. Ref. [23] at equal width
outperforms FDG-PINN on Transport; FDG-PINN outperforms both capacity-matched
new variants there. In the three remaining Helmholtz equations, FDG-PINN has
lower mean Global L2 than the matched Ref. [23] variant. Against the matched
first-gate variant, FDG-PINN has lower means on k=10 and mixed-frequency
Helmholtz and k=20, although the k=20 difference is small (7.716e-3 versus
7.784e-3, about 0.9%). Paired FDG-PINN wins against the first-gate variant
are 3/5 seeds in each of the three Helmholtz equations.
This comparison does not establish a general advantage for FDG-PINN.

The earlier Helmholtz k=20 seed-2222 run was interrupted before a record was
written. The same setting was rerun successfully, and both matched variants
now have five completed seeds. The 42-row pre-Helmholtz state is preserved in
`raw_runs_before_remaining_helmholtz.csv` and
`summary_before_remaining_helmholtz.csv`.

## Resume

The script skips completed `(equation, seed, variant)` entries:

```powershell
& 'F:\Anaconda\envs\PINN\python.exe' 'E:\workpackage\PINN\FDG_CODE\ref23_capacity_ablation\run_ablation.py'
```

No experiment from this folder has yet been incorporated into the manuscript
or the response to reviewers.
