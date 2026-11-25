# FDG-PINN

This repository provides the source code and experimental data associated with the Fourier Double-Gate Physics-Informed Neural Network (FDG-PINN).

FDG-PINN integrates multiscale Fourier feature encoding, a Double-Gate neural network architecture, and adaptive loss weighting to improve the approximation of oscillatory, nonlinear, and mixed-frequency solutions to partial differential equations (PDEs). The repository includes analytical PDE benchmark experiments and real-world temperature-field reconstruction experiments.

## Repository Structure

### `FDG_CODE/`

Source code for the analytical PDE benchmark experiments, including:

* One-dimensional transport equation
* Helmholtz equations
* Reaction–diffusion equation
* Nonlinear Burgers equation
* Nonlinear Schrödinger equation
* Ablation and comparative experiments

These experiments are used to evaluate the accuracy and adaptability of FDG-PINN across different equation structures and solution characteristics.

### `Temperature_forecast_huabei_50day/`

Code and processed data for the 50-day temperature-field reconstruction experiment over **North China**.

This experiment evaluates the capability of FDG-PINN to reconstruct continuous spatiotemporal temperature fields from ERA5-Land data.

### `Temperature_forecast_india/`

Code and processed data for the temperature-field reconstruction experiment over **Northern South Asia**.

### `MANIFEST_FDG_PINN.json`

An inventory of the files included in the released research package.

## Experimental Settings

For the analytical PDE benchmarks, the experiments use a unified frequency set, network architecture, and broadly consistent optimization procedure wherever applicable:

* Fourier frequency set: `{1, 2, 4, 8}`
* Optimization: Adam followed by L-BFGS
* Framework: PyTorch

Specific equation definitions, sampling settings, loss functions, and training parameters are provided in the corresponding source files.

## Reproducibility

To facilitate the reproduction of the reported results, experiments involving multiple independent runs use the random seeds `1111`, `2222`, `3333`, `4444`, and `5555`. Experiments conducted with a single random seed use `2026`.

## Temperature-Field Data

The real-world temperature-field experiments use the **ERA5-Land Hourly** open dataset provided by the European Centre for Medium-Range Weather Forecasts (ECMWF).

The dataset can be accessed through the Google Earth Engine Data Catalog:

https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY

The image collection used in the experiments is:

```text
ECMWF/ERA5_LAND/HOURLY
```
