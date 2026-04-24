# 🌀 NestAR — Nested Sampling Autoregressive Modelling

> **Bayesian ARIMA model selection for astronomical time-series analysis using Nested Sampling.**

[![arXiv](https://img.shields.io/badge/arXiv-2512.01929-b31b1b.svg)](https://arxiv.org/abs/2512.01929)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19567180.svg)](https://doi.org/10.5281/zenodo.19567180)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![BlackJAX](https://img.shields.io/badge/sampler-BlackJAX-purple.svg)](https://github.com/blackjax-devs/blackjax)

---

## Overview

NestAR combines **ARIMA models** with the **Nested Sampling algorithm** to tackle one of the core challenges in time-series analysis: *how do you pick the right model without overfitting?*

Classical approaches like AIC/BIC rely on maximum likelihood optimisation, which can be biased — particularly for complex likelihood landscapes. NestAR takes a fully Bayesian route: it computes **log Bayesian evidences** for every candidate ARIMA(p, d, q) on a user-defined grid, and uses the evidence ratio (Bayes factor) to drive model selection. This comes with a built-in **Occam's penalty** that naturally disfavours unnecessarily complex models — no manual regularisation required.

The sampler is built on [BlackJAX](https://github.com/blackjax-devs/blackjax) with JAX as the computational backend, enabling GPU-accelerated inference across full (p, q) grids in a single run.

**Paper:**
> Naik, A. & Handley, W. (2025). *Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis.* arXiv:[2512.01929](https://arxiv.org/abs/2512.01929)

**Data & Notebooks:**
> All data, NS chains, and reproduction notebooks are archived on Zenodo: [10.5281/zenodo.19567180](https://doi.org/10.5281/zenodo.19567180)

---

## Key Features

- **Bayesian model selection** — compute log-evidences and posterior model probabilities across a full (p, q) grid
- **Intrinsic Occam's razor** — complex models are penalised automatically through the Bayesian evidence
- **Full posterior inference** — recover well-constrained parameter posteriors alongside model selection
- **Constrained normal priors** — stationarity and invertibility enforced directly in the prior via root constraints on the AR and MA polynomials
- **GPU-ready** — vectorised JAX implementation via BlackJAX's nested sampler with slice sampling
- **Persistence** — grid search results can be saved to and reloaded from file
- **Validated on real astronomy** — sunspots, Kepler light curves, TESS photometry, quasar variability

---

## Repository Structure

```
NestAR/
├── ARIMA.py                  # Core ARIMA recursion (ARIMA_fast, ARIMA_forecast)
├── ARIMA_ns.py               # ARIMA_Nested_Sampler class; loglikelihood and prior_parameters helpers
├── norm_prior.py             # Constrained normal prior: stationarity/invertibility via root-checking
├── model_comparison_utils.py # ARIMA_model_comparison grid search, plot_evidence_heatmap, load_evidence_file
└── README.md
```

---

## Installation

```bash
git clone https://github.com/najinkya1313/ARIMA-Nested-Sampling.git
cd ARIMA-Nested-Sampling
pip install blackjax anesthetic fgivenx jax jaxlib numpy matplotlib scipy tqdm
```

For GPU support, install the appropriate JAX backend following the [JAX installation guide](https://github.com/google/jax#installation).

---

## Quickstart

### Fit a single ARIMA model

```python
from ARIMA_ns import ARIMA_Nested_Sampler

model = ARIMA_Nested_Sampler(
    data       = your_time_series,   # array-like
    order      = (2, 0, 1),          # (p, d, q)
    mu_mean    = 0,                  # prior mean for the series long-term mean
    mu_scale   = 1,                  # prior scale for the series long-term mean
    num_live   = 500,                # number of live points
    num_delete = 50,                 # points removed per NS iteration
    seed       = 42,
)

model.summary()           # print log-evidence, posterior means, corner plot
y_fit = model.get_mean_forecasts()
model.mean_fit_plot(compare=True)
```

### Run a full model comparison grid

```python
from model_comparison_utils import ARIMA_model_comparison, plot_evidence_heatmap

log_posteriors, errors = ARIMA_model_comparison(
    data       = your_time_series,
    max_p      = 5,
    max_q      = 5,
    d          = 0,
    num_live   = 500,
    num_delete = 50,
    seed       = 42,
    mu_mean    = 0,
    mu_scale   = 1,
    file_name  = "results.txt",   # optional: save evidences incrementally
)

fig = plot_evidence_heatmap((log_posteriors, errors), max_order=5)
fig.savefig("evidence_heatmap.pdf")
```

### Reload a saved run

```python
from model_comparison_utils import load_evidence_file, plot_evidence_heatmap

log_posteriors, errors = load_evidence_file("results.txt")
fig = plot_evidence_heatmap((log_posteriors, errors), max_order=5)
```

---

## Method

### ARIMA Models

An ARIMA(p, d, q) model characterises a time series through three integer orders:

| Symbol | Meaning |
|--------|---------|
| **p** | Autoregressive order — dependence on past values |
| **d** | Differencing degree — applied to achieve stationarity |
| **q** | Moving-average order — dependence on past noise terms |

### Prior

NestAR uses a **constrained normal prior**. AR and MA coefficients are drawn from zero-mean normal distributions with a tunable scale (`prior_scale`). Stationarity (AR) and invertibility (MA) are enforced by computing the roots of the characteristic polynomials and rejecting samples whose roots lie inside the unit circle. This is implemented via `jax.lax.cond` inside a vectorised prior sampler, keeping the full pipeline JIT-compilable.

### Nested Sampling

[Nested Sampling](https://arxiv.org/abs/2101.09675) computes the **marginal likelihood (evidence)** Z = ∫ L(θ) π(θ) dθ by iteratively contracting the prior volume. NestAR uses BlackJAX's `nss` (nested slice sampler) with the convergence criterion:

```
logZ_live - logZ < -3
```

For each candidate order (p, q), the evidence Z_i is computed independently. Model comparison is then performed via the **posterior model probability**:

$$P_i \propto Z_i \cdot \pi_i$$

With a uniform prior over model orders, this reduces to a direct comparison of evidences, visualised as a heatmap over the (p, q) grid. Over-parametrised models are automatically disfavoured through the Occam's penalty embedded in the evidence integral.

---
## API Reference

### `ARIMA_Nested_Sampler`

| Argument | Type | Description |
|---|---|---|
| `data` | array | Input time series |
| `order` | tuple | `(p, d, q)` ARIMA order |
| `mu_mean` | float | Prior mean for the series long-term mean |
| `mu_scale` | float | Prior scale for the series long-term mean |
| `num_live` | int | Number of live points |
| `num_delete` | int | Points removed per iteration |
| `seed` | int | Random seed |
| `prior_scale` | float | Scale for AR/MA coefficient priors (default: `1`) |
| `inner_steps_factor` | int | MCMC steps per dimension `k = factor × ndim` (default: `6`) |
| `prior_type` | str | `"normal"` (default) or `"uniform"` |
| `prior_bounds` | dict | Required when `prior_type="uniform"` |

**Key attributes after fitting:**

| Attribute | Description |
|---|---|
| `log_evidence` | Log marginal likelihood Z |
| `log_evidence_err` | Uncertainty on log Z (std over 100 bootstrap resamples) |
| `posterior_samples` | `anesthetic.NestedSamples` object |
| `posterior_means` | List of posterior mean values per parameter |

**Methods:**

| Method | Description |
|---|---|
| `summary()` | Print log Z, posterior means, and render corner plot |
| `get_mean_forecasts()` | Return fitted time series using posterior means |
| `mean_fit_plot(compare=True)` | Plot fit against data |
| `direct_forecast(...)` | Posterior-predictive forecast with `fgivenx` envelope |

### `ARIMA_model_comparison`

Runs `ARIMA_Nested_Sampler` over all (p, q) pairs up to `(max_p, max_q)`, excluding the trivial (0, d, 0) model. Prints live progress and optionally appends results to `file_name` after each model. Returns `(log_posteriors, errors)`.

### `plot_evidence_heatmap`

Plots a colour-mapped grid of log posterior probabilities using the `inferno` colormap. The best model is circled in cyan. Accepts `contrast` (clip the lower end of the colour scale), `annotate` (overlay values), `invert` (flip colormap for AIC/BIC), and custom figure dimensions via `fig_width`/`fig_height` kwargs.

### `load_evidence_file`

Reads a `file_name` produced by `ARIMA_model_comparison` and returns `(log_posteriors, errors)`, with optional renormalisation.

---

## Citation

If you use NestAR in your research, please cite the paper:

```bibtex
@article{naik2025nestar,
  title   = {Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis},
  author  = {Naik, Ajinkya and Handley, Will},
  journal = {arXiv preprint arXiv:2512.01929},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.01929}
}
```

---

## Acknowledgements

This work was carried out at the **Institute of Astronomy, University of Cambridge**. The nested sampler is built on [BlackJAX](https://github.com/blackjax-devs/blackjax). Posterior analysis uses [anesthetic](https://github.com/handley-lab/anesthetic) and posterior predictive plots use [fgivenx](https://github.com/handley-lab/fgivenx).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
