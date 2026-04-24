# 🌀 NestAR — Nested Sampling Autoregressive Modelling

> **Bayesian ARIMA model selection for astronomical time-series, powered by Nested Sampling.**

[![arXiv](https://img.shields.io/badge/arXiv-2512.01929-b31b1b.svg)](https://arxiv.org/abs/2512.01929)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![BlackJAX](https://img.shields.io/badge/sampler-BlackJAX-purple.svg)](https://github.com/blackjax-devs/blackjax)

---

## Overview

NestAR combines **ARIMA models** with the **Nested Sampling algorithm** to tackle one of the core challenges in time-series analysis: *how do you pick the right model without overfitting?*

Classical approaches like AIC/BIC rely on maximum likelihood optimisation, which can be biased — especially with complex likelihood landscapes. NestAR takes a fully Bayesian route: it computes **log Bayesian evidences** for every candidate ARIMA(p, d, q) order on a grid, letting the evidence ratio (Bayes factor) drive model selection. This comes with a built-in **Occam's penalty** that naturally disfavours unnecessarily complex models — no manual regularisation needed.

The framework is vectorised and supports **GPU acceleration** via JAX, making it practical to scan large grids of (p, q) orders in a single run.

The method is described in full in:

> **Naik, A. & Handley, W.** (2025). *Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis.* arXiv:[2512.01929](https://arxiv.org/abs/2512.01929)

---

## Key Features

- **Bayesian model selection** — compute log-evidences and posterior probabilities across a full (p, q) grid
- **Intrinsic Occam's razor** — complex models are penalised automatically through the evidence
- **Full posterior inference** — recover well-constrained parameter posteriors alongside model selection
- **Vectorised & GPU-ready** — built on [BlackJAX](https://github.com/blackjax-devs/blackjax) with JAX acceleration
- **Validated on real astronomy data** — sunspots, Kepler light curves, TESS photometry, quasar variability

---

## Repository Structure

```
NestAR/
├── ARIMA.py                  # Core ARIMA model: likelihood, recursion, stationarity checks
├── ARIMA_ns.py               # Nested Sampling loop over ARIMA(p,d,q) grids via BlackJAX
├── norm_prior.py             # Prior definitions for AR/MA coefficients and noise parameters
├── model_comparison_utils.py # Evidence computation, Bayes factors, posterior probability heatmaps
└── README.md
```

---

## Method

### ARIMA Models

An ARIMA(p, d, q) model characterises a time series through:
- **p** — autoregressive order (dependence on past values)
- **d** — degree of differencing (to achieve stationarity)
- **q** — moving-average order (dependence on past noise terms)

### Nested Sampling

[Nested Sampling](https://arxiv.org/abs/2101.09675) is a Bayesian inference algorithm that computes the **marginal likelihood (evidence)** Z = ∫ L(θ) π(θ) dθ by iteratively contracting the prior volume. NestAR uses BlackJAX's implementation with **slice sampling** as the inner MCMC kernel.

For each candidate model order (p, q), the algorithm:
1. Draws live points from the prior
2. Iteratively replaces the lowest-likelihood point, shrinking the prior volume
3. Accumulates the evidence integral until convergence
4. Returns the log-evidence and full posterior samples

Model comparison is then performed via the **posterior model probability**:

$$P_i \propto Z_i \cdot \pi_i$$

where a uniform prior over model orders (π_i = const) reduces this to a direct comparison of evidences.

---

## Astronomical Applications

NestAR has been validated on both simulated and real astronomical datasets:

| Dataset | Instrument | Description |
|---|---|---|
| Sunspot number record | — | Historical yearly sunspot counts |
| KIC 12008916 | Kepler | Red giant stellar light curve |
| Kepler 17 | Kepler | Exoplanet host star photometry |
| 3C 273 | TESS | Quasar optical variability |
| S4 0954+65 | TESS | Blazar variability |

In all cases, NestAR correctly recovered the best-fitting model order and returned well-constrained parameter posteriors.

---

## Installation

```bash
git clone https://github.com/najinkya1313/ARIMA-Nested-Sampling.git
cd ARIMA-Nested-Sampling
pip install blackjax jax jaxlib numpy matplotlib
```

For GPU support, install the appropriate JAX backend following the [JAX installation guide](https://github.com/google/jax#installation).

---

## Quickstart

```python
from ARIMA_ns import run_nested_sampling
from model_comparison_utils import compute_model_posteriors, plot_evidence_heatmap

# Run nested sampling over a grid of ARIMA orders
results = run_nested_sampling(
    time_series=your_data,
    p_max=5,
    q_max=5,
    d=0,
)

# Compare models and plot the evidence heatmap
posteriors = compute_model_posteriors(results)
plot_evidence_heatmap(posteriors)
```

---

## Citation

If you use NestAR in your research, please cite:

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

This work was carried out at the **Institute of Astronomy, University of Cambridge**. The nested sampler is built on [BlackJAX](https://github.com/blackjax-devs/blackjax). We thank the developers of JAX for enabling efficient, hardware-accelerated Bayesian computation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.




