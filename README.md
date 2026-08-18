# Identifiability Guard

Benchmark framework for stress-testing identifiability metrics. Generate controlled ground-truth factors, transform them in specific ways, and measure whether standard metrics correctly detect recovery.

Companion code for [*Who Guards the Guardians? The Challenges of Evaluating Identifiability of Learned Representations*](https://proceedings.mlr.press/v337/joshi26a.html).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## What This Does

Three pieces: generate latent factors Z with known structure → apply systematic transformations → measure recovery with standard metrics.

1. **DGPs** — Generate Z: independent, correlated, or redundant factors
2. **Encoders** — Transform Z → Ẑ: linear, nonlinear, overcomplete, entangled
3. **Metrics** — Score recovery: MCC, DCI, R², MIG, T-MEX, InfoMEC

## Install

```bash
# With uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e .

# Standard path
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Use It

```python
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.metrics import MCC, DCI

# Step 1: Generate 5 independent factors, 1000 samples
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)

# Step 2: Apply diagonal scaling + permutation
encoder = E1ElementwiseLinear(d=5, seed=42)
Z_hat = encoder.encode(Z)

# Step 3: Measure recovery
mcc = MCC()
dci = DCI()

print(f"MCC: {mcc.compute(Z, Z_hat).primary_score:.3f}")
print(f"DCI: {dci.compute(Z, Z_hat).subscores['disentanglement']:.3f}")
```

## Docs

📚 **[Full docs](https://shrutij01.github.io/identifiability-guard/)** | **[GitHub docs](docs/index.md)**

Quick links:
- [Installation](docs/installation.md) — get running
- [DGPs](docs/dgp.md) — generate factors
- [Encoders](docs/encoders.md) — transform factors
- [Metrics](docs/metrics.md) — score recovery
- [Examples](docs/examples.md) — code walkthroughs
- [Contributing](docs/contributing.md) — add components

## What's Included

**DGPs** — 4 factor structures

| Code | What it does |
|------|--------------|
| D1 | Independent Gaussians |
| D2 | Correlated factors |
| D3 | One factor copies another |
| D4 | One factor depends on multiple |

**Encoders** — 10 transformation types

| Code | Transform | Dims |
|------|-----------|------|
| E1 | Diagonal scaling + permutation | m = d |
| E2 | Elementwise nonlinear | m = d |
| E3 | Dense linear mixing | m = d |
| E4 | Dimensionality reduction | m < d |
| E5 | Overcomplete linear | m > d |
| E6 | Multiple codes per factor | m > d |
| E7 | Overcomplete + entangled | m > d |
| E8 | Disjoint sin/cos codes | m > d |
| E9/E10 | Random noise baselines | m = d |

**Metrics** — 6 recovery tests

| Code | What it measures |
|------|------------------|
| MCC | Correlation-based matching |
| DCI | Disentanglement + completeness + informativeness |
| R² | Linear prediction quality |
| MIG | Mutual information gaps |
| T-MEX | Statistical exchangeability |
| InfoMEC | Info-theoretic decomposition |

## Run Experiments

```bash
# Full DGP × Encoder grid
python examples/evaluate_all_combinations_combined.py

# Sample size sweep
python examples/evaluate_sensitivity.py \
    --sweep-samples 500,1000,5000,10000 \
    --dgp D1 --encoder E1 --n-seeds 10
```

## Dev Setup

```bash
uv pip install -e ".[dev]"  # Install dev tools
pytest                       # Run tests
black src/ tests/           # Format
mypy src/                   # Type check
```

## Structure

```
src/identifiability_guard/
├── dgp/         # Factor generators
├── encoders/    # Transformations
├── metrics/     # Recovery scores
└── evaluation/  # Utilities

tests/           # Unit tests
examples/        # Runnable experiments
docs/            # Documentation
```

## Citation

```bibtex
@InProceedings{pmlr-v337-joshi26a,
  title = {Who Guards the Guardians? {The} Challenges of Evaluating Identifiability of Learned Representations},
  author = {Joshi, Shruti and Saulus, Th\'{e}o and Brendel, Wieland and Brouillard, Philippe and Sridhar, Dhanya and Reizinger, Patrik},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence},
  pages = {2618--2660},
  year = {2026},
  editor = {Perković, Emilija and Malinsky, Daniel},
  volume = {337},
  series = {Proceedings of Machine Learning Research},
  month = {17--21 Aug},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v337/main/assets/joshi26a/joshi26a.pdf},
  url = {https://proceedings.mlr.press/v337/joshi26a.html},
  abstract = {Identifiability in representation learning is commonly evaluated using standard metrics (e.g., *MCC, $R^2$, DCI*) on synthetic benchmarks with known ground-truth factors. These metrics are assumed to reflect recovery up to the equivalence class guaranteed by identifiability theory. We show that this assumption holds only under specific structural conditions: each metric implicitly encodes assumptions about both the data-generating process ({DGP}) and the encoder. When these assumptions are violated, metrics become misspecified and can produce systematic false positives and false negatives. Such failures occur both within classical identifiability regimes and in post-hoc settings where identifiability is most needed. We introduce a taxonomy separating {DGP} assumptions from encoder geometry, use it to characterise the validity domains of existing metrics, and release an evaluation suite for reproducible stress testing and comparison.}
}
```

The same paper citation is available in machine-readable form in
[`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
