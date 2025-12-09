# Identifiability Guard

A framework for evaluating identifiability metrics in representation learning.

## Overview

This project provides a modular codebase for studying identifiability metrics in representation learning. It implements:

1. **Data Generating Processes (DGPs)**: Generate ground-truth latent factors with different statistical properties
2. **Encoder Mixings**: Transform latent factors to learned representations in various ways
3. **Identifiability Metrics**: Evaluate how well learned representations recover ground-truth factors

## Installation

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Data Generating Processes (DGPs)

| DGP | Name | Description |
|-----|------|-------------|
| D1 | Independent | Mutually independent, non-redundant factors |
| D2 | Correlated | Statistically dependent but non-redundant factors |
| D3 | Single-redundant | One factor is a function of another single factor |
| D4 | Multi-redundant | One factor is a function of multiple factors |

### Example Usage

```python
from src.dgp import D1Independent, D2Correlated

# Generate independent factors
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)  # Shape: (1000, 5)

# Generate correlated factors
dgp_corr = D2Correlated(d=5, correlation=0.5, seed=42)
Z_corr = dgp_corr.sample(1000)
```

## Encoder Mixings

| Encoder | Name | Description | Status |
|---------|------|-------------|--------|
| E1 | Elementwise Linear | Ẑ_j = a_j · Z_π(j), m = d | ✅ |
| E2 | Elementwise Nonlinear | Ẑ_j = h_j(Z_π(j)), m = d | ✅ |
| E3 | Linearly Entangled | Ẑ = A · Z (dense mixing), m = d | ✅ |
| E4 | Undercomplete Linear | Elementwise scaling, m < d | ✅ |
| E5 | Overcomplete Linear | Redundant scaled copies, m > d | ✅ |
| E6 | Overcomplete Multicodes | Multiple nonlinear codes per factor, m > d | 🚧 TODO |

### Example Usage

```python
from src.dgp import D1Independent
from src.encoders import E1ElementwiseLinear, E3LinearlyEntangled

# Generate data
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)

# Apply elementwise linear encoding
encoder = E1ElementwiseLinear(d=5, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (1000, 5)

# Apply linearly entangled encoding
encoder_ent = E3LinearlyEntangled(d=5, seed=42)
Z_hat_ent = encoder_ent.encode(Z)
```

## Identifiability Metrics

### MCC (Mean Correlation Coefficient) — 🚧 TODO

$$\text{MCC}(\rho) = \frac{1}{k} \max_{\pi \in S_k} \sum_{i=1}^k |\text{Corr}(Z_i, \hat{Z}_{\pi(i)})|$$

### DCI (Disentanglement, Completeness, Informativeness) — 🚧 TODO

- **Disentanglement**: Each code depends on at most one factor
- **Completeness**: Each factor is captured by at most one code
- **Informativeness**: How well codes predict factors

### Example Usage

```python
from src.metrics import MCC, DCI

# Compute MCC
mcc = MCC()
score = mcc.compute(Z, Z_hat)  # Returns float in [0, 1]

# Compute DCI
dci = DCI()
scores = dci.compute(Z, Z_hat)  # Returns dict with D, C, I scores
print(f"Disentanglement: {scores['disentanglement']:.3f}")
print(f"Completeness: {scores['completeness']:.3f}")
print(f"Informativeness: {scores['informativeness']:.3f}")
```

## Project Structure

```
identifiability-guard/
├── src/
│   ├── __init__.py
│   ├── dgp/                    # Data Generating Processes
│   │   ├── __init__.py
│   │   ├── base.py            # BaseDGP abstract class
│   │   ├── d1_independent.py  # D1: Independent factors
│   │   ├── d2_correlated.py   # D2: Correlated factors
│   │   ├── d3_single_redundant.py  # D3: Single-factor redundant
│   │   └── d4_multi_redundant.py   # D4: Multi-factor redundant
│   ├── encoders/               # Encoder Mixings
│   │   ├── __init__.py
│   │   ├── base.py            # BaseEncoder abstract class
│   │   ├── e1_elementwise_linear.py
│   │   ├── e2_elementwise_nonlinear.py
│   │   ├── e3_linearly_entangled.py
│   │   ├── e4_undercomplete_linear.py
│   │   ├── e5_overcomplete_linear.py
│   │   └── e6_overcomplete_multicodes.py
│   └── metrics/                # Identifiability Metrics
│       ├── __init__.py
│       ├── base.py            # BaseMetric abstract class
│       ├── mcc.py             # MCC metric
│       └── dci.py             # DCI metric
├── tests/
│   ├── __init__.py
│   ├── test_dgp.py
│   ├── test_encoders.py
│   └── test_metrics.py
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
pytest tests/
```

## License

MIT
