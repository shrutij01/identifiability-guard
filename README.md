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
| E2 | Elementwise Nonlinear | Ẑ_j = h_j(Z_π(j)), m = d (with strength control) | ✅ |
| E3 | Linearly Entangled | Ẑ = A · Z (dense mixing), m = d | ✅ |
| E4 | Undercomplete Linear | Elementwise scaling, m < d | ✅ |
| E5 | Overcomplete Linear | Redundant scaled copies, m > d | ✅ |
| E6 | Overcomplete Multicodes | Multiple nonlinear codes per factor, m > d | ✅ |
| E7 | Overcomplete Linearly Entangled | Ẑ = A · Z (dense, rank-d), m > d | ✅ |
| E8 | Overcomplete Nonlinear Disjoint | Disjoint sin/cos codes per factor, m > d | ✅ |
| E9 | Random Gaussian | Random noise (baseline/sanity check), m = d | ✅ |

### Example Usage

```python
from src.dgp import D1Independent, D2Correlated, D4MultiRedundant
from src.encoders import (
    E1ElementwiseLinear, 
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
    E7OvercompleteEntangled,
    E8OvercompleteDisjoint,
)

# Generate data
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)

# Apply elementwise linear encoding
encoder = E1ElementwiseLinear(d=5, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (1000, 5)

# Apply linearly entangled encoding
encoder_ent = E3LinearlyEntangled(d=5, seed=42)
Z_hat_ent = encoder_ent.encode(Z)

# E7 Overcomplete linearly entangled (m > d, dense mixing)
encoder_e7 = E7OvercompleteEntangled(d=5, m=10, condition_number=10.0, seed=42)
Z_hat_e7 = encoder_e7.encode(Z)  # Shape: (1000, 10)

# E8 Overcomplete disjoint (sin/cos encoding per factor)
encoder_e8 = E8OvercompleteDisjoint(d=5, codes_per_factor=2, seed=42)
Z_hat_e8 = encoder_e8.encode(Z)  # Shape: (1000, 10)

# E9 Random Gaussian (baseline/sanity check - should have ~0 metrics)
encoder_e9 = E9RandomGaussian(d=5, seed=42)
Z_hat_e9 = encoder_e9.encode(Z)  # Shape: (1000, 5) - random noise

# Parameterized nonlinearity strength (0=linear, 1=fully nonlinear)
encoder_e2 = E2ElementwiseNonlinear(d=5, nonlinearity_strength=0.5, seed=42)
Z_hat_e2 = encoder_e2.encode(Z)

# Parameterized redundancy strength (noise in redundant factors)
dgp_d4 = D4MultiRedundant(d=5, r=1, noise_std=0.1, seed=42)
Z_d4 = dgp_d4.sample(1000)
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
│   │   ├── e6_overcomplete_multicodes.py
│   │   ├── e7_overcomplete_entangled.py   # NEW: E7
│   │   ├── e8_overcomplete_disjoint.py    # NEW: E8
│   │   └── e9_random_gaussian.py          # NEW: E9 (baseline)
│   ├── evaluation/              # NEW: Evaluation Utilities
│   │   ├── __init__.py
│   │   ├── timing.py           # Timing and memory profiling
│   │   ├── multi_seed.py       # Multi-seed evaluation with statistics
│   │   └── sensitivity.py      # Parameter sweep and sensitivity analysis
│   └── metrics/                # Identifiability Metrics
│       ├── __init__.py
│       ├── base.py            # BaseMetric abstract class
│       ├── mcc.py             # MCC metric
│       └── dci.py             # DCI metric
├── tests/
│   ├── __init__.py
│   ├── test_dgp.py
│   ├── test_encoders.py
│   ├── test_evaluation.py      # NEW: Tests for evaluation utilities
│   └── test_metrics.py
├── examples/
│   ├── evaluate_all_combinations_combined.py
│   └── evaluate_sensitivity.py  # NEW: Sensitivity analysis script
├── pyproject.toml
└── README.md
```

## Evaluation Utilities

The framework includes comprehensive evaluation utilities for analyzing identifiability metrics under different conditions.

### Timing and Memory Profiling

```python
from src.evaluation import time_block, memory_profiler, profile_block

# Time a code block
with time_block("Data generation"):
    Z = dgp.sample(10000)

# Profile memory usage
with memory_profiler("Model training"):
    model.fit(X, y)

# Combined profiling
with profile_block("Full evaluation") as profile:
    results = evaluate_model(data)
print(f"Time: {profile['elapsed']:.2f}s, Peak Memory: {profile['peak_mb']:.2f} MB")
```

### Multi-Seed Evaluation

```python
from src.evaluation import run_multi_seed_evaluation

def eval_fn(seed):
    dgp = D1Independent(d=5, seed=seed)
    encoder = E1ElementwiseLinear(d=5, seed=seed)
    Z = dgp.sample(1000)
    Z_hat = encoder.encode(Z)
    return {"mcc": compute_mcc(Z, Z_hat)}

# Run with multiple seeds and get statistics
raw_results, aggregated = run_multi_seed_evaluation(
    eval_fn,
    n_seeds=10,
    base_seed=42,
)

# Access mean, std, confidence intervals
print(f"MCC: {aggregated['mcc']['mean']:.3f} ± {aggregated['mcc']['std']:.3f}")
print(f"95% CI: [{aggregated['mcc']['ci_lower']:.3f}, {aggregated['mcc']['ci_upper']:.3f}]")
```

### Sensitivity Analysis

```bash
# Sweep over sample sizes
python examples/evaluate_sensitivity.py \
    --sweep-samples 1000,5000,10000 \
    --dgp D1 --encoder E1 \
    --n-seeds 5

# Sweep over correlation values (D2)
python examples/evaluate_sensitivity.py \
    --sweep-correlation 0.0,0.3,0.5,0.7,0.9 \
    --encoder E2 \
    --n-seeds 10

# Sweep over nonlinearity strength (E2)
python examples/evaluate_sensitivity.py \
    --sweep-nonlinearity 0.0,0.25,0.5,0.75,1.0 \
    --dgp D1 \
    --n-seeds 5

# Results are saved as JSON and camera-ready plots
```

## Example Scripts

### Combined Heatmap Visualization

Generate a comprehensive heatmap showing all DGP × Encoder combinations:

```bash
# Basic usage (default: 5000 samples, 4 factors)
python examples/evaluate_all_combinations_combined.py

# Custom configuration
python examples/evaluate_all_combinations_combined.py \
    --samples 10000 \
    --factors 6 \
    --seed 123 \
    --output results/my_heatmap.png
```

This produces:
- A multi-panel figure with one heatmap per DGP
- Each cell shows metric scores (0-100 scale) for encoder × metric combinations  
- A timing/memory profiling table at the bottom
- Title includes samples and factors configuration

### Sensitivity Analysis Sweep

Run comprehensive sensitivity analysis with statistical aggregation:

```bash
# Sample size sweep (all 7 metrics plotted)
python examples/evaluate_sensitivity.py \
    --sweep-samples 500,1000,2500,5000,10000 \
    --dgp D1 --encoder E1 \
    --n-seeds 10

# Correlation sweep for D2
python examples/evaluate_sensitivity.py \
    --sweep-correlation 0.0,0.2,0.4,0.6,0.8,1.0 \
    --dgp D2 --encoder E2 \
    --n-seeds 5

# Select specific metrics to compute (faster)
python examples/evaluate_sensitivity.py \
    --sweep-samples 1000,5000,10000 \
    --metrics dci_disentanglement,mcc_pearson,r2 \
    --n-seeds 5

# Run all metrics
python examples/evaluate_sensitivity.py \
    --sweep-samples 1000,5000,10000 \
    --all-metrics \
    --n-seeds 5
```

Output includes:
- JSON files with raw results and statistics
- Camera-ready sensitivity plots with 95% CI error bands
- Support for 7 metrics: DCI (3 subscores), MCC (3 variants), R²

## Running Tests

```bash
pytest tests/
```

## Running Experiments

Each experiment is a self-contained script in the `experiments/` directory.
Run them from the project root:

```bash
# Run all experiments sequentially
for f in experiments/exp*.py; do python "$f"; done

# Experiment 1 – Invariance across DGP types
python experiments/exp01_invariance_across_dgps.py

# Experiment 2 – Nonlinearity sensitivity (+ encoder NL sweep via sensitivity.py)
python experiments/exp02_nonlinearity_sensitivity.py

# Experiment 3 – Correlation sign effect (+ correlation sweeps via sensitivity.py)
python experiments/exp03_correlation_sign.py

# Experiment 4 – 2D heatmap: correlation × nonlinearity
python experiments/exp04_correlation_vs_entanglement.py

# Experiment 5 – Predictability vs disentanglement (+ NL & correlation sweeps)
python experiments/exp05_predictability_vs_disentanglement.py

# Experiment 6 – Dropped variables & dimension inflation (+ factor sweeps)
python experiments/exp06_dropped_variables.py

# Experiment 9 – Overcomplete representations
python experiments/exp09_overcomplete_representations.py

# Experiment 10 – Sample sensitivity grid (+ sample sweep D2×E3)
python experiments/exp10_sample_sensitivity.py

# Experiment 11 – Null-encoder inflation (+ sample sweep D1×E10)
python experiments/exp11_metric_inflation.py
```

Results (PDF + PNG plots) are saved to `results/experiments/<expNN>/`.

### Combined Heatmap (big_table.py)

Generate a comprehensive heatmap showing all DGP × Encoder × Metric combinations:

```bash
# Default settings (300 samples, 5 factors)
python experiments/big_table.py

# Custom configuration
python experiments/big_table.py --samples 10000 --factors 6 --seed 123

# Specify output path
python experiments/big_table.py --output results/my_heatmap.png
```

Output: a multi-panel figure with one heatmap per DGP, encoder × metric cells
annotated with scores, and a timing/memory profiling table.

### Sensitivity Analysis (sensitivity.py)

Run parameter sweeps with multi-seed aggregation and camera-ready plots:

```bash
# Sweep sample sizes
python experiments/sensitivity.py \
    --sweep-samples 500,1000,2500,5000,10000 \
    --dgp D1 --encoder E1 --n-seeds 5

# Sweep correlation values (always uses D2)
python experiments/sensitivity.py \
    --sweep-correlation 0.0,0.3,0.5,0.7,0.9 \
    --encoder E2 --n-seeds 5

# Sweep number of factors
python experiments/sensitivity.py \
    --sweep-factors 3,5,7,10 \
    --dgp D2 --encoder E4 --n-seeds 5

# Sweep nonlinearity strength (E2 encoder)
python experiments/sensitivity.py \
    --sweep-nonlinearity 0.0,0.25,0.5,0.75,1.0 \
    --dgp D1 --n-seeds 5

# Sweep encoder nonlinearity strength (E2, explicit label)
python experiments/sensitivity.py \
    --sweep-encoder-nonlinearity 0.0,0.25,0.5,0.75,1.0 \
    --dgp D1 --n-seeds 5

# Compute all metrics (default uses a fast subset)
python experiments/sensitivity.py \
    --sweep-samples 500,1000,5000 \
    --all-metrics --n-seeds 5

# Select specific metrics
python experiments/sensitivity.py \
    --sweep-samples 1000,5000 \
    --metrics dci_disentanglement,mcc_pearson,r2

# Custom output directory
python experiments/sensitivity.py \
    --sweep-samples 1000,5000 \
    --output-dir results/my_sweep
```

Each sweep produces:
- A JSON file with raw per-seed results and aggregated statistics
- A camera-ready PNG plot with 95% CI error bands (one panel per metric)

## License

MIT
