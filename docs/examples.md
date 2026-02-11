# Examples

Practical examples for using Identifiability Guard.

## Basic Workflow

### End-to-End Example

```python
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.metrics import MCC, DCI

# 1. Generate ground-truth factors
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)

# 2. Apply encoder
encoder = E1ElementwiseLinear(d=5, seed=42)
Z_hat = encoder.encode(Z)

# 3. Evaluate identifiability
mcc = MCC()
dci = DCI()

mcc_result = mcc.compute(Z, Z_hat)
dci_result = dci.compute(Z, Z_hat)

print(f"MCC: {mcc_result.primary_score:.3f}")
print(f"DCI-D: {dci_result.subscores['disentanglement']:.3f}")
print(f"DCI-C: {dci_result.subscores['completeness']:.3f}")
```

## Comparing DGPs

Evaluate how metrics behave across different data generating processes.

```python
from identifiability_guard.dgp import D1Independent, D2Correlated, D3SingleRedundant
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.metrics import MetricRegistry

# Create encoder
encoder = E1ElementwiseLinear(d=5, seed=42)

# Create DGPs
dgps = {
    "D1-Independent": D1Independent(d=5, seed=42),
    "D2-Correlated": D2Correlated(d=5, correlation=0.5, seed=42),
    "D3-Redundant": D3SingleRedundant(d=5, r=1, seed=42),
}

# Setup metrics
registry = MetricRegistry()
registry.register_defaults()

# Evaluate each DGP
for dgp_name, dgp in dgps.items():
    Z = dgp.sample(1000)
    Z_hat = encoder.encode(Z)
    results = registry.compute_all(Z, Z_hat)

    print(f"\n{dgp_name}:")
    print(f"  MCC: {results['mcc'].primary_score:.3f}")
    print(f"  DCI-D: {results['dci'].subscores['disentanglement']:.3f}")
    print(f"  R²: {results['r2'].primary_score:.3f}")
```

## Comparing Encoders

Test different encoder types on the same data.

```python
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
)
from identifiability_guard.metrics import MCC

# Generate data
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)

# Create encoders
encoders = {
    "E1-Linear": E1ElementwiseLinear(d=5, seed=42),
    "E2-Nonlinear": E2ElementwiseNonlinear(d=5, nonlinearity_strength=0.5, seed=42),
    "E3-Entangled": E3LinearlyEntangled(d=5, seed=42),
}

# Evaluate each encoder
mcc = MCC()
for encoder_name, encoder in encoders.items():
    Z_hat = encoder.encode(Z)
    result = mcc.compute(Z, Z_hat)
    print(f"{encoder_name}: MCC = {result.primary_score:.3f}")
```

## Parameter Sweeps

### Sample Size Sensitivity

```python
import numpy as np
import matplotlib.pyplot as plt
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.metrics import MCC, DCI

# Setup
dgp = D1Independent(d=5, seed=42)
encoder = E1ElementwiseLinear(d=5, seed=42)
mcc = MCC()
dci = DCI()

# Sweep over sample sizes
sample_sizes = [100, 500, 1000, 5000, 10000]
mcc_scores = []
dci_scores = []

for n in sample_sizes:
    Z = dgp.sample(n)
    Z_hat = encoder.encode(Z)

    mcc_scores.append(mcc.compute(Z, Z_hat).primary_score)
    dci_scores.append(dci.compute(Z, Z_hat).subscores['disentanglement'])

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, mcc_scores, marker='o', label='MCC')
plt.plot(sample_sizes, dci_scores, marker='s', label='DCI-D')
plt.xlabel('Sample Size')
plt.ylabel('Metric Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sample_size_sensitivity.png', dpi=150)
```

### Correlation Strength Sweep

```python
from identifiability_guard.dgp import D2Correlated
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.metrics import MCC

encoder = E1ElementwiseLinear(d=5, seed=42)
mcc = MCC()

correlations = np.linspace(0.0, 0.9, 10)
scores = []

for corr in correlations:
    dgp = D2Correlated(d=5, correlation=corr, seed=42)
    Z = dgp.sample(1000)
    Z_hat = encoder.encode(Z)
    scores.append(mcc.compute(Z, Z_hat).primary_score)

plt.figure(figsize=(8, 5))
plt.plot(correlations, scores, marker='o')
plt.xlabel('Correlation Strength')
plt.ylabel('MCC Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_sweep.png', dpi=150)
```

## Multi-Seed Evaluation

Robust evaluation with confidence intervals.

```python
from identifiability_guard.evaluation import run_multi_seed_evaluation
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E3LinearlyEntangled
from identifiability_guard.metrics import MCC, DCI

def evaluate_once(seed):
    dgp = D1Independent(d=5, seed=seed)
    encoder = E3LinearlyEntangled(d=5, seed=seed + 1000)

    Z = dgp.sample(1000)
    Z_hat = encoder.encode(Z)

    mcc = MCC()
    dci = DCI()

    return {
        "mcc": mcc.compute(Z, Z_hat).primary_score,
        "dci_d": dci.compute(Z, Z_hat).subscores['disentanglement'],
    }

# Run with multiple seeds
raw_results, stats = run_multi_seed_evaluation(
    evaluate_once,
    n_seeds=10,
    base_seed=42,
)

# Print results with confidence intervals
print(f"MCC: {stats['mcc']['mean']:.3f} ± {stats['mcc']['std']:.3f}")
print(f"  95% CI: [{stats['mcc']['ci_lower']:.3f}, {stats['mcc']['ci_upper']:.3f}]")
print(f"DCI-D: {stats['dci_d']['mean']:.3f} ± {stats['dci_d']['std']:.3f}")
print(f"  95% CI: [{stats['dci_d']['ci_lower']:.3f}, {stats['dci_d']['ci_upper']:.3f}]")
```

## Timing and Profiling

Analyze computational costs.

```python
from identifiability_guard.evaluation import time_block, profile_block
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E3LinearlyEntangled
from identifiability_guard.metrics import MetricRegistry

dgp = D1Independent(d=10, seed=42)
encoder = E3LinearlyEntangled(d=10, seed=42)
registry = MetricRegistry()
registry.register_defaults()

# Time individual operations
with time_block("Data generation"):
    Z = dgp.sample(10000)

with time_block("Encoding"):
    Z_hat = encoder.encode(Z)

# Profile complete evaluation
with profile_block("Metric computation") as profile:
    results = registry.compute_all(Z, Z_hat)

print(f"Time: {profile['elapsed']:.2f}s")
print(f"Peak Memory: {profile['peak_mb']:.2f} MB")
```

## Heatmap Visualization

Generate comprehensive DGP × Encoder heatmaps.

```python
import numpy as np
import matplotlib.pyplot as plt
from identifiability_guard.dgp import D1Independent, D2Correlated
from identifiability_guard.encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
)
from identifiability_guard.metrics import MCC

# Define experimental grid
dgps = {
    "D1": D1Independent(d=5, seed=42),
    "D2": D2Correlated(d=5, correlation=0.5, seed=42),
}

encoders = {
    "E1": E1ElementwiseLinear(d=5, seed=42),
    "E2": E2ElementwiseNonlinear(d=5, nonlinearity_strength=0.5, seed=42),
    "E3": E3LinearlyEntangled(d=5, seed=42),
}

# Compute scores
mcc = MCC()
scores = np.zeros((len(dgps), len(encoders)))

for i, (dgp_name, dgp) in enumerate(dgps.items()):
    Z = dgp.sample(1000)
    for j, (enc_name, encoder) in enumerate(encoders.items()):
        Z_hat = encoder.encode(Z)
        scores[i, j] = mcc.compute(Z, Z_hat).primary_score

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(scores, cmap='RdYlGn', vmin=0, vmax=1)

ax.set_xticks(range(len(encoders)))
ax.set_yticks(range(len(dgps)))
ax.set_xticklabels(encoders.keys())
ax.set_yticklabels(dgps.keys())

# Add values to cells
for i in range(len(dgps)):
    for j in range(len(encoders)):
        ax.text(j, i, f'{scores[i, j]:.2f}',
                ha='center', va='center', color='black')

ax.set_xlabel('Encoder')
ax.set_ylabel('DGP')
ax.set_title('MCC Scores')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('dgp_encoder_heatmap.png', dpi=150)
```

## Running Example Scripts

The repository includes pre-built example scripts:

### Combined Heatmap

```bash
# Basic usage
python examples/evaluate_all_combinations_combined.py

# Custom configuration
python examples/evaluate_all_combinations_combined.py \
    --samples 10000 \
    --factors 6 \
    --seed 123 \
    --output results/heatmap.png
```

### Sensitivity Analysis

```bash
# Sample size sweep
python examples/evaluate_sensitivity.py \
    --sweep-samples 500,1000,5000,10000 \
    --dgp D1 --encoder E1 \
    --n-seeds 10

# Correlation sweep
python examples/evaluate_sensitivity.py \
    --sweep-correlation 0.0,0.3,0.6,0.9 \
    --dgp D2 --encoder E2 \
    --n-seeds 5

# All metrics
python examples/evaluate_sensitivity.py \
    --sweep-samples 1000,5000 \
    --all-metrics \
    --n-seeds 5
```

## Next Steps

- See [API Reference](api/index.md) for detailed class documentation
- Read [Metrics Guide](metrics.md) for metric details
- Check [Contributing](contributing.md) for adding new components
