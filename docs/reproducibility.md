# Reproducibility Guide

## Overview

Reproducibility means that running the same code with the same data produces identical results every time. For scientific research, this is not optional—it's foundational. When metrics produce different scores across runs, we cannot distinguish genuine differences in model performance from random noise.

This guide explains how to ensure reproducible metric evaluations in `identifiability-guard`. We cover the principles, the implementation details, and the specific controls you need.

---

## Why Reproducibility Matters

Consider evaluating a representation learning model with MCC (Mean Correlation Coefficient). You run it twice:

- **Run 1**: MCC = 0.847
- **Run 2**: MCC = 0.853

Is the 0.006 difference meaningful? Without reproducibility guarantees, you cannot know. The variation might come from:

1. Random train/test splits
2. Random initialization in regression solvers
3. Nondeterministic linear algebra operations
4. Random tie-breaking in optimization algorithms

Reproducibility eliminates these sources of noise. When you control randomness properly, identical inputs produce identical outputs—always.

---

## Core Principles

### 1. Explicit Random State Management

Every source of randomness must be controlled by a seed. This includes:

- NumPy's random number generator
- scikit-learn's random state parameters
- Data shuffling and splitting operations
- Any sampling or stochastic optimization

**Never** rely on implicit global random state. Always pass explicit `random_state` parameters.

### 2. Deterministic Operations

Where possible, use deterministic algorithms:

- Matrix decompositions with fixed algorithms (not randomized)
- Stable sorting (important when values are equal)
- Deterministic linear algebra backends

When using stochastic algorithms (like stochastic gradient descent), control their random state explicitly.

### 3. Data Ordering Independence

Metrics should not depend on the order of input samples. If they do, document this clearly and ensure consistent ordering.

Most metrics in this library are order-independent by design (they compute statistics over all samples). The exception is metrics that use sequential train/test splits—these require deterministic splitting.

---

## Implementation in identifiability-guard

### Random State Architecture

All metrics that use randomness accept a `random_state` parameter:

```python
from identifiability_guard.metrics import DCIMetric, MCCMetric

# Seed for reproducibility
seed = 42

# Metrics accept random_state
dci = DCIMetric(random_state=seed)
mcc = MCCMetric(random_state=seed)

# Results will be identical across runs
result1 = dci.compute(Z, Z_hat)
result2 = dci.compute(Z, Z_hat)
assert result1.primary_score == result2.primary_score  # Always passes
```

This parameter is **not optional** for metrics that use randomness. The library enforces this by requiring it in the constructor.

### What Gets Seeded?

Different metrics have different sources of randomness:

#### DCI (Disentanglement-Completeness-Informativeness)

- **Train/test split**: 80/20 split of samples
- **Lasso regression**: Random initialization in coordinate descent
- **Gradient boosting**: Random subsampling and feature selection

```python
# DCI uses random_state for:
# 1. train_test_split
# 2. All sklearn estimators (Lasso, GradientBoostingRegressor)
dci = DCIMetric(random_state=42)
```

#### MCC (Mean Correlation Coefficient)

- **Train/test split**: For computing test-set correlations
- **Permutation optimization**: Random initialization in linear assignment

```python
# MCC uses random_state for:
# 1. train_test_split
# Note: scipy.optimize.linear_sum_assignment is deterministic
mcc = MCCMetric(random_state=42)
```

#### InfoMEC (Information-theoretic Mechanism Entropy Compression)

- **Train/test split**: For logistic regression evaluation
- **Logistic regression**: Random initialization in solver
- **Cross-validation folds**: Random shuffling in k-fold CV

```python
# InfoMEC uses random_state for:
# 1. train_test_split
# 2. LogisticRegressionCV with cv=5 (stratified k-fold)
infomec = InfoMECMetric(random_state=42)
```

#### MIG (Mutual Information Gap)

- **No randomness**: MIG is fully deterministic
- Uses histogram binning and mutual information (both deterministic)

```python
# MIG does not use random_state
# Results are always identical for same inputs
mig = MIGMetric(num_bins=20)
```

#### T-MEX (Topological Maximum Entropy over Exogenous variables)

- **Minimal randomness**: Only in edge case fallbacks
- Primary computation is deterministic bracket search

```python
# T-MEX uses random_state only for variance estimation fallback
tmex = TMEXMetric(random_state=42)
```

---

## Step-by-Step: Reproducible Evaluation

### Single Metric Evaluation

```python
import numpy as np
from identifiability_guard.metrics import DCIMetric

# Step 1: Set global NumPy seed (defensive)
# This catches any operations that don't respect random_state
np.random.seed(42)

# Step 2: Load your data
Z = np.load("true_factors.npy")      # (n_samples, n_factors)
Z_hat = np.load("learned_codes.npy")  # (n_samples, n_codes)

# Step 3: Create metric with explicit random_state
dci = DCIMetric(random_state=42)

# Step 4: Compute
result = dci.compute(Z, Z_hat)
print(f"DCI: {result.primary_score:.6f}")
# Output: DCI: 0.847362 (always the same)
```

### Multiple Metrics Evaluation

```python
from identifiability_guard.metrics import (
    DCIMetric, MCCMetric, MIGMetric,
    InfoMECMetric, TMEXMetric
)

# Use same seed for all metrics
seed = 42
np.random.seed(seed)

# Create metrics
metrics = {
    "DCI": DCIMetric(random_state=seed),
    "MCC": MCCMetric(random_state=seed),
    "MIG": MIGMetric(num_bins=20),  # Deterministic, no seed needed
    "InfoMEC": InfoMECMetric(random_state=seed),
    "TMEX": TMEXMetric(random_state=seed),
}

# Evaluate all
results = {}
for name, metric in metrics.items():
    result = metric.compute(Z, Z_hat)
    results[name] = result.primary_score
    print(f"{name}: {result.primary_score:.6f}")

# Save results
np.save("metric_results.npy", results)
```

Output (identical every run):
```
DCI: 0.847362
MCC: 0.923451
MIG: 0.756234
InfoMEC: 0.681290
TMEX: 0.892103
```

### Batch Evaluation Over Multiple Models

```python
import numpy as np
from pathlib import Path
from identifiability_guard.metrics import DCIMetric, MCCMetric

# Fixed seed for all evaluations
SEED = 42
np.random.seed(SEED)

# Load ground truth (same across all models)
Z = np.load("ground_truth_factors.npy")

# Evaluate multiple models
model_dirs = Path("models").glob("model_*")
results = []

for model_dir in sorted(model_dirs):  # sorted() ensures consistent order
    # Load learned representations
    Z_hat = np.load(model_dir / "learned_codes.npy")

    # Compute metrics with same seed
    dci = DCIMetric(random_state=SEED)
    mcc = MCCMetric(random_state=SEED)

    results.append({
        "model": model_dir.name,
        "dci": dci.compute(Z, Z_hat).primary_score,
        "mcc": mcc.compute(Z, Z_hat).primary_score,
    })

# Results are reproducible across runs
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

---

## Advanced Topics

### Cross-Validation and Nested Randomness

Some metrics use cross-validation internally (e.g., InfoMEC). The random state controls both the outer train/test split and the inner CV folds:

```python
from identifiability_guard.metrics import InfoMECMetric

# Single seed controls all randomness
infomec = InfoMECMetric(random_state=42)

# Internally:
# 1. train_test_split uses random_state=42
# 2. LogisticRegressionCV with cv=StratifiedKFold(shuffle=True, random_state=42)
# Result: Fully reproducible
```

The library handles this automatically. You only need to set one seed.

### Numerical Precision Across Platforms

Even with fixed random seeds, numerical results can vary slightly across:

- Different CPU architectures (x86 vs ARM)
- Different BLAS libraries (OpenBLAS vs MKL)
- Different NumPy versions

For **bit-exact reproducibility** across platforms:

1. **Pin NumPy version**: Use same version everywhere
   ```bash
   uv pip install numpy==1.24.3
   ```

2. **Use same BLAS**: Specify BLAS library explicitly
   ```bash
   # Use OpenBLAS everywhere
   uv pip install numpy[openblas]
   ```

3. **Avoid threading nondeterminism**: Disable parallelism
   ```python
   import os
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["OPENBLAS_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   ```

For **practical reproducibility** (same results on same machine), the default settings are sufficient. Differences across platforms are typically in the 6th-7th decimal place.

### Parallel Computation

DCI supports parallel computation via `n_jobs` parameter:

```python
# Parallel computation with reproducibility
dci = DCIMetric(random_state=42, n_jobs=4)

# Each parallel job gets a different but deterministic random state
# Library handles this internally using:
# joblib.Parallel(random_state=42)
```

Results are reproducible regardless of `n_jobs` value, but:

- Different `n_jobs` may produce slightly different numerical results (different parallel task scheduling)
- **Same `n_jobs` always produces identical results**

For strict reproducibility across `n_jobs` settings, use `n_jobs=1`.

---

## Validation and Testing

### How to Verify Reproducibility

Test your evaluation pipeline:

```python
import numpy as np
from identifiability_guard.metrics import DCIMetric

# Generate test data
np.random.seed(0)
Z = np.random.randn(1000, 10)
Z_hat = np.random.randn(1000, 10)

# Run metric 10 times with same seed
scores = []
for _ in range(10):
    dci = DCIMetric(random_state=42)
    result = dci.compute(Z, Z_hat)
    scores.append(result.primary_score)

# Check all scores are identical
assert len(set(scores)) == 1, "Metric is not reproducible!"
print(f"✓ All 10 runs produced identical score: {scores[0]:.6f}")
```

### Common Pitfalls

**Pitfall 1: Forgetting to set random_state**

```python
# Wrong: Uses random global state
dci = DCIMetric()  # Will raise error

# Right: Explicit random_state
dci = DCIMetric(random_state=42)
```

The library prevents this by making `random_state` a required parameter for metrics that need it.

**Pitfall 2: Modifying data in place**

```python
# Wrong: Modifies original data
Z_normalized = (Z - Z.mean(axis=0)) / Z.std(axis=0)
Z_normalized += np.random.randn(*Z.shape) * 0.01  # Adds noise

# Right: Create copies
Z_noisy = Z.copy()
Z_noisy += np.random.randn(*Z.shape) * 0.01
```

**Pitfall 3: Using different data splits**

```python
# Wrong: Split before passing to metric
from sklearn.model_selection import train_test_split
Z_train, Z_test = train_test_split(Z, test_size=0.2)
# Metric will split again internally → different splits

# Right: Let metric handle splitting
# Metrics that need splits do it internally with controlled random_state
dci = DCIMetric(random_state=42)
result = dci.compute(Z, Z_hat)  # Handles splitting internally
```

---

## Checklist for Reproducible Evaluation

Before running experiments, verify:

- [ ] All metrics initialized with explicit `random_state`
- [ ] NumPy global seed set (defensive, catches uncontrolled randomness)
- [ ] Data loading order is deterministic (e.g., sorted file paths)
- [ ] NumPy version pinned in requirements
- [ ] For parallel metrics, same `n_jobs` used across runs
- [ ] Results saved with sufficient precision (use `.npy` or float64 in CSV)

Example evaluation script header:

```python
#!/usr/bin/env python3
"""
Reproducible metric evaluation.

Requirements:
- numpy==1.24.3
- scikit-learn==1.3.0
- identifiability-guard==0.1.0

Usage:
    python evaluate.py --seed 42 --models models/*.npy
"""
import numpy as np
from identifiability_guard.metrics import DCIMetric, MCCMetric

# Set global seed
SEED = 42
np.random.seed(SEED)

# Initialize metrics
dci = DCIMetric(random_state=SEED, n_jobs=1)  # n_jobs=1 for strict reproducibility
mcc = MCCMetric(random_state=SEED)
```

---

## Summary

Reproducibility in metric evaluation requires:

1. **Explicit random state control**: Pass `random_state` to all metrics that use randomness
2. **Deterministic operations**: Metrics use deterministic algorithms where possible
3. **Consistent environments**: Pin dependencies for cross-platform reproducibility
4. **Validation**: Test that repeated evaluations produce identical results

The `identifiability-guard` library is designed to make this easy. By requiring `random_state` parameters and handling nested randomness automatically, it ensures that your evaluations are reproducible by default.

When in doubt, set a seed and verify by running twice. If results differ, you've found a source of uncontrolled randomness—file an issue, and we'll fix it.
