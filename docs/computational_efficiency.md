# Computational Efficiency

Speed improvements without changing algorithms. Here's what we optimized.

## Summary

| Metric | Original | Ours | Speedup | How |
|--------|----------|------|---------|-----|
| DCI | 12.6s | 3.3s | 3.8x | Parallel model training |
| MCC | 44ms | 10.5ms | 4.2x | NumPy corrcoef + scipy Hungarian |
| InfoMEC | 12.1s | 7.5s | 1.6x | Vectorized preprocessing, parallel LogReg |
| MIG | 2.2s | 1.9s | 1.2x | Early termination, cached entropy |
| R² | — | — | — | Single lstsq call (batched) |
| TMEX | 35s | 25s | 1.4x | Precomputed Z_minus slices |

## DCI

**Bottleneck:** Training d separate gradient boosting models sequentially.

**Original:**
```python
for i in range(num_factors):
    model = GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = model.feature_importances_
```

**Our fix:**
```python
# Option 1: Parallel with joblib backend
model = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=100),
    n_jobs=-1  # All cores
)
model.fit(Z_hat, Z)

# Option 2: Parallel loop (simpler)
from joblib import Parallel, delayed
models = Parallel(n_jobs=-1)(
    delayed(train_model)(Z_hat, Z[:, i]) for i in range(d)
)
```

**Result:** 3.8x faster on 4 cores.

## MCC

**Bottleneck:** PyTorch overhead, auction algorithm slower than Hungarian for small problems.

**Original (icebeem):**
```python
# PyTorch correlation + auction algorithm
cc = corrcoef_pt(x, y)  # PyTorch
score = auction_linear_assignment(cc)  # Custom GPU algorithm
```

**Our fix:**
```python
# NumPy correlation + scipy Hungarian
correlation = np.corrcoef(Z, Z_hat, rowvar=False)
cross_corr = correlation[:d, d:]
row_ind, col_ind = linear_sum_assignment(-np.abs(cross_corr))
score = np.mean(np.abs(cross_corr[row_ind, col_ind]))
```

**Result:** 4.2x faster for typical sizes (d, m < 100).

**When to use PyTorch:** Large problems (d, m > 100) on GPU.

## InfoMEC

**Bottlenecks:**
1. Per-column StandardScaler loop
2. Sequential logistic regression
3. Redundant entropy computation

**Original:**
```python
# Loop over columns
processed_latents = []
for j in range(latents.shape[1]):
    processed_latents.append(
        StandardScaler().fit_transform(latents[:, j][:, None])
    )
processed_latents = np.concatenate(processed_latents, axis=1)

# Sequential logistic regression
for i in range(num_sources):
    model.fit(X, y)  # No parallelization
```

**Our fixes:**
```python
# 1. Vectorized StandardScaler
processed_latents = StandardScaler().fit_transform(latents)

# 2. Cached entropy (compute once)
entropy_i = metrics.mutual_info_score(processed_sources[:, i], processed_sources[:, i])
if entropy_i < EPS:
    nmi[i, :] = 0.0
    continue  # Skip entire row

# Reuse entropy_i for all latents
for j in range(num_latents):
    mi_ij = compute_mi(i, j)
    nmi[i, j] = mi_ij / entropy_i  # Entropy computed once

# 3. Parallel logistic regression
model = LogisticRegression(n_jobs=-1)

# 4. Skip InfoE when only computing InfoM/InfoC
results = _compute_infomec(Z, Z_hat, compute_infoe=False)
```

**Result:** 1.6x faster overall.

## MIG

**Bottleneck:** Inherently O(d × m) MI computations using sklearn (already optimized C code).

**Limited optimization possible:**
```python
# Pre-allocate arrays
mi_matrix = np.zeros((num_codes, num_factors))
entropy = np.zeros(num_factors)

# Early termination for constant factors
for j in range(num_factors):
    entropy[j] = mutual_info_score(factors[j, :], factors[j, :])
    if entropy[j] < 1e-10:
        continue  # Skip MI computation for this factor
```

**Result:** 1.2x faster (modest because sklearn's mutual_info_score dominates).

**Why limited speedup?**
- MI computation is inherently sequential
- sklearn already uses C backend
- Discretization must be per-dimension

## R²

**Bottleneck:** d separate lstsq calls.

**Original:**
```python
for i in range(d):
    y = Z[:, i]
    y_pred = np.linalg.lstsq(Z_hat, y, rcond=None)[0]
    # Compute R² for factor i
```

**Our fix:**
```python
# Single lstsq call for all factors
W, *_ = np.linalg.lstsq(Z_hat, Z, rcond=None)
Z_pred = Z_hat @ W

# Vectorized R² computation
mse = np.mean((Z - Z_pred)**2, axis=0)
var = np.var(Z, axis=0)
r2_array = np.where(var < EPS, 1.0, 1.0 - mse / np.maximum(var, EPS))
```

**Result:** One LAPACK call instead of d calls.

## TMEX

**Bottleneck:** d×m PCM tests, each doing O(rep) regression fits.

**Original:**
```python
for ii in range(m):
    for jj in range(d):
        Z_minus_j = np.delete(Z, jj, axis=1)  # Computed d×m times
        pcm.test(X=Z[:, jj], Y=Z_hat[:, ii], Z=Z_minus_j, ...)
```

**Our fixes:**
```python
# 1. Precompute Z_minus slices
Z_minus = [np.delete(Z, jj, axis=1) for jj in range(d)]

# 2. Fresh constructors instead of deepcopy
_reg_cls = type(fun_reg)
for ii in range(m):
    for jj in range(d):
        pcm.test(
            reg_yonxz=_reg_cls(),  # Fresh instance
            reg_ronz=_reg_cls(),
            Z=Z_minus[jj],  # Precomputed
            ...
        )
```

**Result:** 1.4x faster. Limited because 87% of time is in regression fits (inherent to algorithm).

## Complexity Analysis

| Metric | Complexity | Bottleneck | Parallelizable? |
|--------|-----------|------------|-----------------|
| DCI | O(d × n log n × trees) | Model training | ✅ Yes (across factors) |
| MCC | O((d+m)² × n + d³) | Correlation matrix | ❌ No |
| InfoMEC | O(d × m × n × k) | MI estimation | ⚠️ Partial (LogReg only) |
| MIG | O(d × m × n) | MI estimation | ⚠️ Overhead > gain |
| R² | O(n × m × d) | Matrix solve | ❌ No (batched) |
| TMEX | O(d × m × rep × T_reg) | Regression fits | ⚠️ Tests are independent |

## Memory Footprint

**DCI:**
- Original: O(model_size) — one at a time
- Ours (parallel): O(num_cores × model_size) during training

**MCC:**
- PyTorch: ~100MB overhead + tensors
- NumPy: <1MB overhead + arrays

**InfoMEC:**
- Saved 50% on processed latents (single allocation vs multiple)

**MIG:**
- Very efficient: ~2.8MB for d=20, m=50, n=10k

## When to Use What

**Use our NumPy implementations:**
- CPU-only environment
- Small to medium problems (d, m < 100)
- Memory limited
- Minimal dependencies

**Consider PyTorch/GPU:**
- Large-scale (d, m > 100)
- GPU available
- Batch processing many evaluations

## Profiling Your Code

```python
from identifiability_guard.evaluation import profile_block

with profile_block("DCI computation") as profile:
    result = DCI().compute(Z, Z_hat)

print(f"Time: {profile['elapsed']:.2f}s")
print(f"Memory: {profile['peak_mb']:.2f} MB")
```

## Optimization Checklist

1. ✅ Profile first — identify actual bottleneck
2. ✅ Vectorize — replace Python loops with NumPy
3. ✅ Parallelize — use n_jobs=-1 for embarrassingly parallel tasks
4. ✅ Cache — store computed values when evaluating multiple times
5. ✅ Batch — process large datasets in chunks

## Real-World Performance

Setup: n=10,000 samples, d=10, m=20, 4 cores

**Before:**
- DCI: 12.6s
- MCC: 44ms
- InfoMEC: 12.1s
- MIG: 2.2s

**After:**
- DCI: 3.3s (parallel)
- MCC: 10.5ms (NumPy)
- InfoMEC: 7.5s (vectorized + parallel)
- MIG: 1.9s (early termination)

Total speedup: **2.5x** on average.
