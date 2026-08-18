# Numerical Stability

Identifiability metrics encounter degenerate inputs routinely: constant factors,
inactive codes, zero-entropy discretizations, singular regressions, and
non-finite intermediate values. The library handles these cases at three
levels: reject invalid inputs, apply explicit metric-level fallbacks, and
record failed evaluation runs without silently changing the data.

## Stability contract

### Reject invalid inputs

`BaseMetric` validates every call before dispatching to a metric implementation.
The checks live in `src/identifiability_guard/metrics/base.py`:

```python
if not np.all(np.isfinite(Z)):
    raise ValueError("Z contains NaN or Inf values")
if not np.all(np.isfinite(Z_hat)):
    raise ValueError("Z_hat contains NaN or Inf values")
```

`_validate_samples()` also checks that inputs are two-dimensional, have the
same number of samples, satisfy the metric's minimum sample count, and contain
at least one factor and one code. `_validate_matrix()` applies the analogous
shape and finiteness checks to precomputed relationship matrices.

The evaluation layer performs the same fail-fast check through
`validate_array()` in `src/identifiability_guard/evaluation/helpers.py`.
Neither layer replaces NaN or Inf with synthetic observations.

### Reject invalid outputs

Every metric returns a `MetricResult`. Its constructor checks that the primary
score and all subscores lie within the declared score range. Because comparisons
with NaN are false, non-finite scores cannot pass this validation silently.

### Handle internal degeneracy explicitly

Finite inputs can still produce undefined intermediate quantities. For example,
a constant factor has zero entropy, and a constant code has undefined
correlation. Each metric therefore defines a local fallback rather than relying
on global sanitization.

## Metric-specific guards

| Metric | Degenerate case | Behaviour |
| --- | --- | --- |
| DCI | Zero or highly concentrated importance matrix | Use scale-aware entropy smoothing; use uniform weights when total importance is zero; clip scores to `[0, 1]`. |
| MCC | Constant columns or non-finite correlations | Map non-finite correlation entries to zero before assignment; bound the legacy auction algorithm; clip the final score. |
| InfoMEC | Zero-sum NMI rows or columns | Exclude uninformative factors or codes from the corresponding ratios. |
| InfoMEC | No active learned coordinates | Return `InfoM = InfoC = 0` and compute InfoE when requested. |
| InfoE | Logistic-regression failure | Exclude the affected factor from the average and emit a warning. |
| MIG | Zero-entropy factor | Exclude the factor and emit a warning; return zero if every factor is excluded. |
| MIG | One learned coordinate | Return a zero gap by default; `single_code_gap_zero=False` enables the top-MI convention. |
| R² | Zero-variance factor | Treat its contribution as zero because R² is undefined. |
| R² | Non-finite or negative value | Replace or clip it conservatively into `[0, 1]`. |
| T-MEX | Failed variance bracket or non-finite statistic | Return `p = 1` / `stat = -inf`, or floor the variance, so the fallback cannot create a false rejection. |

### DCI

DCI uses a scale-aware epsilon when computing entropy and avoids dividing by a
zero total importance:

```python
eps = safe_entropy_eps(importance_matrix)
raw = 1.0 - scipy.stats.entropy(
    importance_matrix.T + eps,
    base=importance_matrix.shape[1],
    axis=0,
)
per_code = np.clip(raw, 0.0, 1.0)

if importance_matrix.sum() < EPS:
    importance_matrix = np.ones_like(importance_matrix)
```

Gradient-boosting informativeness can be negative for continuous factors. The
reported score is clipped to `[0, 1]`, while the raw value remains available in
the result metadata.

### MCC

Constant columns can make a correlation matrix non-finite. Both NumPy and
PyTorch paths convert those entries to zero before solving the assignment:

```python
cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
```

The PyTorch covariance path floors zero standard deviations, and the legacy
auction solver has an iteration cap. The result metadata records the number of
constant columns in both the factors and learned representation.

### InfoMEC

InfoM and InfoC exclude zero-sum NMI columns or rows rather than dividing by
zero. Constant factors receive zero NMI. If InfoE's logistic regression fails
for a factor, that factor is excluded from the average; if every factor fails,
the score is zero.

### MIG

MIG excludes factors whose discretized entropy is zero. With only one learned
coordinate, the default gap is zero because there is no runner-up coordinate:

```python
if sorted_m.shape[0] < 2:
    if single_code_gap_zero:
        return 0.0, {"zero_entropy_factors": num_zero_entropy}
    per_factor = sorted_m[0, valid_mask] / entropy[valid_mask]
```

### R²

R² is undefined for zero-variance targets. Those factors contribute zero, and
remaining non-finite or negative values are mapped conservatively before the
average:

```python
zero_var_mask = var < EPS
r2_array = np.where(
    zero_var_mask,
    0.0,
    1.0 - mse / np.maximum(var, EPS),
)
r2_array = np.nan_to_num(r2_array, nan=0.0, posinf=1.0, neginf=0.0)
r2_array = np.clip(r2_array, 0.0, 1.0)
```

### T-MEX

The projected-covariance test bounds its variance-bracket search. A failed
search returns `p = 1` and `stat = -inf`; otherwise, the variance is floored
before division:

```python
var_L = max(np.mean(L**2) - np.mean(L) ** 2, 1e-12)
stat = np.sqrt(n_test) * np.mean(L) / np.sqrt(var_L)
if not np.isfinite(stat):
    stat = -np.inf
```

These fallbacks are conservative: numerical failure cannot be interpreted as
evidence against the null.

## Shared numerical utilities

Common helpers live in
`src/identifiability_guard/metrics/_numerical.py`:

```python
EPS = 1e-12
SCALE_EPS_FACTOR = 1e-10

def safe_entropy_eps(values):
    return max(EPS, float(np.max(np.abs(values))) * SCALE_EPS_FACTOR)

def safe_divide(numerator, denominator, fallback=0.0):
    if abs(denominator) < EPS:
        return fallback
    return numerator / denominator

def clamp(value, lo=0.0, hi=1.0):
    return float(np.clip(value, lo, hi))

def sanitize_mi(mi_value):
    if not np.isfinite(mi_value) or mi_value < 0:
        return 0.0
    return float(mi_value)
```

## Reporting numerical fallbacks

Metrics that exclude dimensions or apply numerical fallbacks expose counts in
`MetricResult.metadata["nan_info"]`. This makes a finite score auditable rather
than hiding the edge case.

| Metric | Representative fields |
| --- | --- |
| MIG | `zero_entropy_factors` |
| InfoE | `logistic_regression_failures`, `factors_used`, `factors_total` |
| InfoM / InfoC | `infom_zero_sum_latents`, `infoc_zero_sum_factors`, nested `infoe` details |
| MCC | `z_constant_columns`, `zhat_constant_columns` |
| R² | `zero_variance_factors`, `nonfinite_r2_count`, `valid_factors`, `total_factors` |

Metrics also emit warnings when a failure excludes part of the score. Callers
that need strict behaviour can promote these warnings to errors.

## Evaluation-layer failures

Parameter sweeps and multi-seed evaluations distinguish a failed run from a
valid low score:

- `sensitivity.py` and `multi_seed.py` record failed metric runs as `np.nan`.
- `extract_metric_scores()` initializes requested-but-missing metrics to
  `np.nan`.
- `compute_sensitivity_statistics()` excludes NaN runs from its mean, standard
  deviation, and confidence interval. If every run failed, all reported
  statistics remain NaN.

This use of NaN is confined to evaluation results; NaN is never accepted as a
metric input or valid `MetricResult` score.

## Design principles

| Principle | Implementation |
| --- | --- |
| Fail fast at boundaries | Reject malformed or non-finite inputs before metric computation. |
| Never inject observations | Do not replace invalid samples with zeros or large finite values. |
| Make degeneracy metric-specific | Define the meaning of constant or inactive dimensions within each metric. |
| Prefer conservative fallbacks | Use zero contribution or failure-to-reject when exclusion is impossible. |
| Preserve observability | Record exclusions and substitutions in metadata and warnings. |
| Keep failed runs distinct | Represent evaluation failures as NaN rather than as a valid score of zero. |

## Common triggers

| Situation | Commonly affected metrics | Reason |
| --- | --- | --- |
| Very small sample size | MIG, InfoE, T-MEX | Histogram bins collapse, regressions become underdetermined, or variance estimates degenerate. |
| Constant factor | MIG, R², InfoMEC | Zero entropy, zero variance, or zero NMI. |
| Constant or inactive code | MCC, InfoM, InfoC | Undefined correlation or zero-sum NMI. |
| Strongly overcomplete representation | MCC, InfoM | More opportunities for spurious matching and inactive coordinates. |
| Degenerate predictive model | DCI, InfoE | A fitted model has no usable signal or fails to converge. |

## Testing

The focused regression suite currently contains 13 tests covering constant
columns, single-code MIG, zero-importance DCI, T-MEX variance guards, and related
edge cases:

```bash
PYTHONPATH=src pytest -q tests/test_numerical_stability.py
```

Run the complete suite before changing fallback semantics, because the guards
also interact with metric, evaluation, and low-level MCC tests:

```bash
PYTHONPATH=src pytest -q
```
