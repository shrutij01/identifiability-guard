# Numerical Stability & NaN Handling

This document summarises every measure taken across the codebase to prevent,
detect, and handle NaN / Inf / numerical‑instability issues.

---

## 1  Boundary defences (input & output validation)

### 1.1  Input validation — `BaseMetric._validate_samples()`

**File:** `src/metrics/base.py`

Every call to `metric.compute(Z, Z_hat)` passes through
`_validate_samples()` before any computation begins.  The check

```python
if not np.all(np.isfinite(Z)):
    raise ValueError("Z contains NaN or Inf values")
```

is applied to both `Z` and `Z_hat`.  Any NaN or Inf in the raw inputs is a
**hard error** — the metric refuses to run.  This prevents corrupt data from
silently propagating into scores.

The same `np.isfinite` check exists in `_validate_matrix()` for the
`compute_from_matrix()` path.

### 1.2  Output validation — `MetricResult.__post_init__()`

**File:** `src/metrics/base.py`

`MetricResult` is a frozen dataclass.  On construction it asserts:

```python
if not (self.score_min <= self.primary_score <= self.score_max): ...
```

Because `NaN <= x` is always `False` in IEEE 754, a NaN primary score (or
subscore) will **raise `ValueError`** rather than being stored silently.
This acts as a last line of defence: if a metric's internal logic accidentally
produces NaN, the result cannot be returned to the caller.

### 1.3  Evaluation‑layer validation — `validate_array()`

**File:** `src/evaluation/helpers.py`

`evaluate_combination()` calls `validate_array(Z, …)` and
`validate_array(Z_hat, …)` before handing data to the registry.
This function **raises `ValueError`** on NaN or Inf — it never silently
mutates data.

> The previous `sanitize_array()` replaced NaN with 0 and Inf with ±1e10.
> This was removed because it silently injected false observations (a vector
> of zeros looks like a real sample at the origin) and could bias every
> metric.  `sanitize_array` still exists for backward compatibility but
> emits a `DeprecationWarning` and delegates to `validate_array`.

---

## 2  Per‑metric internal guards

Each metric can encounter division‑by‑zero or degenerate‑data situations
*after* the inputs have already passed validation.  The table below
summarises every such guard.

| Metric | Trigger | Guard | Bias direction | Tracked in metadata |
|--------|---------|-------|----------------|---------------------|
| **MIG** | Factor has zero entropy after discretization (constant column) → `gap / 0` | Exclude that factor from the mean; warn via `warnings.warn` | None (factor skipped) | `nan_info.zero_entropy_factors` |
| **InfoE** | Logistic regression raises an exception (e.g. singular data) | Return `np.nan` for that factor's conditional entropy; exclude factor from average; warn | None (factor skipped) | `nan_info.logistic_regression_failures`, `factors_used`, `factors_total` |
| **InfoM** | A latent has zero total NMI across all factors → `max/0` | Exclude that latent column from the modularity ratio | None (latent skipped) | `nan_info.infom_zero_sum_latents` |
| **InfoC** | A factor has zero total NMI across all latents → `max/0` | Exclude that factor row from the compactness ratio | None (factor skipped) | `nan_info.infoc_zero_sum_factors` |
| **InfoMEC (NMI matrix)** | Source entropy = 0 when normalising MI | `nmi[i,j] = 0.0` when `entropy_i == 0` | Downward (conservative) | — |
| **InfoMEC (edge)** | No active latents (all ranges ≈ 0) | Return InfoM = InfoC = 0.0; still compute InfoE | Downward (conservative) | `nan_info.edge_case` |
| **MCC** | Constant column → NaN in `np.corrcoef` | `np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)` | Downward (zero correlation assumed) | `nan_info.z_constant_columns`, `zhat_constant_columns` |
| **MCC** | Final score is non‑finite | Replace with 0.0 | Downward | `nan_info.score_was_nonfinite` |
| **R²** | Factor has zero variance → `1 - mse/0` | Set R² = 1.0 (constant perfectly explained) | — (mathematically correct) | `nan_info.zero_variance_factors` |
| **R²** | `lstsq` produces non‑finite R² | `np.nan_to_num(…, nan=0.0, posinf=0.0, neginf=0.0)` before averaging | Downward | `nan_info.nonfinite_r2_count` |
| **DCI** | `log(0)` in `scipy.stats.entropy` | Add `1e-11` epsilon to importance matrix | Negligible | — |
| **DCI** | Importance matrix sums to 0 (no signal) | Replace with `np.ones_like` for weighting | Uniform weighting (neutral) | — |
| **DCI** | Informativeness (R² from GBT) can be negative | `np.clip(…, 0.0, 1.0)` | Upward (negative clipped to 0) | Raw value stored in `metadata.test_informativeness_raw` |
| **T-MEX** | PCM test statistic is NaN (variance of L = 0) | `stat = -np.inf` → `pval = 1.0` (no rejection) | Conservative (cannot reject null) | — |

---

## 3  NaN tracking via `metadata['nan_info']`

Every metric that performs an internal NaN substitution or exclusion records
what happened in the `nan_info` key of `MetricResult.metadata`.  This lets
downstream code (evaluation scripts, dashboards) detect when a score was
influenced by numerical edge cases.

### 3.1  Structure by metric

**MIG**
```python
{'zero_entropy_factors': int}
```

**InfoE** (standalone or inside InfoMEC)
```python
{'logistic_regression_failures': int,
 'factors_used': int,
 'factors_total': int}
```

**InfoM / InfoC / InfoMEC** (combined)
```python
{'infom_zero_sum_latents': int,
 'infoc_zero_sum_factors': int,
 'infoe': { … },          # same structure as InfoE above
 'edge_case': str | absent}
```

**MCC**
```python
{'z_constant_columns': int,
 'zhat_constant_columns': int,
 'score_was_nonfinite': bool}
```

**R²**
```python
{'zero_variance_factors': int,
 'nonfinite_r2_count': int}
```

---

## 4  Evaluation‑layer NaN handling

### 4.1  Failed metric runs — `sensitivity.py` / `multi_seed.py`

When a metric computation raises an exception during a parameter sweep or
multi‑seed evaluation, the score for that run is recorded as `np.nan`.
Back‑filling ensures all metrics have the same array length regardless of
which runs failed:

```python
seed_results[metric_name].append(np.nan)
```

### 4.2  Statistics computation — `compute_sensitivity_statistics()`

Before computing mean / std / confidence intervals the function filters NaN
values:

```python
valid_values = values_array[~np.isnan(values_array)]
```

If all values are NaN the statistics themselves are reported as NaN.

### 4.3  Score extraction — `extract_metric_scores()`

`extract_metric_scores()` initialises every requested metric name with
`np.nan` and only overwrites if the result is not `None`.  Downstream code
therefore always receives a complete dict, with NaN signalling "not
computed".

---

## 5  Design rationale

| Principle | Implementation |
|-----------|---------------|
| **Fail fast at the boundary** | `_validate_samples` and `validate_array` reject NaN/Inf inputs immediately. |
| **Never silently inject data** | The old `sanitize_array` (replace NaN→0, Inf→1e10) was deprecated because it created fake observations that could bias metrics. |
| **Exclude, don't substitute** | Internally, metrics that encounter degenerate dimensions (zero entropy, zero variance, logistic regression failure) *exclude* that dimension from the average rather than substituting an arbitrary value. |
| **Downward bias when forced to substitute** | When exclusion is not possible (e.g. a single NaN correlation entry in a matrix that feeds the Hungarian algorithm in MCC), the fallback is 0.0 — the conservative/pessimistic value. |
| **Track everything** | All substitutions and exclusions are recorded in `metadata['nan_info']` so users can audit whether a score was affected. |
| **Warn loudly** | `warnings.warn()` is called whenever a dimension is excluded, so users see the message even if they don't inspect metadata. |

---

## 6  Situations that commonly trigger NaN guards

| Situation | Metrics affected | Root cause |
|-----------|-----------------|------------|
| Very small sample size (n < 50) | MIG, InfoE, T-MEX | Histogram bins collapse (zero entropy); logistic regression underdetermined; PCM variance = 0 |
| Constant factor in Z | MIG, R² | Zero entropy / zero variance after discretization |
| Constant or near-constant code in Z_hat | MCC, InfoM/C | NaN correlation; zero NMI column |
| Overcomplete disjoint encoder (E8) | InfoM | Many inactive latents → zero‑sum NMI columns |
| Highly entangled encoder | InfoE | Logistic regression may fail to converge |
