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
| **DCI** | Only 1 factor (`num_factors == 1`) → `entropy(base=1)` undefined | `disentanglement_per_code` returns `np.ones(num_codes)` | Mathematically correct (trivially disentangled) | — |
| **DCI** | Only 1 code (`num_codes == 1`) → `entropy(base=1)` undefined in completeness | `completeness_per_factor` returns `np.ones(num_factors)` | Mathematically correct (trivially complete) | — |
| **DCI** | Disentanglement/completeness score has float rounding to ≈ −1e-16 | `np.clip(…, 0.0, 1.0)` | Negligible | — |
| **MIG** | Only 1 code dimension → `sorted_m[1, :]` index out of bounds | Return gap = 0 with warning | Mathematically correct (no second code to compare) | — |
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
| Undercomplete encoder with m=1 code dimension | MIG, DCI | MIG: `sorted_m[1,…]` out of bounds (only 1 row). DCI: `entropy(base=1)` undefined. Both now guarded. |
| Uniform negative correlation with d>2 factors | D2 DGP | Uniform ρ matrix is PSD only if ρ > −1/(d−1). Experiments 3 & 4 use d=2 to allow full (−1,1) range. |
| CI computation with zero standard error | multi_seed | `scipy.stats.t.interval(scale=0)` produces `inf * 0 = NaN`. Guarded by checking `sem > 0`. |

---

## 7  Detailed diff vs. original implementations (T‑MEX, MIG, InfoMEC)

This section lists **every** numerical‑stability guard that was added
compared to the original reference code, for T‑MEX, MIG, and InfoMEC.

**Originals used as reference:**

| Metric | Original file |
|--------|---------------|
| T‑MEX | `context_pycomets-main/pycomets/pcm.py` + `context_a-measurement-perspective-of-crl-main/small_example.py` |
| MIG | `context_disentanglement_lib-master/disentanglement_lib/evaluation/metrics/mig.py` (+ `utils.py`) |
| InfoMEC | `context_latent_quantization-main/disentangle/metrics/infomec.py` |

### 7.1  T‑MEX

| # | Guard | Location (new code) | Original behaviour | What was added |
|---|-------|---------------------|--------------------|----------------|
| 1 | `np.isnan(stat) → stat = -np.inf` | `src/metrics/tmex.py:237–238` | **Already present** in `pycomets/pcm.py:283` (`-np.Inf`) | Nothing — only cosmetic alias change (`np.Inf` → `np.inf`). |
| 2 | Input NaN/Inf rejection | `src/metrics/base.py:196–199` (inherited via `BaseMetric._validate_samples`) | No check in `comp_tmex` or `PCM.test` | `np.all(np.isfinite(Z))` and `np.all(np.isfinite(Z_hat))` — raises `ValueError`. |
| 3 | Output NaN rejection | `src/metrics/base.py:50–52` (inherited via `MetricResult.__post_init__`) | No check — returns raw int error count | `NaN <= x` is `False` in IEEE 754 → a NaN score raises `ValueError`. |
| 4 | Minimum sample count | `src/metrics/tmex.py` — `required_min_samples = 50` | No check | Raises `ValueError` if `n < 50`. |

> **Summary:** Zero new guards inside the PCM algorithm itself. All new
> protections come from the `BaseMetric` framework wrapping (input
> finiteness, output range, minimum samples).

### 7.2  MIG

| # | Guard | Location (new code) | Original behaviour | What was added |
|---|-------|---------------------|--------------------|----------------|
| 1 | Zero‑entropy factors excluded from mean | `src/metrics/mig.py:93` — `valid_mask = entropy > 0.0` | `np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])` — NaN propagates into the mean | Mask out factors where `entropy == 0`; divide only over `valid_mask` elements (line 106). |
| 2 | Warning on excluded factors | `src/metrics/mig.py:96–100` | Silent NaN | `warnings.warn(f"MIG: {num_zero_entropy} factor(s) have zero entropy …")` |
| 3 | All factors constant → return 0.0 | `src/metrics/mig.py:102–104` | NaN score | `return 0.0, {'zero_entropy_factors': num_zero_entropy}` |
| 4 | Single code dimension → gap is 0 | `src/metrics/mig.py:107–111` | `sorted_m[1, …]` → `IndexError` | Guard `sorted_m.shape[0] < 2`; return 0.0 with warning. |
| 5 | `nan_info` tracking | `src/metrics/mig.py:113` (return value) | Not tracked | Returns `{'zero_entropy_factors': int}` alongside score. |
| 6 | Score clipped to [0, 1] | `src/metrics/mig.py:185` — `np.clip(mig_score, 0.0, 1.0)` | Unbounded (can be > 1 or < 0 with noisy discretization) | Ensures `MetricResult` range constraint is satisfied. |
| 6 | `num_bins < 2` validation | `src/metrics/mig.py:161–162` | No check (uses gin config) | `raise ValueError(f"num_bins must be >= 2, got {num_bins}")` |
| 7 | Minimum sample count | `src/metrics/mig.py:167–168` — `max(30, self.num_bins * 2)` | No check | Raises `ValueError` if `n < max(30, 2 * num_bins)`. |
| 8 | `compute_from_matrix`: zero‑MI guard | `src/metrics/mig.py:217–222` | Path does not exist in original | `if max_mi > 1e-10: … else: mig_per_factor[j] = 0.0` — prevents `0/0`. |
| 9 | Input NaN/Inf rejection | `src/metrics/base.py:196–199` (inherited) | No check | `np.all(np.isfinite(…))` — raises `ValueError`. |
| 10 | Output NaN rejection | `src/metrics/base.py:50–52` (inherited) | No check | NaN score → `ValueError`. |

### 7.3  InfoMEC

| # | Guard | Location (new code) | Original behaviour | What was added |
|---|-------|---------------------|--------------------|----------------|
| 1 | NMI normalisation: zero source entropy → `0.0` | `src/metrics/infomec.py:131` — `mi_ij / entropy_i if entropy_i > 0 else 0.0` | `ret[i, :] /= entropy` — unconditional, `0/0 = NaN` | Conditional division; zero entropy yields `nmi[i,j] = 0.0`. |
| 2 | `_process_sources`: histogram‑bins continuous columns | `src/metrics/infomec.py:39–46` | `LabelEncoder` always (creates as many classes as unique values for continuous data → degenerate entropy) | Detects continuous columns (`len(unique_vals) > num_bins`) and uses `np.histogram` + `np.digitize` instead. |
| 3 | No active latents → InfoM = InfoC = 0.0 | `src/metrics/infomec.py:296–308` | No guard; indexes into empty `pruned_nmi` → crash or NaN | Returns `{'infom': 0.0, 'infoc': 0.0, …, 'edge_case': 'no_active_latents'}`. |
| 4 | InfoM: zero column‑sum latents excluded | `src/metrics/infomec.py:315–317` — `valid_cols = col_sums > 0` | `np.max(…, axis=0) / np.sum(…, axis=0)` — `0/0 = NaN` propagates into mean | Exclude zero‑sum columns; compute ratio only over `valid_cols`; track count in `infom_zero_sum_latents`. |
| 5 | InfoM: `num_sources == 1` guard | `src/metrics/infomec.py:321–324` | `(… - 1/num_sources) / (1 - 1/num_sources)` — `0/0` when `num_sources == 1` | Uses `np.mean(modularity_ratios)` directly when `num_sources == 1`. |
| 6 | InfoM: clip to [0, 1] | `src/metrics/infomec.py:327` — `np.clip(infom, 0.0, 1.0)` | Unbounded | Ensures valid score range. |
| 7 | InfoM/C: `np.errstate(divide='ignore', invalid='ignore')` | `src/metrics/infomec.py:314, 330` | No error‑state management | Suppresses NumPy runtime warnings from guarded divisions. |
| 8 | InfoC: zero row‑sum factors excluded | `src/metrics/infomec.py:331–333` — `valid_rows = row_sums > 0` | `np.max(…, axis=1) / np.sum(…, axis=1)` — `0/0 = NaN` propagates into mean | Exclude zero‑sum rows; compute ratio only over `valid_rows`; track count in `infoc_zero_sum_factors`. |
| 9 | InfoC: `num_active_latents == 1` guard | `src/metrics/infomec.py:337–340` | `(… - 1/num_active_latents) / (1 - 1/num_active_latents)` — `0/0` when 1 | Uses `np.mean(compactness_ratios)` directly when `num_active_latents == 1`. |
| 10 | InfoC: clip to [0, 1] | `src/metrics/infomec.py:343` — `np.clip(infoc, 0.0, 1.0)` | Unbounded | Ensures valid score range. |
| 11 | InfoE: logistic regression wrapped in try/except | `src/metrics/infomec.py:161–170` | `model.fit(X, y)` — exception propagates | `except Exception: return np.nan` — caller excludes that factor from average. |
| 12 | InfoE: suppress convergence warnings | `src/metrics/infomec.py:162–164` | No warning management | `warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')` |
| 13 | InfoE: NaN entropy check before division | `src/metrics/infomec.py:224–227` | NaN propagates into NPI | `if np.isnan(h_si_given_z) or np.isnan(h_si): … continue` — factor skipped. |
| 14 | InfoE: `h_si > 0` before dividing | `src/metrics/infomec.py:230–234` | `(h_si - h_si_given_z) / h_si` — `0/0 = NaN` if marginal entropy is 0 | `if h_si > 0: … else: npi = 0.0`. |
| 15 | InfoE: NPI clipped to [0, 1] | `src/metrics/infomec.py:232` — `max(0.0, min(1.0, npi))` | Unbounded (can be negative or > 1) | Clips each per‑factor NPI value. |
| 16 | InfoE: NaN‑filtered averaging + fallback to 0.0 | `src/metrics/infomec.py:244–246` | `np.mean(…)` — NaN contaminates mean | `valid = npi_array[~np.isnan(npi_array)]`; `float(np.mean(valid)) if len(valid) > 0 else 0.0`. |
| 17 | InfoE: warning when factors excluded | `src/metrics/infomec.py:238–242` | Silent | `warnings.warn(f"InfoE: logistic regression failed for {nan_entropy_count} factor(s); …")` |
| 18 | InfoE: `nan_info` tracking | `src/metrics/infomec.py:248–252` (return value) | Not tracked | Returns `{'logistic_regression_failures': int, 'factors_used': int, 'factors_total': int}`. |
| 19 | Input NaN/Inf rejection | `src/metrics/base.py:196–199` (inherited) | No check | `np.all(np.isfinite(…))` — raises `ValueError`. |
| 20 | Output NaN rejection | `src/metrics/base.py:50–52` (inherited) | No check | NaN score → `ValueError`. |