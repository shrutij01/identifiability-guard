# Identifiability Metrics

Metrics quantify how well learned representations $\hat{Z}$ recover ground-truth factors $Z$.

## Overview

All metrics inherit from `BaseMetric` and return `MetricResult` objects with:
- `primary_score`: Main metric value (float)
- `subscores`: Dictionary of sub-metrics (optional)
- `metadata`: Additional diagnostic information

| Metric | Primary Score | Subscores | Range |
|--------|--------------|-----------|-------|
| MCC | Mean max correlation | Pearson, Spearman, RDC | [0, 1] |
| DCI | Disentanglement | Completeness, Informativeness | [0, 1] |
| R² | Coefficient of determination | - | [0, 1] |
| MIG | Mutual Information Gap | - | [0, ∞) |
| T-MEX | p-value for exchangeability | - | [0, 1] |
| InfoMEC | Modularity | Explicitness, Compactness | [0, 1] |

## MCC (Mean Correlation Coefficient)

Measures alignment via correlation-based matching.

```python
from identifiability_guard.metrics import MCC

mcc = MCC()
result = mcc.compute(Z, Z_hat)

print(f"MCC (Pearson): {result.primary_score:.3f}")
print(f"Spearman: {result.subscores['spearman']:.3f}")
print(f"RDC: {result.subscores['rdc']:.3f}")
```

**Mathematical Definition:**
$$\text{MCC}(\rho) = \frac{1}{k} \max_{\pi \in S_k} \sum_{i=1}^k |\text{Corr}(Z_i, \hat{Z}_{\pi(i)})|$$

**Parameters:**
- None (uses default correlation methods)

**Returns:**
- `primary_score`: Pearson-based MCC
- `subscores`: `{'pearson', 'spearman', 'rdc'}`

**Interpretation:**
- 1.0: Perfect recovery (possibly with permutation/scaling)
- 0.0: No correlation between Z and Ẑ

## DCI (Disentanglement, Completeness, Informativeness)

Evaluates three complementary aspects of identifiability.

```python
from identifiability_guard.metrics import DCI

dci = DCI()
result = dci.compute(Z, Z_hat)

print(f"Disentanglement: {result.subscores['disentanglement']:.3f}")
print(f"Completeness: {result.subscores['completeness']:.3f}")
print(f"Informativeness: {result.subscores['informativeness']:.3f}")
```

**Components:**
- **Disentanglement**: Each code depends on at most one factor
- **Completeness**: Each factor is captured by at most one code
- **Informativeness**: Predictive power of codes for factors

**Parameters:**
- `regressor` (str): 'lasso', 'gradient_boosting', or 'random_forest' (default: 'gradient_boosting')

**Returns:**
- `primary_score`: Disentanglement score
- `subscores`: `{'disentanglement', 'completeness', 'informativeness'}`

**Interpretation:**
- Disentanglement = 1.0: Each code captures exactly one factor
- Completeness = 1.0: Each factor is captured by exactly one code
- Informativeness → 1.0: Codes are highly predictive of factors

## R² (Coefficient of Determination)

Measures variance explained by optimal linear regression.

```python
from identifiability_guard.metrics import R2

r2 = R2()
result = r2.compute(Z, Z_hat)

print(f"R²: {result.primary_score:.3f}")
```

**Mathematical Definition:**
$$R^2 = 1 - \frac{\sum (Z_i - \hat{Z}_i)^2}{\sum (Z_i - \bar{Z})^2}$$

**Returns:**
- `primary_score`: R² score

**Interpretation:**
- 1.0: Perfect linear prediction
- 0.0: No better than predicting mean

## MIG (Mutual Information Gap)

Quantifies the gap between the top two mutual information values.

```python
from identifiability_guard.metrics import MIG

mig = MIG()
result = mig.compute(Z, Z_hat)

print(f"MIG: {result.primary_score:.3f}")
```

**Mathematical Definition:**
$$\text{MIG} = \frac{1}{d} \sum_{j=1}^d \left( I(Z_j; \hat{Z}_{(1)}) - I(Z_j; \hat{Z}_{(2)}) \right)$$

where $(1), (2)$ denote the top two codes by MI with factor $Z_j$.

**Returns:**
- `primary_score`: MIG score

**Interpretation:**
- Higher values indicate clearer factor-code associations
- 0.0: Multiple codes equally informative about each factor

## T-MEX (Testing for Measurement Exchangeability)

Statistical test for whether two representations are exchangeable.

```python
from identifiability_guard.metrics import TMEX

tmex = TMEX()
result = tmex.compute(Z, Z_hat)

print(f"T-MEX p-value: {result.primary_score:.3f}")
```

**Returns:**
- `primary_score`: p-value for exchangeability test

**Interpretation:**
- p < 0.05: Reject exchangeability (representations differ)
- p > 0.05: Cannot reject (representations may be equivalent)

## InfoMEC (Modularity, Explicitness, Compactness)

Information-theoretic decomposition of identifiability.

```python
from identifiability_guard.metrics import InfoMEC

infomec = InfoMEC()
result = infomec.compute(Z, Z_hat)

print(f"Modularity: {result.subscores['modularity']:.3f}")
print(f"Explicitness: {result.subscores['explicitness']:.3f}")
print(f"Compactness: {result.subscores['compactness']:.3f}")
```

**Components:**
- **InfoM (Modularity)**: Each code is informative about one factor
- **InfoE (Explicitness)**: Information is explicitly represented
- **InfoC (Compactness)**: Information is compactly represented

**Returns:**
- `primary_score`: Modularity score
- `subscores`: `{'modularity', 'explicitness', 'compactness'}`

You can also compute individual components:

```python
from identifiability_guard.metrics import InfoM, InfoE, InfoC

infom = InfoM().compute(Z, Z_hat)
infoe = InfoE().compute(Z, Z_hat)
infoc = InfoC().compute(Z, Z_hat)
```

## Metric Registry

Unified interface for computing multiple metrics.

```python
from identifiability_guard.metrics import MetricRegistry

# Create registry and register all metrics
registry = MetricRegistry()
registry.register_defaults()

# Compute all metrics at once
results = registry.compute_all(Z, Z_hat)

for metric_name, result in results.items():
    print(f"{metric_name}: {result.primary_score:.3f}")

# Or use specific metrics
mcc = registry.create("mcc")
result = mcc.compute(Z, Z_hat)
```

**Available metric names:**
- `"dci"` - DCI metric
- `"mcc"` - MCC metric
- `"r2"` - R² metric
- `"mig"` - MIG metric
- `"tmex"` - T-MEX metric
- `"infomec"` - InfoMEC metric
- `"infom"` - InfoM only
- `"infoe"` - InfoE only
- `"infoc"` - InfoC only

## Base Class API

All metrics implement the `BaseMetric` interface:

```python
class BaseMetric:
    def compute(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """
        Compute metric.

        Args:
            Z: Ground-truth factors of shape (n_samples, d)
            Z_hat: Learned representations of shape (n_samples, m)

        Returns:
            MetricResult with primary_score, subscores, and metadata
        """
        pass
```

**MetricResult structure:**
```python
@dataclass
class MetricResult:
    primary_score: float
    subscores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Usage Tips

**Comprehensive Evaluation:**
```python
from identifiability_guard.metrics import MetricRegistry

registry = MetricRegistry()
registry.register_defaults()

# Compute all metrics
results = registry.compute_all(Z, Z_hat)

# Extract scores for analysis
scores = {
    "mcc_pearson": results["mcc"].primary_score,
    "dci_d": results["dci"].subscores["disentanglement"],
    "dci_c": results["dci"].subscores["completeness"],
    "r2": results["r2"].primary_score,
    "mig": results["mig"].primary_score,
}
```

**Handling Dimensionality Mismatch:**
```python
# Metrics automatically handle m ≠ d
Z = dgp.sample(1000)  # Shape: (1000, 5)
Z_hat = encoder.encode(Z)  # Shape: (1000, 10) - overcomplete

# All metrics work correctly
mcc_result = MCC().compute(Z, Z_hat)  # Matches via optimal assignment
dci_result = DCI().compute(Z, Z_hat)  # Trains regressors Z_hat -> Z
```

**Statistical Validation:**
```python
# Use T-MEX to test if two encoders produce equivalent representations
encoder1 = E1ElementwiseLinear(d=5, seed=42)
encoder2 = E3LinearlyEntangled(d=5, seed=123)

Z_hat1 = encoder1.encode(Z)
Z_hat2 = encoder2.encode(Z)

tmex = TMEX()
result = tmex.compute(Z_hat1, Z_hat2)

if result.primary_score < 0.05:
    print("Representations are significantly different")
else:
    print("Cannot reject exchangeability")
```

**Computational Considerations:**
```python
# Some metrics are expensive on large datasets
# Use sampling for quick estimates
Z_large = dgp.sample(100_000)
Z_hat_large = encoder.encode(Z_large)

# Subsample for faster computation
indices = np.random.choice(100_000, size=5000, replace=False)
Z_subset = Z_large[indices]
Z_hat_subset = Z_hat_large[indices]

results = registry.compute_all(Z_subset, Z_hat_subset)
```

## Metric Comparison

| Metric | Computational Cost | Handles m≠d | Interpretability | Use Case |
|--------|-------------------|-------------|------------------|----------|
| MCC | Low | ✅ Yes | High | Quick correlation check |
| DCI | Medium | ✅ Yes | High | Comprehensive analysis |
| R² | Low | ✅ Yes | High | Linear recovery quality |
| MIG | Medium | ✅ Yes | Medium | Information-theoretic view |
| T-MEX | High | ✅ Yes | Low | Statistical testing |
| InfoMEC | High | ✅ Yes | Medium | Detailed decomposition |

**Recommended workflow:**
1. Start with MCC for quick assessment
2. Use DCI for detailed disentanglement analysis
3. Add MIG/InfoMEC for information-theoretic perspective
4. Use T-MEX for statistical validation when needed
