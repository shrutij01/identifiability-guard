# API Reference

Complete API documentation for Identifiability Guard.

## Modules

- [DGP (Data Generating Processes)](dgp.md)
- [Encoders](encoders.md)
- [Metrics](metrics.md)
- [Evaluation Utilities](evaluation.md)

## Quick Navigation

### Data Generating Processes

- [`BaseDGP`](dgp.md#basedgp) - Abstract base class
- [`D1Independent`](dgp.md#d1independent) - Independent factors
- [`D2Correlated`](dgp.md#d2correlated) - Correlated factors
- [`D3SingleRedundant`](dgp.md#d3singleredundant) - Single-factor redundancy
- [`D4MultiRedundant`](dgp.md#d4multiredundant) - Multi-factor redundancy

### Encoders

- [`BaseEncoder`](encoders.md#baseencoder) - Abstract base class
- [`E1ElementwiseLinear`](encoders.md#e1elementwiselinear) - Diagonal scaling
- [`E2ElementwiseNonlinear`](encoders.md#e2elementwisenonlinear) - Nonlinear elementwise
- [`E3LinearlyEntangled`](encoders.md#e3linearlyentangled) - Dense linear mixing
- [`E4UndercompleteLinear`](encoders.md#e4undercompletelinear) - Dimensionality reduction
- [`E5OvercompleteLinear`](encoders.md#e5overcompletelinear) - Redundant copies
- [`E6OvercompleteMulticodes`](encoders.md#e6overcompletemulticodes) - Multiple codes per factor
- [`E7OvercompleteEntangled`](encoders.md#e7overcompleteentangled) - Dense overcomplete
- [`E8OvercompleteDisjoint`](encoders.md#e8overcompletedisjoint) - Disjoint sin/cos codes
- [`E9RandomGaussian`](encoders.md#e9randomgaussian) - Gaussian noise baseline
- [`E10RandomUniform`](encoders.md#e10randomuniform) - Uniform noise baseline

### Metrics

- [`BaseMetric`](metrics.md#basemetric) - Abstract base class
- [`MetricResult`](metrics.md#metricresult) - Result container
- [`MCCMetric`](metrics.md#mccmetric) - Mean Correlation Coefficient
- [`DCIMetric`](metrics.md#dcimetric) - Disentanglement, Completeness, Informativeness
- [`R2Metric`](metrics.md#r2metric) - Coefficient of Determination
- [`MIGMetric`](metrics.md#migmetric) - Mutual Information Gap
- [`TMEXMetric`](metrics.md#tmexmetric) - Testing for Measurement Exchangeability
- [`InfoMECMetric`](metrics.md#infomecmetric) - Modularity, Explicitness, Compactness
- [`MetricRegistry`](metrics.md#metricregistry) - Unified metric interface

### Evaluation Utilities

- [`time_block`](evaluation.md#time_block) - Time a code block
- [`memory_profiler`](evaluation.md#memory_profiler) - Profile memory usage
- [`profile_block`](evaluation.md#profile_block) - Combined timing + memory
- [`run_multi_seed_evaluation`](evaluation.md#run_multi_seed_evaluation) - Multi-seed stats

## Type Aliases

Common type hints used throughout the package:

```python
import numpy as np
from typing import Optional, Dict, Any, Callable

# Array types
ArrayLike = np.ndarray
LatentFactors = np.ndarray  # Shape: (n_samples, d)
Representations = np.ndarray  # Shape: (n_samples, m)

# Function types
EvaluationFunction = Callable[[int], Dict[str, float]]

# Optional types
Seed = Optional[int]
```

## Constants

```python
DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 1000
MIN_SAMPLES_FOR_METRICS = 50
```

## Exceptions

The package uses standard Python exceptions with descriptive messages:

```python
# Dimension mismatch
raise ValueError("Z and Z_hat must have the same number of samples")

# Invalid parameter
raise ValueError("correlation must be in [0, 1]")

# Not implemented
raise NotImplementedError("Subclasses must implement sample()")
```
