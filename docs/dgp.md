# Data Generating Processes (DGPs)

DGPs generate ground-truth latent factors $Z$ with controlled statistical properties.

## Overview

All DGPs inherit from `BaseDGP` and provide a unified `.sample(n_samples)` interface.

| DGP | Name | Statistical Properties |
|-----|------|----------------------|
| D1 | Independent | Mutually independent, non-redundant |
| D2 | Correlated | Statistically dependent, non-redundant |
| D3 | Single-redundant | One factor is a function of another |
| D4 | Multi-redundant | One factor depends on multiple others |

## D1: Independent Factors

Generates mutually independent latent factors from a Gaussian distribution.

```python
from identifiability_guard.dgp import D1Independent

dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)  # Shape: (1000, 5)
```

**Parameters:**
- `d` (int): Number of latent factors
- `seed` (int, optional): Random seed for reproducibility

**Mathematical Definition:**
$$Z_i \sim \mathcal{N}(0, 1), \quad Z_i \perp Z_j \text{ for } i \neq j$$

## D2: Correlated Factors

Generates correlated factors with controlled correlation strength.

```python
from identifiability_guard.dgp import D2Correlated

dgp = D2Correlated(d=5, correlation=0.5, seed=42)
Z = dgp.sample(1000)
```

**Parameters:**
- `d` (int): Number of latent factors
- `correlation` (float): Target correlation coefficient in [0, 1]
- `seed` (int, optional): Random seed

**Mathematical Definition:**
Factors are sampled from a multivariate Gaussian with off-diagonal correlation $\rho$.

## D3: Single-Redundant Factors

One factor is a deterministic function of another single factor.

```python
from identifiability_guard.dgp import D3SingleRedundant

dgp = D3SingleRedundant(d=5, r=1, noise_std=0.1, seed=42)
Z = dgp.sample(1000)
```

**Parameters:**
- `d` (int): Number of latent factors
- `r` (int): Number of redundant factors (default: 1)
- `noise_std` (float): Noise level in redundant factors (default: 0.0)
- `seed` (int, optional): Random seed

**Mathematical Definition:**
$$Z_{\text{redundant}} = Z_{\text{source}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

## D4: Multi-Redundant Factors

One factor depends on multiple other factors.

```python
from identifiability_guard.dgp import D4MultiRedundant

dgp = D4MultiRedundant(d=5, r=1, noise_std=0.1, seed=42)
Z = dgp.sample(1000)
```

**Parameters:**
- `d` (int): Total number of factors
- `r` (int): Number of redundant factors
- `noise_std` (float): Noise level (default: 0.0)
- `seed` (int, optional): Random seed

**Mathematical Definition:**
$$Z_{\text{redundant}} = f(Z_{\text{sources}}) + \epsilon$$

where $f$ combines multiple source factors.

## Base Class API

All DGPs implement the `BaseDGP` interface:

```python
class BaseDGP:
    def __init__(self, d: int, seed: Optional[int] = None):
        """
        Args:
            d: Number of latent factors
            seed: Random seed for reproducibility
        """
        pass

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, d)
        """
        pass
```

## Usage Tips

**Reproducibility:**
```python
# Always set seed for reproducible experiments
dgp = D1Independent(d=5, seed=42)
Z1 = dgp.sample(1000)
Z2 = dgp.sample(1000)  # Different samples

# Create new instance for identical samples
dgp_copy = D1Independent(d=5, seed=42)
Z3 = dgp_copy.sample(1000)  # Identical to Z1
```

**Batch Generation:**
```python
# Generate large batches efficiently
dgp = D1Independent(d=10, seed=42)
Z_large = dgp.sample(100_000)  # Efficient vectorized generation
```

**Validation:**
```python
import numpy as np

# Verify independence (D1)
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(10000)
corr_matrix = np.corrcoef(Z.T)
assert np.allclose(corr_matrix, np.eye(5), atol=0.1)

# Verify correlation (D2)
dgp = D2Correlated(d=5, correlation=0.7, seed=42)
Z = dgp.sample(10000)
mean_offdiag = np.mean(np.abs(np.corrcoef(Z.T) - np.eye(5)))
assert mean_offdiag > 0.5  # Should have substantial correlation
```
