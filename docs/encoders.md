# Encoder Mixings

Encoders transform ground-truth latents $Z$ to learned representations $\hat{Z}$ with controlled properties.

## Overview

All encoders inherit from `BaseEncoder` and provide a `.encode(Z)` interface.

| Encoder | Type | Dimensionality | Description |
|---------|------|----------------|-------------|
| E1 | Elementwise Linear | $m = d$ | Diagonal scaling with permutation |
| E2 | Elementwise Nonlinear | $m = d$ | Invertible nonlinear functions |
| E3 | Linearly Entangled | $m = d$ | Dense linear mixing |
| E4 | Undercomplete Linear | $m < d$ | Dimensionality reduction |
| E5 | Overcomplete Linear | $m > d$ | Redundant linear copies |
| E6 | Overcomplete Multicodes | $m > d$ | Multiple codes per factor |
| E7 | Overcomplete Entangled | $m > d$ | Dense mixing, rank-d |
| E8 | Overcomplete Disjoint | $m > d$ | Disjoint sin/cos codes |
| E9 | Random Gaussian | $m = d$ | Null baseline (noise) |
| E10 | Random Uniform | $m = d$ | Null baseline (noise) |

## E1: Elementwise Linear

Applies diagonal scaling with permutation.

```python
from identifiability_guard.encoders import E1ElementwiseLinear

encoder = E1ElementwiseLinear(d=5, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (n_samples, 5)
```

**Mathematical Definition:**
$$\hat{Z}_j = a_j \cdot Z_{\pi(j)}$$

where $a_j$ are random scales and $\pi$ is a random permutation.

**Parameters:**
- `d` (int): Dimensionality (input and output)
- `seed` (int, optional): Random seed

## E2: Elementwise Nonlinear

Applies invertible nonlinear transformations with controllable strength.

```python
from identifiability_guard.encoders import E2ElementwiseNonlinear

encoder = E2ElementwiseNonlinear(d=5, nonlinearity_strength=0.5, seed=42)
Z_hat = encoder.encode(Z)
```

**Parameters:**
- `d` (int): Dimensionality
- `nonlinearity_strength` (float): Strength in [0, 1] (0=linear, 1=fully nonlinear)
- `seed` (int, optional): Random seed

**Mathematical Definition:**
$$\hat{Z}_j = h_j(Z_{\pi(j)})$$

where $h_j$ are invertible nonlinear functions (e.g., tanh, leaky ReLU).

## E3: Linearly Entangled

Dense linear mixing with full-rank matrix.

```python
from identifiability_guard.encoders import E3LinearlyEntangled

encoder = E3LinearlyEntangled(d=5, seed=42)
Z_hat = encoder.encode(Z)
```

**Mathematical Definition:**
$$\hat{Z} = A \cdot Z$$

where $A \in \mathbb{R}^{d \times d}$ is a full-rank random matrix.

**Parameters:**
- `d` (int): Dimensionality
- `seed` (int, optional): Random seed

## E4: Undercomplete Linear

Dimensionality reduction via elementwise scaling.

```python
from identifiability_guard.encoders import E4UndercompleteLinear

encoder = E4UndercompleteLinear(d=5, m=3, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (n_samples, 3)
```

**Parameters:**
- `d` (int): Input dimensionality
- `m` (int): Output dimensionality (must satisfy $m < d$)
- `seed` (int, optional): Random seed

## E5: Overcomplete Linear

Redundant scaled copies of latent factors.

```python
from identifiability_guard.encoders import E5OvercompleteLinear

encoder = E5OvercompleteLinear(d=5, m=10, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (n_samples, 10)
```

**Parameters:**
- `d` (int): Input dimensionality
- `m` (int): Output dimensionality (must satisfy $m > d$)
- `seed` (int, optional): Random seed

## E6: Overcomplete Multicodes

Multiple nonlinear codes per latent factor.

```python
from identifiability_guard.encoders import E6OvercompleteMulticodes

encoder = E6OvercompleteMulticodes(d=5, codes_per_factor=3, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (n_samples, 15)
```

**Parameters:**
- `d` (int): Number of latent factors
- `codes_per_factor` (int): Number of codes per factor ($m = d \times \text{codes\_per\_factor}$)
- `seed` (int, optional): Random seed

## E7: Overcomplete Entangled

Dense linear mixing with controlled condition number.

```python
from identifiability_guard.encoders import E7OvercompleteEntangled

encoder = E7OvercompleteEntangled(d=5, m=10, condition_number=10.0, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (n_samples, 10)
```

**Mathematical Definition:**
$$\hat{Z} = A \cdot Z$$

where $A \in \mathbb{R}^{m \times d}$ has rank $d$ and controlled singular values.

**Parameters:**
- `d` (int): Input dimensionality
- `m` (int): Output dimensionality ($m > d$)
- `condition_number` (float): Ratio of largest to smallest singular value
- `seed` (int, optional): Random seed

## E8: Overcomplete Disjoint

Disjoint sin/cos codes per factor (Fourier-like encoding).

```python
from identifiability_guard.encoders import E8OvercompleteDisjoint

encoder = E8OvercompleteDisjoint(d=5, codes_per_factor=2, seed=42)
Z_hat = encoder.encode(Z)  # Shape: (n_samples, 10)
```

**Mathematical Definition:**
For each factor $Z_i$, creates codes: $[\sin(\omega_i Z_i), \cos(\omega_i Z_i)]$

**Parameters:**
- `d` (int): Number of latent factors
- `codes_per_factor` (int): Codes per factor
- `seed` (int, optional): Random seed

## E9/E10: Random Baselines

Null encoders that output noise (for sanity checks).

```python
from identifiability_guard.encoders import E9RandomGaussian, E10RandomUniform

# Gaussian noise
encoder_gaussian = E9RandomGaussian(d=5, seed=42)
Z_hat_noise = encoder_gaussian.encode(Z)  # Random Gaussian, ignores Z

# Uniform noise
encoder_uniform = E10RandomUniform(d=5, seed=42)
Z_hat_uniform = encoder_uniform.encode(Z)  # Random uniform, ignores Z
```

**Use Case:**
Verify that metrics correctly assign ~0 scores to random representations.

## Base Class API

All encoders implement the `BaseEncoder` interface:

```python
class BaseEncoder:
    def __init__(self, d: int, seed: Optional[int] = None):
        """
        Args:
            d: Input dimensionality
            seed: Random seed for reproducibility
        """
        pass

    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Encode latent factors.

        Args:
            Z: Ground-truth factors of shape (n_samples, d)

        Returns:
            Encoded representations of shape (n_samples, m)
        """
        pass
```

## Usage Tips

**Combining DGPs and Encoders:**
```python
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E3LinearlyEntangled

dgp = D1Independent(d=5, seed=42)
encoder = E3LinearlyEntangled(d=5, seed=123)

Z = dgp.sample(1000)
Z_hat = encoder.encode(Z)
```

**Parameter Sweeps:**
```python
# Vary nonlinearity strength
strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
for s in strengths:
    encoder = E2ElementwiseNonlinear(d=5, nonlinearity_strength=s, seed=42)
    Z_hat = encoder.encode(Z)
    # Evaluate metrics...
```

**Dimensionality Control:**
```python
# Undercomplete
encoder_under = E4UndercompleteLinear(d=10, m=5, seed=42)  # 10 -> 5

# Exact
encoder_exact = E1ElementwiseLinear(d=10, seed=42)  # 10 -> 10

# Overcomplete
encoder_over = E5OvercompleteLinear(d=10, m=20, seed=42)  # 10 -> 20
```

**Sanity Checks:**
```python
# Random encoders should give near-zero metric scores
encoder_random = E9RandomGaussian(d=5, seed=42)
Z_hat_random = encoder_random.encode(Z)

from identifiability_guard.metrics import MCC
mcc = MCC()
score = mcc.compute(Z, Z_hat_random)
assert score.primary_score < 0.1  # Should be near zero
```
