"""D2: Correlated, non-redundant data generating process."""

from typing import Optional

import numpy as np

from .base import BaseDGP


class D2Correlated(BaseDGP):
    r"""
    D2: Correlated, non-redundant DGP.
    
    The coordinates $Z_1, \ldots, Z_d$ are statistically dependent 
    (e.g., $\text{Cov}(Z_i, Z_j) \neq 0$ for some $i \neq j$), but still non-redundant:
    no $Z_j$ is a deterministic function of the others.
    
    Factors are sampled from a multivariate normal with a specified
    correlation structure.
    """
    
    def __init__(
        self,
        d: int,
        correlation: float = 0.5,
        correlation_matrix: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the D2 Correlated DGP.
        
        Args:
            d: Number of latent factors.
            correlation: Uniform pairwise correlation coefficient (used if
                correlation_matrix is not provided). Must be in (-1, 1).
            correlation_matrix: Optional (d, d) correlation matrix. If provided,
                overrides the uniform correlation parameter.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(d=d, seed=seed)
        
        if correlation_matrix is not None:
            if correlation_matrix.shape != (d, d):
                raise ValueError(f"correlation_matrix must have shape ({d}, {d})")
            self.cov_matrix = correlation_matrix
        else:
            if not -1 < correlation < 1:
                raise ValueError("correlation must be in (-1, 1)")
            # Create uniform correlation matrix
            self.cov_matrix = np.full((d, d), correlation)
            np.fill_diagonal(self.cov_matrix, 1.0)
        
        # Verify positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if np.any(eigvals < -1e-10):
            raise ValueError("Covariance matrix must be positive semi-definite")
        
        self.correlation = correlation
    
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples with correlated factors.
        
        Args:
            n: Number of samples to generate.
            
        Returns:
            Z: Array of shape (n, d) with correlated normal factors.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        mean = np.zeros(self.d)
        return self._rng.multivariate_normal(mean, self.cov_matrix, size=n)
