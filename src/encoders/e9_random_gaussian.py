"""E9: Random Gaussian encoder (baseline for comparison)."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E9RandomGaussian(BaseEncoder):
    r"""
    E9: Random Gaussian encoder (baseline/null encoder).
    
    This encoder ignores the input entirely and outputs random Gaussian values.
    It serves as a baseline/null encoder to verify that identifiability metrics
    correctly identify non-informative representations.
    
    Output: $\hat{Z}_j \sim \mathcal{N}(\mu, \sigma^2)$ for all $j$, independent of $Z$.
    
    Expected behavior:
    - All identifiability metrics should yield low/random scores
    - DCI disentanglement and completeness should be near 0
    - MCC should be near 0 (random correlation)
    - R² should be near 0 (no predictive power)
    
    This encoder is useful for:
    - Validating that metrics work correctly
    - Establishing a lower bound for metric performance
    - Sanity checking evaluation pipelines
    """
    
    def __init__(
        self,
        d: int,
        m: int = None,
        mean: float = 0.0,
        std: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E9 random Gaussian encoder.
        
        Args:
            d: Input dimensionality (ground-truth factors) - ignored during encoding.
            m: Output dimensionality. Defaults to d if not provided.
            mean: Mean of the Gaussian distribution (default: 0.0).
            std: Standard deviation of the Gaussian distribution (default: 1.0).
            seed: Optional random seed for reproducibility.
        """
        if m is None:
            m = d
        
        super().__init__(d=d, m=m, seed=seed)
        self.mean = mean
        self.std = std
    
    @property
    def name(self) -> str:
        """Return a descriptive name for this encoder."""
        return f"E9: Random Gaussian (μ={self.mean}, σ={self.std})"

    @property
    def display_name(self) -> str:
        """Short display label for visuals."""
        return "E9RandomGaussian"
    
    def _initialize_parameters(self) -> None:
        """No parameters to initialize for random encoder."""
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Generate random Gaussian output ignoring input Z.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors (ignored).
            
        Returns:
            Z_hat: Array of shape (n, m) with random Gaussian values.
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} features, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        n_samples = Z.shape[0]
        
        # Generate random Gaussian output (completely ignores Z)
        Z_hat = self._rng.normal(loc=self.mean, scale=self.std, size=(n_samples, self.m))
        
        return Z_hat
