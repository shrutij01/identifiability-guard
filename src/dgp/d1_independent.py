"""D1: Independent, non-redundant data generating process."""

from typing import Optional

import numpy as np

from .base import BaseDGP


class D1Independent(BaseDGP):
    """
    D1: Independent, non-redundant DGP.
    
    The coordinates Z_1, ..., Z_d are mutually independent and non-redundant:
    no Z_j is (deterministically) a function of the remaining coordinates.
    
    Each factor is sampled independently from a standard normal distribution.
    """
    
    def __init__(self, d: int, seed: Optional[int] = None):
        """
        Initialize the D1 Independent DGP.
        
        Args:
            d: Number of latent factors.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(d=d, seed=seed)
    
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples with independent factors.
        
        Args:
            n: Number of samples to generate.
            
        Returns:
            Z: Array of shape (n, d) with independent standard normal factors.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        return self._rng.standard_normal(size=(n, self.d))
