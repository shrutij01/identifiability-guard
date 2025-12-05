"""Base class for Data Generating Processes (DGPs)."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDGP(ABC):
    """
    Abstract base class for Data Generating Processes.
    
    A DGP generates ground-truth latent factors Z ∈ R^d with specific
    statistical properties (independence, correlation, redundancy).
    
    Attributes:
        d: Number of latent factors (dimensionality).
        seed: Random seed for reproducibility.
    """
    
    def __init__(self, d: int, seed: Optional[int] = None):
        """
        Initialize the DGP.
        
        Args:
            d: Number of latent factors.
            seed: Optional random seed for reproducibility.
        """
        if d < 1:
            raise ValueError("d must be at least 1")
        self.d = d
        self.seed = seed
        self._rng = np.random.default_rng(seed)
    
    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples from the DGP.
        
        Args:
            n: Number of samples to generate.
            
        Returns:
            Z: Array of shape (n, d) containing latent factors.
        """
        pass
    
    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with a new or original seed."""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
    
    @property
    def name(self) -> str:
        """Return the name of the DGP class."""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.name}(d={self.d}, seed={self.seed})"
