"""Base class for Encoder Mixings."""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List

import numpy as np


class BaseEncoder(ABC):
    r"""
    Abstract base class for Encoder Mixings.
    
    An encoder transforms ground-truth latent factors $Z \in \mathbb{R}^d$ to 
    learned representations $\hat{Z} \in \mathbb{R}^m$.
    
    Attributes:
        d: Input dimensionality (number of ground-truth factors).
        m: Output dimensionality (number of learned coordinates).
        seed: Random seed for reproducibility.
    """
    
    def __init__(self, d: int, m: Optional[int] = None, seed: Optional[int] = None):
        """
        Initialize the encoder.
        
        Args:
            d: Input dimensionality (ground-truth factors).
            m: Output dimensionality (learned coordinates). Defaults to d.
            seed: Optional random seed for reproducibility.
        """
        if d < 1:
            raise ValueError("d must be at least 1")
        self.d = d
        self.m = m if m is not None else d
        if self.m < 1:
            raise ValueError("m must be at least 1")
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._initialized = False
    
    @abstractmethod
    def _initialize_parameters(self) -> None:
        """Initialize encoder parameters (scaling factors, permutations, etc.)."""
        pass
    
    @abstractmethod
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Transform latent factors to learned representations.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, m) containing learned coordinates.
        """
        pass
    
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        """Shorthand for encode()."""
        return self.encode(Z)
    
    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with a new or original seed."""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self._initialized = False

    @staticmethod
    def _get_default_nonlinear_functions() -> List[Callable[[np.ndarray], np.ndarray]]:
        """Return a list of default invertible nonlinear functions."""
        return [
            lambda x: np.tanh(x),                       # tanh (invertible)
            lambda x: np.sinh(x),                       # sinh (invertible)
            lambda x: np.sign(x) * np.abs(x) ** 0.5,    # signed sqrt (invertible)
            lambda x: x ** 3,                           # cube (invertible)
            lambda x: x ** 5,                           # fifth power (invertible)
            lambda x: np.exp(x) - 1,                    # exp - 1 (invertible)
            lambda x: np.sign(x) * np.log1p(np.abs(x)), # signed log1p
        ]
    
    @property
    def name(self) -> str:
        """Return the name of the encoder class."""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.name}(d={self.d}, m={self.m}, seed={self.seed})"
