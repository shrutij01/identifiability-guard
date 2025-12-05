"""Base class for Identifiability Metrics."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np


class BaseMetric(ABC):
    """
    Abstract base class for Identifiability Metrics.
    
    A metric computes an identifiability score M(Z, Ẑ) ∈ [0, 1] from
    ground-truth factors Z ∈ R^d and learned coordinates Ẑ ∈ R^m.
    Higher scores indicate better identifiability.
    """
    
    @abstractmethod
    def compute(
        self, Z: np.ndarray, Z_hat: np.ndarray
    ) -> Union[float, Dict[str, float]]:
        """
        Compute the identifiability metric.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            Z_hat: Array of shape (n, m) containing learned coordinates.
            
        Returns:
            score: Either a single float score in [0, 1], or a dict of 
                   named scores (each in [0, 1]).
        """
        pass
    
    def __call__(
        self, Z: np.ndarray, Z_hat: np.ndarray
    ) -> Union[float, Dict[str, float]]:
        """Shorthand for compute()."""
        return self.compute(Z, Z_hat)
    
    @property
    def name(self) -> str:
        """Return the name of the metric class."""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.name}()"
