"""E1: Exact, elementwise linear encoder."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E1ElementwiseLinear(BaseEncoder):
    """
    E1: Exact, elementwise linear encoder.
    
    Same dimensionality m = d. Each learned coordinate is a scaled version 
    of exactly one ground-truth coordinate, up to permutation:
        Ẑ_j = a_j * Z_π(j), where a_j ≠ 0
    
    This map is information-preserving and elementwise linear.
    """
    
    def __init__(
        self,
        d: int,
        scale_range: tuple = (0.5, 2.0),
        permute: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E1 elementwise linear encoder.
        
        Args:
            d: Dimensionality (same for input and output).
            scale_range: Range (min, max) for random scaling factors.
            permute: Whether to apply a random permutation.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(d=d, m=d, seed=seed)
        self.scale_range = scale_range
        self.permute = permute
        
        # Parameters to be initialized
        self.scales: Optional[np.ndarray] = None
        self.permutation: Optional[np.ndarray] = None
    
    def _initialize_parameters(self) -> None:
        """Initialize scaling factors and permutation."""
        # Random scaling factors (non-zero)
        self.scales = self._rng.uniform(
            self.scale_range[0], self.scale_range[1], size=self.d
        )
        # Random signs
        signs = self._rng.choice([-1, 1], size=self.d)
        self.scales = self.scales * signs
        
        # Random permutation
        if self.permute:
            self.permutation = self._rng.permutation(self.d)
        else:
            self.permutation = np.arange(self.d)
        
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply elementwise linear transformation with permutation.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, d) with Ẑ_j = a_j * Z_π(j).
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        # Apply permutation then scaling
        Z_permuted = Z[:, self.permutation]
        Z_hat = Z_permuted * self.scales
        return Z_hat
