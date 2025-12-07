"""E5: Overcomplete, elementwise linear encoder."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E5OvercompleteLinear(BaseEncoder):
    r"""
    E5: Overcomplete, elementwise linear encoder.
    
    Higher dimensionality $m > d$. Some ground-truth factors appear as multiple 
    scaled coordinates in $\hat{Z}$:
        $\hat{Z}_{j1} = a_{j1} * Z_i$, $\hat{Z}_{j2} = a_{j2} * Z_i$,
    
    possibly with additional unused or noisy dimensions.
    """
    
    def __init__(
        self,
        d: int,
        m: int,
        scale_range: tuple = (0.5, 2.0),
        seed: Optional[int] = None,
    ):
        """
        Initialize the E5 overcomplete linear encoder.
        
        Args:
            d: Input dimensionality (ground-truth factors).
            m: Output dimensionality (must be > d).
            scale_range: Range (min, max) for random scaling factors.
            seed: Optional random seed for reproducibility.
        """
        if m <= d:
            raise ValueError(f"E5 requires m > d, got m={m}, d={d}")
        super().__init__(d=d, m=m, seed=seed)
        self.scale_range = scale_range
        
        # Parameters to be initialized
        self.scales: Optional[np.ndarray] = None
        self.source_indices: Optional[np.ndarray] = None
    
    def _initialize_parameters(self) -> None:
        """Initialize scaling factors and source mapping."""
        # Map each of m outputs to one of d inputs
        # First ensure each input is used at least once, then randomly assign rest
        base_assignment = np.arange(self.d)
        extra_assignments = self._rng.choice(self.d, size=self.m - self.d, replace=True)
        self.source_indices = np.concatenate([base_assignment, extra_assignments])
        self._rng.shuffle(self.source_indices)
        
        # Random scaling factors (non-zero)
        self.scales = self._rng.uniform(
            self.scale_range[0], self.scale_range[1], size=self.m
        )
        
        # Random signs
        signs = self._rng.choice([-1, 1], size=self.m)
        self.scales = self.scales * signs
        
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply overcomplete elementwise linear transformation.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, m) with m > d (redundant) coordinates.
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        # Map each output to its source input and scale
        Z_hat = Z[:, self.source_indices] * self.scales
        return Z_hat
