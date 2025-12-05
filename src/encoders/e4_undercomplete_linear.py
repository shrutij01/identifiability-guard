"""E4: Undercomplete, elementwise linear encoder."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E4UndercompleteLInear(BaseEncoder):
    """
    E4: Undercomplete, elementwise linear encoder.
    
    Lower dimensionality m < d. Each learned coordinate is a scaled version 
    of a distinct ground-truth factor:
        Ẑ_j = a_j * Z_i(j), j = 1, ..., m
    
    with all i(j) distinct and a_j ≠ 0. This encoder is elementwise but lossy:
    some ground-truth factors have no representation in Ẑ.
    """
    
    def __init__(
        self,
        d: int,
        m: int,
        scale_range: tuple = (0.5, 2.0),
        seed: Optional[int] = None,
    ):
        """
        Initialize the E4 undercomplete linear encoder.
        
        Args:
            d: Input dimensionality (ground-truth factors).
            m: Output dimensionality (must be < d).
            scale_range: Range (min, max) for random scaling factors.
            seed: Optional random seed for reproducibility.
        """
        if m >= d:
            raise ValueError(f"E4 requires m < d, got m={m}, d={d}")
        super().__init__(d=d, m=m, seed=seed)
        self.scale_range = scale_range
        
        # Parameters to be initialized
        self.scales: Optional[np.ndarray] = None
        self.selected_indices: Optional[np.ndarray] = None
    
    def _initialize_parameters(self) -> None:
        """Initialize scaling factors and select which factors to keep."""
        # Randomly select m distinct factors from d
        self.selected_indices = self._rng.choice(self.d, size=self.m, replace=False)
        self.selected_indices.sort()  # Keep order for reproducibility
        
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
        Apply undercomplete elementwise linear transformation.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, m) with m < d factors preserved.
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        # Select subset of factors and scale
        Z_selected = Z[:, self.selected_indices]
        Z_hat = Z_selected * self.scales
        return Z_hat
