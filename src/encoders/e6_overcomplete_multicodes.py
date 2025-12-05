"""E6: Overcomplete, multiple codes per factor encoder."""

from typing import Callable, List, Optional

import numpy as np

from .base import BaseEncoder


class E6OvercompleteMulticodes(BaseEncoder):
    """
    E6: Overcomplete, multiple codes per factor encoder.
    
    Higher dimensionality m > d. Multiple learned coordinates represent the 
    same ground-truth factor, potentially nonlinearly:
        Ẑ_j1 = h_j1(Z_i), Ẑ_j2 = h_j2(Z_i) for some i
    
    with enough coordinates overall to keep the map (approximately) 
    information-preserving. This encoder is overcomplete and has multiple 
    codes per factor.
    """
    
    def __init__(
        self,
        d: int,
        m: int,
        nonlinear_fns: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E6 overcomplete multicodes encoder.
        
        Args:
            d: Input dimensionality (ground-truth factors).
            m: Output dimensionality (must be > d).
            nonlinear_fns: List of invertible nonlinear functions to apply.
                If None, uses a mix of default invertible functions.
            seed: Optional random seed for reproducibility.
        """
        if m <= d:
            raise ValueError(f"E6 requires m > d, got m={m}, d={d}")
        super().__init__(d=d, m=m, seed=seed)
        self.nonlinear_fns = nonlinear_fns
        
        # Parameters to be initialized
        self.source_indices: Optional[np.ndarray] = None
        self._functions: Optional[List[Callable]] = None
    
    @staticmethod
    def _get_default_nonlinear_functions() -> List[Callable[[np.ndarray], np.ndarray]]:
        """Return a list of default invertible nonlinear functions."""
        return [
            lambda x: x,                             # identity
            lambda x: np.tanh(x),                    # tanh
            lambda x: np.sinh(x),                    # sinh
            lambda x: np.sign(x) * np.abs(x) ** 0.5, # signed sqrt
            lambda x: x ** 3,                        # cube
            lambda x: np.sign(x) * np.log1p(np.abs(x)),  # signed log1p
        ]
    
    def _initialize_parameters(self) -> None:
        """Initialize source mapping and select nonlinear functions."""
        # Map each of m outputs to one of d inputs
        # Ensure each input is used at least once
        base_assignment = np.arange(self.d)
        extra_assignments = self._rng.choice(self.d, size=self.m - self.d, replace=True)
        self.source_indices = np.concatenate([base_assignment, extra_assignments])
        self._rng.shuffle(self.source_indices)
        
        # Set up functions
        if self.nonlinear_fns is not None:
            if len(self.nonlinear_fns) < self.m:
                raise ValueError(f"Need at least {self.m} functions, got {len(self.nonlinear_fns)}")
            self._functions = self.nonlinear_fns[:self.m]
        else:
            # Cycle through default functions
            default_fns = self._get_default_nonlinear_functions()
            self._functions = [default_fns[i % len(default_fns)] for i in range(self.m)]
        
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply overcomplete nonlinear transformation with multiple codes.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, m) with multiple nonlinear codes per factor.
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        n = Z.shape[0]
        Z_hat = np.zeros((n, self.m))
        
        # Apply nonlinear function to each mapped source
        for j in range(self.m):
            source_idx = self.source_indices[j]
            Z_hat[:, j] = self._functions[j](Z[:, source_idx])
        
        return Z_hat
