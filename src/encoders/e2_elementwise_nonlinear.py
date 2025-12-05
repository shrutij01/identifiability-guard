"""E2: Exact, elementwise invertible nonlinear encoder."""

from typing import Callable, List, Optional

import numpy as np

from .base import BaseEncoder


class E2ElementwiseNonlinear(BaseEncoder):
    """
    E2: Exact, elementwise invertible nonlinear encoder.
    
    Same dimensionality m = d. Each learned coordinate is an invertible 
    nonlinear function of exactly one ground-truth coordinate:
        Ẑ_j = h_j(Z_π(j))
    
    where each h_j is scalar and invertible, and π is a permutation.
    This map is information-preserving and elementwise nonlinear.
    """
    
    def __init__(
        self,
        d: int,
        nonlinear_fns: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        permute: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E2 elementwise nonlinear encoder.
        
        Args:
            d: Dimensionality (same for input and output).
            nonlinear_fns: List of d invertible nonlinear functions. 
                If None, uses a mix of default invertible functions.
            permute: Whether to apply a random permutation.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(d=d, m=d, seed=seed)
        self.permute = permute
        self.nonlinear_fns = nonlinear_fns
        
        # Parameters to be initialized
        self.permutation: Optional[np.ndarray] = None
        self._functions: Optional[List[Callable]] = None
    
    @staticmethod
    def _get_default_nonlinear_functions() -> List[Callable[[np.ndarray], np.ndarray]]:
        """Return a list of default invertible nonlinear functions."""
        return [
            lambda x: np.tanh(x),                    # tanh (invertible)
            lambda x: np.sinh(x),                    # sinh (invertible)
            lambda x: np.sign(x) * np.abs(x) ** 0.5, # signed sqrt (invertible)
            lambda x: x ** 3,                        # cube (invertible)
            lambda x: np.sign(x) * np.log1p(np.abs(x)),  # signed log1p
        ]
    
    def _initialize_parameters(self) -> None:
        """Initialize permutation and select nonlinear functions."""
        # Random permutation
        if self.permute:
            self.permutation = self._rng.permutation(self.d)
        else:
            self.permutation = np.arange(self.d)
        
        # Set up functions
        if self.nonlinear_fns is not None:
            if len(self.nonlinear_fns) != self.d:
                raise ValueError(f"Expected {self.d} functions, got {len(self.nonlinear_fns)}")
            self._functions = self.nonlinear_fns
        else:
            # Cycle through default functions
            default_fns = self._get_default_nonlinear_functions()
            self._functions = [default_fns[i % len(default_fns)] for i in range(self.d)]
        
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply elementwise nonlinear transformation with permutation.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, d) with Ẑ_j = h_j(Z_π(j)).
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        # Apply permutation
        Z_permuted = Z[:, self.permutation]
        
        # Apply nonlinear functions elementwise
        Z_hat = np.zeros_like(Z_permuted)
        for j in range(self.d):
            Z_hat[:, j] = self._functions[j](Z_permuted[:, j])
        
        return Z_hat
