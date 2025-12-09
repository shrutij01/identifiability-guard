"""E6: Overcomplete, multiple codes per factor encoder."""

from typing import Callable, List, Optional

import numpy as np

from .base import BaseEncoder


class E6OvercompleteMulticodes(BaseEncoder):
    r"""
    E6: Overcomplete, multiple codes per factor encoder.
    
    Higher dimensionality $m > d$. Multiple learned coordinates represent the 
    same ground-truth factor, potentially nonlinearly:
        $\hat{Z}_{j1} = h_{j1}(Z_1, \ldots, Z_d)$,
    
    This encoder is overcomplete and has multiple codes per factor.
    """
    
    def __init__(
        self,
        d: int,
        m: int = None,
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
        if m is None:
            m = d + 1
        
        if m <= d:
            raise ValueError(f"E6 requires m > d, got m={m}, d={d}")
        
        super().__init__(d=d, m=m, seed=seed)
        self.nonlinear_fns = nonlinear_fns if nonlinear_fns is not None else (
            lambda args: args[:, 0] * np.sum(args[:, 1:], axis=1)
        )
        
        # Parameters to be initialized
        self.source_indices: Optional[np.ndarray] = None
        self._functions: Optional[List[Callable]] = None
        
    def _initialize_parameters(self) -> None:
        """Initialize source mapping and select nonlinear functions."""
        # Map each of m outputs to one of d inputs
        # Ensure each input is used at least once
        base_assignment = np.arange(self.d)
        extra_assignments = self._rng.choice(self.d, size=self.m - self.d, replace=True)
        self.source_indices = np.concatenate([base_assignment, extra_assignments])
        self._rng.shuffle(self.source_indices)
        
        # Set up functions
        if self._functions is not None:
            if len(self._functions) < self.d:
                raise ValueError(f"Need at least {self.d} functions, got {len(self._functions)}")
            self._functions = self._functions[:self.d]
        else:
            # Cycle through default functions
            default_fns = self._get_default_nonlinear_invertible_functions()
            self._functions = [default_fns[i % len(default_fns)] for i in range(self.d)]
        
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
        overcomplete = self.nonlinear_fns(Z)

        Z_hat = np.zeros((n, self.m))
        Z_hat[:, :self.d] = Z  # First d outputs are direct copies
        Z_hat[:, self.d:] = overcomplete.reshape((n, self.m - self.d))  # Remaining are nonlinear codes

        # apply nonlinear functions everywhere
        for i in range(self.d):
            Z_hat[:, i] = self._functions[i](Z_hat[:, i])
        
        return Z_hat
