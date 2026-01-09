"""E2: Exact, elementwise invertible nonlinear encoder."""

from typing import Callable, List, Optional

import numpy as np

from .base import BaseEncoder


class E2ElementwiseNonlinear(BaseEncoder):
    r"""
    E2: Exact, elementwise invertible nonlinear encoder.
    
    Same dimensionality $m = d$. Each learned coordinate is an invertible 
    nonlinear function of exactly one ground-truth coordinate:
        $\hat{Z}_j = h_j(Z_{\pi(j)})$,
    
    where each $h_j$ is scalar and invertible, and $\pi$ is a permutation.
    This map is information-preserving and elementwise nonlinear.
    """
    
    def __init__(
        self,
        d: int,
        nonlinear_fns: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        permute: bool = True,
        nonlinearity_strength: float = 1.0,
        scale_range: tuple = (0.5, 2.0),
        seed: Optional[int] = None,
    ):
        """
        Initialize the E2 elementwise nonlinear encoder.
        
        Args:
            d: Dimensionality (same for input and output).
            nonlinear_fns: List of d invertible nonlinear functions. 
                If None, uses a mix of default invertible functions.
            permute: Whether to apply a random permutation.
            nonlinearity_strength: Strength of nonlinearity in [0, 1].
                0 = identity (linear), 1 = fully nonlinear.
                Interpolates as: f(x) = (1-α)*x + α*h(x).
            seed: Optional random seed for reproducibility.
        """
        super().__init__(d=d, m=d, seed=seed)
        self.permute = permute
        self.nonlinear_fns = nonlinear_fns
        
        if not 0.0 <= nonlinearity_strength <= 1.0:
            raise ValueError(f"nonlinearity_strength must be in [0, 1], got {nonlinearity_strength}")
        self.nonlinearity_strength = nonlinearity_strength
        
        # Parameters to be initialized
        self.permutation: Optional[np.ndarray] = None
        self._functions: Optional[List[Callable]] = None

        # To test different non-linearity strengths, when strength < 1.0
        self.scale_range = scale_range
        self.scales: Optional[np.ndarray] = None
        
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
            default_fns = self._get_default_nonlinear_invertible_functions()
            self._functions = [default_fns[i % len(default_fns)] for i in range(self.d)]
        
        # Random scaling factors (non-zero)
        self.scales = self._rng.uniform(
            self.scale_range[0], self.scale_range[1], size=self.d
        )

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
        
        # Apply permutation then nonlinear functions
        Z_permuted = Z[:, self.permutation]
        Z_hat = np.zeros_like(Z_permuted)
        
        # Apply nonlinearity with strength parameter
        # f(x) = (1 - α) * x + α * h(x), where α = nonlinearity_strength
        alpha = self.nonlinearity_strength
        for j in range(self.d):
            h_x = self._functions[j](Z_permuted[:, j])
            Z_hat[:, j] = (1 - alpha) * (self.scales[j] * Z_permuted[:, j]) + alpha * h_x
        
        return Z_hat
