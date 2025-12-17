"""E8: Overcomplete, nonlinear disjoint subsets encoder."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E8OvercompleteDisjoint(BaseEncoder):
    r"""
    E8: Overcomplete, nonlinear disjoint subsets encoder.
    
    Higher dimensionality $m > d$, where $m = k \cdot d$ for some integer $k \geq 2$.
    Each ground-truth factor $Z_i$ is encoded into $k$ disjoint coordinates using
    nonlinear transformations (default: sin/cos pairs).
    
    For the default sin/cos encoding with k=2:
        $\hat{Z}_{2i} = \sin(Z_i)$
        $\hat{Z}_{2i+1} = \cos(Z_i)$
    
    This allows perfect reconstruction via $Z_i = \text{atan2}(\sin(Z_i), \cos(Z_i))$.
    The subsets are disjoint: each learned coordinate depends on exactly one factor.
    """
    
    def __init__(
        self,
        d: int,
        codes_per_factor: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E8 overcomplete disjoint encoder.
        
        Args:
            d: Input dimensionality (ground-truth factors).
            codes_per_factor: Number of codes per factor (k). Defaults to 2.
                Output dimensionality will be m = k * d.
            seed: Optional random seed for reproducibility.
        """
        if codes_per_factor < 2:
            raise ValueError(f"codes_per_factor must be at least 2, got {codes_per_factor}")
        
        m = codes_per_factor * d
        super().__init__(d=d, m=m, seed=seed)
        self.codes_per_factor = codes_per_factor
        
        # Parameters to be initialized
        self.permutation: Optional[np.ndarray] = None
    
    def _initialize_parameters(self) -> None:
        """Initialize random permutation of factors."""
        # Apply a random permutation to factors for variety
        self.permutation = self._rng.permutation(self.d)
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply overcomplete disjoint nonlinear transformation.
        
        For codes_per_factor=2 (default), uses sin/cos encoding:
            Ẑ[2i] = sin(Z[π(i)])
            Ẑ[2i+1] = cos(Z[π(i)])
        
        For codes_per_factor > 2, uses a mix of nonlinear functions.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, m) where m = codes_per_factor * d,
                   with disjoint subsets of codes per factor.
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        n = Z.shape[0]
        Z_hat = np.zeros((n, self.m))
        
        # Apply permutation
        Z_permuted = Z[:, self.permutation]
        
        # Default nonlinear functions for multiple codes per factor
        if self.codes_per_factor == 2:
            # Use sin/cos for perfect reconstruction
            for i in range(self.d):
                Z_hat[:, 2*i] = np.sin(Z_permuted[:, i])
                Z_hat[:, 2*i+1] = np.cos(Z_permuted[:, i])
        else:
            # Use a variety of nonlinear functions
            functions = [
                np.sin,
                np.cos,
                lambda x: np.tanh(x),
                lambda x: np.sign(x) * np.sqrt(np.abs(x)),
                lambda x: x**3 / (1 + np.abs(x**2)),  # Bounded cubic
                lambda x: np.sinh(x) / (1 + np.abs(x)),  # Bounded sinh
            ]
            
            for i in range(self.d):
                for k in range(self.codes_per_factor):
                    fn = functions[k % len(functions)]
                    Z_hat[:, i * self.codes_per_factor + k] = fn(Z_permuted[:, i])
        
        return Z_hat
    
    def decode(self, Z_hat: np.ndarray) -> np.ndarray:
        """
        Attempt to reconstruct original factors from encoded representation.
        
        For sin/cos encoding (codes_per_factor=2), uses atan2 for perfect reconstruction.
        For other encodings, this is not generally possible.
        
        Args:
            Z_hat: Array of shape (n, m) containing encoded coordinates.
            
        Returns:
            Z_reconstructed: Array of shape (n, d) with reconstructed factors.
        """
        if Z_hat.shape[1] != self.m:
            raise ValueError(f"Expected input with {self.m} codes, got {Z_hat.shape[1]}")
        
        if not self._initialized:
            raise RuntimeError("Encoder not initialized. Call encode() first.")
        
        n = Z_hat.shape[0]
        Z_reconstructed = np.zeros((n, self.d))
        
        if self.codes_per_factor == 2:
            # Use atan2 to recover angle from sin/cos
            for i in range(self.d):
                sin_val = Z_hat[:, 2*i]
                cos_val = Z_hat[:, 2*i+1]
                Z_reconstructed[:, i] = np.arctan2(sin_val, cos_val)
        else:
            # For other encodings, use the first code as a rough approximation
            # (not guaranteed to be accurate)
            for i in range(self.d):
                Z_reconstructed[:, i] = Z_hat[:, i * self.codes_per_factor]
        
        # Undo permutation
        Z_unpermuted = np.zeros_like(Z_reconstructed)
        Z_unpermuted[:, self.permutation] = Z_reconstructed
        
        return Z_unpermuted
