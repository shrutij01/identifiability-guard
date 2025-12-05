"""E3: Exact, linearly entangled encoder."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E3LinearlyEntangled(BaseEncoder):
    """
    E3: Exact, linearly entangled encoder.
    
    Same dimensionality m = d. The encoder is a dense invertible linear map:
        Ẑ = A @ Z
    
    where A ∈ R^{d×d} is invertible and has at least one row with two or 
    more nonzero entries, so each coordinate of Ẑ mixes several factors.
    """
    
    def __init__(
        self,
        d: int,
        condition_number: float = 10.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E3 linearly entangled encoder.
        
        Args:
            d: Dimensionality (same for input and output).
            condition_number: Target condition number for the mixing matrix.
                Higher values mean more ill-conditioned (but still invertible).
            seed: Optional random seed for reproducibility.
        """
        super().__init__(d=d, m=d, seed=seed)
        self.condition_number = condition_number
        
        # Parameters to be initialized
        self.mixing_matrix: Optional[np.ndarray] = None
    
    def _initialize_parameters(self) -> None:
        """Initialize the invertible mixing matrix."""
        # Generate a random matrix
        A = self._rng.standard_normal(size=(self.d, self.d))
        
        # Ensure it's invertible with desired conditioning via SVD
        U, s, Vt = np.linalg.svd(A)
        
        # Scale singular values to achieve target condition number
        s_new = np.linspace(1.0, 1.0 / self.condition_number, self.d)
        
        # Reconstruct with new singular values
        self.mixing_matrix = U @ np.diag(s_new) @ Vt
        
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply dense linear mixing transformation.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, d) with Ẑ = Z @ A.T (dense linear mixing).
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        # Apply linear transformation: Z_hat = Z @ A.T (equivalent to A @ Z.T).T
        Z_hat = Z @ self.mixing_matrix.T
        return Z_hat
