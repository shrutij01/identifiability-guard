"""E7: Overcomplete, linearly entangled encoder."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E7OvercompleteEntangled(BaseEncoder):
    r"""
    E7: Overcomplete, linearly entangled encoder.
    
    Higher dimensionality $m > d$. The encoder is a dense linear map:
        $\hat{Z} = A Z$,
    
    where $A \in \mathbb{R}^{m \times d}$ is rank-d (full rank) with dense rows,
    so each coordinate of $\hat{Z}$ mixes multiple factors. Unlike E3 which is square
    and invertible, E7 is overcomplete and loses information through the non-invertible
    mapping.
    """
    
    def __init__(
        self,
        d: int,
        m: int = None,
        condition_number: float = 10.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E7 overcomplete entangled encoder.
        
        Args:
            d: Input dimensionality (ground-truth factors).
            m: Output dimensionality (must be > d). Defaults to 2*d if not provided.
            condition_number: Target condition number for the mixing matrix.
                Higher values mean more ill-conditioned (but still full rank).
            seed: Optional random seed for reproducibility.
        """
        if m is None:
            m = 2 * d
        
        if m <= d:
            raise ValueError(f"E7 requires m > d, got m={m}, d={d}")
        
        super().__init__(d=d, m=m, seed=seed)
        self.condition_number = condition_number
        
        # Parameters to be initialized
        self.mixing_matrix: Optional[np.ndarray] = None
    
    def _initialize_parameters(self) -> None:
        """Initialize the rank-d mixing matrix with specified condition number."""
        # Generate a random matrix of shape (m, d)
        A = self._rng.standard_normal(size=(self.m, self.d))
        
        # Use SVD to control the condition number
        # For overcomplete matrix (m > d), we have U (m×m), S (m×d as diagonal), Vt (d×d)
        # We need to construct A with controlled singular values
        
        # Generate orthonormal bases
        U, _ = np.linalg.qr(self._rng.standard_normal(size=(self.m, self.m)))
        V, _ = np.linalg.qr(self._rng.standard_normal(size=(self.d, self.d)))
        
        # Create singular values with desired condition number
        # The matrix has d singular values (since rank is d)
        s = np.linspace(1.0, 1.0 / self.condition_number, self.d)
        
        # Construct the matrix: A = U @ Sigma @ V^T
        # where Sigma is (m, d) with first d rows being diag(s) and rest zeros
        Sigma = np.zeros((self.m, self.d))
        np.fill_diagonal(Sigma, s)
        
        self.mixing_matrix = U @ Sigma @ V.T
        
        self._initialized = True
    
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply overcomplete dense linear mixing transformation.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            
        Returns:
            Z_hat: Array of shape (n, m) with m > d overcomplete coordinates,
                   where each coordinate is a linear combination of all factors.
        """
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} factors, got {Z.shape[1]}")
        
        if not self._initialized:
            self._initialize_parameters()
        
        # Apply linear transformation: Z_hat = Z @ A.T
        Z_hat = Z @ self.mixing_matrix.T
        return Z_hat
