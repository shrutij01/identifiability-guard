"""D4: Multi-factor redundant data generating process."""

from typing import Callable, Optional

import numpy as np

from .base import BaseDGP


class D4MultiRedundant(BaseDGP):
    """
    D4: Multi-factor redundant DGP.
    
    At least one coordinate is a nonlinear function of two (or more) factors,
    for example: (Z_1, Z_2, Z_3, ...) = (S_1, S_2, g(S_1, S_2), ...)
    so that Z_3 carries no information beyond (Z_1, Z_2).
    
    The DGP generates d-1 independent source factors, then creates one
    redundant factor as a nonlinear function of the first two sources.
    """
    
    def __init__(
        self,
        d: int,
        redundant_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the D4 Multi-factor redundant DGP.
        
        Args:
            d: Number of latent factors (must be >= 3).
            redundant_fn: Function g that maps two source factors to the
                redundant factor. Defaults to g(x, y) = x * y.
            noise_std: Standard deviation of optional noise added to the 
                redundant factor. Default is 0 (deterministic).
            seed: Optional random seed for reproducibility.
        """
        if d < 3:
            raise ValueError("d must be at least 3 for D4 (need 2 sources + redundant)")
        super().__init__(d=d, seed=seed)
        
        self.redundant_fn = redundant_fn if redundant_fn is not None else (lambda x, y: x * y)
        self.noise_std = noise_std
    
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples with one multi-factor redundant coordinate.
        
        Args:
            n: Number of samples to generate.
            
        Returns:
            Z: Array of shape (n, d) where Z[:, 2] = g(Z[:, 0], Z[:, 1]) + noise.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        # Generate d-1 independent source factors
        sources = self._rng.standard_normal(size=(n, self.d - 1))
        
        # Create redundant factor from first two sources
        redundant = self.redundant_fn(sources[:, 0], sources[:, 1])
        if self.noise_std > 0:
            redundant = redundant + self._rng.normal(0, self.noise_std, size=n)
        
        # Assemble Z = [S_1, S_2, g(S_1, S_2), S_3, S_4, ...]
        Z = np.zeros((n, self.d))
        Z[:, 0] = sources[:, 0]
        Z[:, 1] = sources[:, 1]
        Z[:, 2] = redundant
        if self.d > 3:
            Z[:, 3:] = sources[:, 2:]
        
        return Z
