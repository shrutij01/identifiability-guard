"""D3: Single-factor redundant data generating process."""

from typing import Callable, Optional

import numpy as np

from .base import BaseDGP


class D3SingleRedundant(BaseDGP):
    """
    D3: Single-factor redundant DGP.
    
    At least one coordinate is a nonlinear function of a single other factor,
    for example: (Z_1, Z_2, ...) = (S_1, f(S_1), ...)
    so that Z_2 carries no information beyond Z_1.
    
    The DGP generates d-1 independent source factors, then creates one
    redundant factor as a nonlinear function of the first source.
    """
    
    def __init__(
        self,
        d: int,
        redundant_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the D3 Single-factor redundant DGP.
        
        Args:
            d: Number of latent factors (must be >= 2).
            redundant_fn: Function f that maps the first source factor to the
                redundant factor. Defaults to f(x) = x^2.
            noise_std: Standard deviation of optional noise added to the 
                redundant factor. Default is 0 (deterministic).
            seed: Optional random seed for reproducibility.
        """
        if d < 2:
            raise ValueError("d must be at least 2 for D3 (need source + redundant)")
        super().__init__(d=d, seed=seed)
        
        self.redundant_fn = redundant_fn if redundant_fn is not None else (lambda x: x**2)
        self.noise_std = noise_std
    
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples with one redundant factor.
        
        Args:
            n: Number of samples to generate.
            
        Returns:
            Z: Array of shape (n, d) where Z[:, 1] = f(Z[:, 0]) + noise.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        # Generate d-1 independent source factors
        sources = self._rng.standard_normal(size=(n, self.d - 1))
        
        # Create redundant factor from first source
        redundant = self.redundant_fn(sources[:, 0])
        if self.noise_std > 0:
            redundant = redundant + self._rng.normal(0, self.noise_std, size=n)
        
        # Assemble Z = [S_1, f(S_1), S_2, S_3, ...]
        Z = np.zeros((n, self.d))
        Z[:, 0] = sources[:, 0]
        Z[:, 1] = redundant
        if self.d > 2:
            Z[:, 2:] = sources[:, 1:]
        
        return Z
