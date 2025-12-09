"""D4: Multi-factor redundant data generating process."""

from typing import Callable, List, Optional, Union

import numpy as np

from .base import BaseDGP


class D4MultiRedundant(BaseDGP):
    r"""
    D4: Multi-factor redundant DGP.
    
    At least one coordinate is a nonlinear function of two (or more) factors,
    for example: $(Z_1, Z_2, Z_3, \ldots) = (S_1, S_2, g(S_1, S_2), \ldots)$
    so that $Z_3$ carries no information beyond $(Z_1, Z_2)$.
    
    The DGP generates $d-r$ independent source factors, then creates $r$
    redundant factors as a nonlinear function of the first two sources.
    """
    
    def __init__(
        self,
        d: int,
        r: int = None,
        redundant_fns: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        r"""
        Initialize the D4 Multi-factor redundant DGP.
        
        Args:
            d: Number of latent factors (must be >= 3).
            redundant_fn: Function $g:\mathbb{R}^{d-r} \to \mathbb{R}^r$ that maps $d-r$ source factors to the $r$
                redundant factor. Defaults to $g_{d-r+i}(x_1, \ldots, x_{d-r}) = x_i * \sum_{j \neq i} x_j$.
            noise_std: Standard deviation of optional noise added to the
                redundant factor. Default is 0 (deterministic).
            seed: Optional random seed for reproducibility.
        """
        if d < 3:
            raise ValueError("d must be at least 3 for D4 (need 2 sources + redundant)")
        super().__init__(d=d, seed=seed)
        
        if r is None:
            r = int(np.sqrt(d))
        
        self.r = r
        if r < 1 or r >= d/2:
            raise ValueError(f"r must be at least 1 and less than d/2 for D4, got r={r}, d={d}")
        
        if redundant_fns is not None:
            if callable(redundant_fns):
                # Single function provided, wrap in list
                self.redundant_fns = [redundant_fns] * self.r
            elif isinstance(redundant_fns, list):
                if len(redundant_fns) != self.r:
                    raise ValueError(f"Length of redundant_fns ({len(redundant_fns)}) must match r ({self.r})")
                self.redundant_fns = redundant_fns
            else:
                raise ValueError("redundant_fns must be a callable or list of callables")
        else:
            default_fns = self._get_default_bivariate_functions()
            self.redundant_fns = [default_fns[i % len(default_fns)] for i in range(self.r)]
        
        self.noise_std = noise_std
    
    @staticmethod
    def _get_default_bivariate_functions() -> List[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Return a list of default bivariate nonlinear functions."""
        return [
            lambda x, y: x * y,                         # product
            lambda x, y: x + y,                         # sum
            lambda x, y: x**2 + y**2,                   # sum of squares
            lambda x, y: np.tanh(x) * np.tanh(y),       # tanh product
            lambda x, y: np.sign(x * y) * np.sqrt(np.abs(x * y)),  # signed sqrt of product
        ]
    
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples with multi-factor redundant coordinates.
        
        Args:
            n: Number of samples to generate.
            
        Returns:
            Z: Array of shape (n, d) where Z[:, 2:2+r] = g_i(Z[:, 0], Z[:, 1]) + noise,
               and remaining columns are independent sources.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        # Generate d-r independent source factors
        sources = self._rng.standard_normal(size=(n, self.d - self.r))
        
        # Create redundant factors from first two sources
        redundant = np.column_stack([fn(sources[:, 0], sources[:, 1]) for fn in self.redundant_fns])
        if self.noise_std > 0:
            redundant = redundant + self._rng.normal(0, self.noise_std, size=(n, self.r))
        
        # Assemble Z = [S_1, S_2, g(S_1, S_2), ..., S_3, S_4, ...]
        Z = np.zeros((n, self.d))
        Z[:, 0] = sources[:, 0]  # First source
        Z[:, 1] = sources[:, 1]  # Second source
        Z[:, 2:2+self.r] = redundant  # Redundant factors
        Z[:, 2+self.r:] = sources[:, 2:]  # Remaining independent sources
        
        return Z
