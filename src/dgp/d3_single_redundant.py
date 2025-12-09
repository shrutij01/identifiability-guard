"""D3: Single-factor redundant data generating process."""

from typing import Callable, Optional

import numpy as np

from .base import BaseDGP


class D3SingleRedundant(BaseDGP):
    r"""
    D3: Single-factor redundant DGP.
    
    At least one coordinate is a nonlinear function of a single other factor,
    for example: ($Z_1, Z_2, \ldots$) = ($S_1, f(S_1), \ldots$)
    so that $Z_2$ carries no information beyond $Z_1$.
    
    The DGP generates $d-r$ independent source factors, then creates $r$
    redundant factors as a nonlinear function of the first source.
    """
    
    def __init__(
        self,
        d: int,
        r: int = 1,
        redundant_fns: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        r"""
        Initialize the D3 Single-factor redundant DGP.
        
        Args:
            d: Number of latent factors (must be >= 2).
            r: Number of redundant factors.
            redundant_fns: Functions $f_i:\mathbb{R} \to \mathbb{R}$ that map the first source factor to the
                redundant factors. 
            noise_std: Standard deviation of optional noise added to the 
                redundant factor. Default is 0 (deterministic).
            seed: Optional random seed for reproducibility.
        """
        if d < 2:
            raise ValueError("d must be at least 2 for D3 (need source + redundant)")
        super().__init__(d=d, seed=seed)
        
        self.r = r
        if r < 1 or r >= d/2:
            raise ValueError("r must be at least 1 and less than d/2 for D3")
        
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
            default_fns = self._get_default_nonlinear_invertible_functions()
            self.redundant_fns = [default_fns[i % len(default_fns)] for i in range(self.r)]

        self.noise_std = noise_std
    
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples with one redundant factor.
        
        Args:
            n: Number of samples to generate.
            
        Returns:
            Z: Array of shape (n, d) where Z[:, 1:1+r] = f_i(Z[:, 0]) + noise,
               and remaining columns are independent sources.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        # Generate d-r independent source factors
        sources = self._rng.standard_normal(size=(n, self.d - self.r))
        
        # Create redundant factors from first source
        redundant = np.column_stack([fn(sources[:, 0]) for fn in self.redundant_fns])
        if self.noise_std > 0:
            redundant = redundant + self._rng.normal(0, self.noise_std, size=(n, self.r))
        
        # Assemble Z = [S_1, f(S_1), ..., f(S_1), S_2, S_3, ...]
        Z = np.zeros((n, self.d))
        Z[:, 0] = sources[:, 0]  # First source
        Z[:, 1:1+self.r] = redundant  # Redundant factors
        Z[:, 1+self.r:] = sources[:, 1:]  # Remaining independent sources
        
        return Z
