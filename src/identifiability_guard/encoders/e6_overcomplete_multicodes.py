"""E6: Overcomplete, multiple codes per factor encoder."""

from typing import Callable, List, Optional

import numpy as np

from .base import BaseEncoder


class E6OvercompleteMulticodes(BaseEncoder):
    r"""
    E6: Overcomplete, multiple codes per factor encoder.

    Higher dimensionality $m > d$. Each learned coordinate depends on exactly
    one ground-truth factor via an invertible nonlinear function, but a single
    factor may be encoded by multiple coordinates (multi-codes):
        $\hat{Z}_j = f_j\bigl(Z_{\mathrm{src}(j)}\bigr)$

    This is the nonlinear analogue of E5 (overcomplete linear).
    """

    def __init__(
        self,
        d: int,
        m: int = None,
        nonlinear_fns: Optional[
            List[Callable[[np.ndarray], np.ndarray]]
        ] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the E6 overcomplete multicodes encoder.

        Args:
            d: Input dimensionality (ground-truth factors).
            m: Output dimensionality (must be > d).
            nonlinear_fns: List of univariate invertible nonlinear functions.
                Each function maps R^n → R^n element-wise.
                If None, uses a mix of default invertible functions.
                Functions are cycled if fewer than m are provided.
            seed: Optional random seed for reproducibility.
        """
        if m is None:
            m = d + 1

        if m <= d:
            raise ValueError(f"E6 requires m > d, got m={m}, d={d}")

        super().__init__(d=d, m=m, seed=seed)
        self._user_fns = nonlinear_fns

        # Parameters to be initialized
        self.source_indices: Optional[np.ndarray] = None
        self._functions: Optional[List[Callable]] = None

    def _initialize_parameters(self) -> None:
        """Initialize source mapping and select nonlinear functions."""
        # Map each of m outputs to one of d inputs
        # Ensure each input is used at least once
        base_assignment = np.arange(self.d)
        extra_assignments = self._rng.choice(
            self.d, size=self.m - self.d, replace=True
        )
        self.source_indices = np.concatenate(
            [base_assignment, extra_assignments]
        )
        self._rng.shuffle(self.source_indices)

        # Set up one function per output column, cycling through available fns
        if self._user_fns is not None:
            pool = self._user_fns
        else:
            pool = self._get_default_nonlinear_invertible_functions()
        self._functions = [pool[j % len(pool)] for j in range(self.m)]

        self._initialized = True

    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply overcomplete nonlinear transformation with multiple codes.

        Each output column j reads a single source factor (determined by
        source_indices[j]) and applies a univariate nonlinear function.

        Args:
            Z: Array of shape (n, d) containing ground-truth factors.

        Returns:
            Z_hat: Array of shape (n, m) with multiple nonlinear codes per factor.
        """
        if Z.shape[1] != self.d:
            raise ValueError(
                f"Expected input with {self.d} factors, got {Z.shape[1]}"
            )

        if not self._initialized:
            self._initialize_parameters()

        # Each output column picks its source factor, then applies its nonlinear fn
        Z_mapped = Z[:, self.source_indices]  # (n, m)
        Z_hat = np.empty_like(Z_mapped)
        for j in range(self.m):
            Z_hat[:, j] = self._functions[j](Z_mapped[:, j])

        return Z_hat
