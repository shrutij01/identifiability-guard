"""E10: Random Uniform encoder (baseline for comparison)."""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class E10RandomUniform(BaseEncoder):
    r"""
    E10: Random Uniform encoder (baseline/null encoder).

    This encoder ignores the input entirely and outputs random values drawn
    uniformly from [-1, 1]. It serves as a baseline/null encoder to verify that
    identifiability metrics correctly identify non-informative representations.
    """

    def __init__(
        self,
        d: int,
        m: int = None,
        low: float = -1.0,
        high: float = 1.0,
        seed: Optional[int] = None,
    ):
        if m is None:
            m = d
        super().__init__(d=d, m=m, seed=seed)
        self.low = low
        self.high = high

    @property
    def name(self) -> str:
        return f"E10: Random Uniform (low={self.low}, high={self.high})"

    @property
    def display_name(self) -> str:
        return "E10RandomUniform"

    def _initialize_parameters(self) -> None:
        self._initialized = True

    def encode(self, Z: np.ndarray) -> np.ndarray:
        if Z.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} features, got {Z.shape[1]}")
        if not self._initialized:
            self._initialize_parameters()
        n_samples = Z.shape[0]
        return self._rng.uniform(low=self.low, high=self.high, size=(n_samples, self.m))
