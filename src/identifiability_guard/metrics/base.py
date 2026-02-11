"""Base class for Identifiability Metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """
    Immutable result container for metric computation.

    Attributes:
        primary_score: Main metric score. Higher is better.
        subscores: Optional dict of named subscores (e.g., {"disentanglement": 0.8}).
        metadata: Optional dict of additional information (e.g., {"method": "lasso"}).
        score_min: Inclusive lower bound the metric is expected to stay above.
        score_max: Inclusive upper bound the metric is expected to stay below.
    """

    primary_score: float
    subscores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    score_min: float = 0.0
    score_max: float = 1.0

    def __post_init__(self):
        """Validate invariants."""
        if not isinstance(self.score_min, (int, float)) or not isinstance(
            self.score_max, (int, float)
        ):
            raise TypeError(
                f"score_min and score_max must be numeric, got "
                f"{type(self.score_min)}, {type(self.score_max)}"
            )
        if self.score_min > self.score_max:
            raise ValueError(
                f"score_min {self.score_min} cannot exceed score_max {self.score_max}"
            )

        if not isinstance(self.primary_score, (int, float)):
            raise TypeError(
                f"primary_score must be numeric, got {type(self.primary_score)}"
            )
        if not (self.score_min <= self.primary_score <= self.score_max):
            raise ValueError(
                f"primary_score {self.primary_score} not in "
                f"[{self.score_min}, {self.score_max}]"
            )

        if self.subscores is not None:
            for key, value in self.subscores.items():
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"subscore '{key}' must be numeric, got {type(value)}"
                    )
                if not (self.score_min <= value <= self.score_max):
                    raise ValueError(
                        f"subscore '{key}'={value} not in "
                        f"[{self.score_min}, {self.score_max}]"
                    )


class BaseMetric(ABC):
    r"""
    Abstract base class for Identifiability Metrics.

    A metric computes an identifiability score $M(Z, \hat{Z}) \in [0, 1]$ from
    ground-truth factors $Z \in \mathbb{R}^{n \times d}$ and learned coordinates
    $\hat{Z} \in \mathbb{R}^{n \times m}$.

    Higher scores indicate better identifiability.

    Two computation modes:
    1. compute(Z, Z_hat): From samples (n, d) and (n, m)
    2. compute_from_matrix(R): From precomputed relationship matrix (d, m)

    Design principles:
    - Single responsibility: Only compute metrics from arrays.
    - Fail fast: Validate inputs before computation.
    - Type safe: Return structured MetricResult, not Union types.
    """

    @abstractmethod
    def _compute_impl(
        self,
        Z: np.ndarray,
        Z_hat: np.ndarray,
    ) -> MetricResult:
        """
        Compute the identifiability metric from samples (implementation).

        Subclasses implement this method. Input validation is already done.

        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            Z_hat: Array of shape (n, m) containing learned coordinates.

        Returns:
            MetricResult with primary_score in [0, 1] and optional subscores.
        """
        pass

    def compute(
        self,
        Z: np.ndarray,
        Z_hat: np.ndarray,
    ) -> MetricResult:
        """
        Compute the identifiability metric from samples.

        Validates inputs, then delegates to _compute_impl().

        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            Z_hat: Array of shape (n, m) containing learned coordinates.

        Returns:
            MetricResult with primary_score in [0, 1] and optional subscores.

        Raises:
            ValueError: If inputs are invalid (wrong shape, NaN/Inf values, etc.)
        """
        self._validate_samples(Z, Z_hat)
        result = self._compute_impl(Z, Z_hat)
        self._validate_result_type(result)
        self._validate_result_range(result)
        return result

    def compute_from_matrix(self, R: np.ndarray) -> MetricResult:
        """
        Compute the identifiability metric from a precomputed relationship matrix.

        This is useful when you already have a correlation matrix, importance matrix,
        or other pairwise relationship matrix and want to skip recomputing it.

        Args:
            R: Array of shape (d, m) representing relationships between
                d ground-truth factors and m learned codes.

        Returns:
            MetricResult with primary_score in [0, 1] and optional subscores.

        Raises:
            NotImplementedError: If this metric doesn't support matrix input.
            ValueError: If matrix is invalid.
        """
        raise NotImplementedError(
            f"{self.name} does not support compute_from_matrix(). "
            f"Use compute(Z, Z_hat) instead."
        )

    def _validate_samples(self, Z: np.ndarray, Z_hat: np.ndarray) -> None:
        """
        Validate sample inputs before computation.

        Args:
            Z: Ground-truth factors array.
            Z_hat: Learned coordinates array.

        Raises:
            ValueError: If inputs don't meet requirements.
        """
        # Check array types
        if not isinstance(Z, np.ndarray) or not isinstance(Z_hat, np.ndarray):
            raise TypeError(
                f"Expected numpy arrays, got {type(Z)}, {type(Z_hat)}"
            )

        # Check dimensionality
        if Z.ndim != 2 or Z_hat.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays (n, d) and (n, m), got shapes {Z.shape} and {Z_hat.shape}"
            )

        # Check sample count match
        if Z.shape[0] != Z_hat.shape[0]:
            raise ValueError(
                f"Sample count mismatch: Z has {Z.shape[0]} samples, "
                f"Z_hat has {Z_hat.shape[0]} samples"
            )

        # Check minimum samples
        min_samples = self.required_min_samples
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError(
                f"{self.name}.required_min_samples must be an int >= 1, got {min_samples}"
            )
        if Z.shape[0] < min_samples:
            raise ValueError(
                f"{self.name} requires at least {min_samples} samples, got {Z.shape[0]}"
            )

        # Check for NaN/Inf
        if not np.all(np.isfinite(Z)):
            raise ValueError("Z contains NaN or Inf values")
        if not np.all(np.isfinite(Z_hat)):
            raise ValueError("Z_hat contains NaN or Inf values")

        # Check for empty dimensions
        if Z.shape[1] == 0 or Z_hat.shape[1] == 0:
            raise ValueError(
                f"Empty feature dimensions: Z has {Z.shape[1]} factors, "
                f"Z_hat has {Z_hat.shape[1]} codes"
            )
        self._validate_additional_sample_constraints(Z, Z_hat)

    def _validate_matrix(self, R: np.ndarray) -> None:
        """
        Validate matrix input before computation.

        Args:
            R: Relationship matrix.

        Raises:
            ValueError: If matrix is invalid.
        """
        if not isinstance(R, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(R)}")

        if R.ndim != 2:
            raise ValueError(f"Expected 2D matrix (d, m), got shape {R.shape}")

        if R.shape[0] == 0 or R.shape[1] == 0:
            raise ValueError(f"Empty matrix dimensions: {R.shape}")

        if not np.all(np.isfinite(R)):
            raise ValueError("Matrix contains NaN or Inf values")

    def __call__(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """
        Shorthand for compute().

        Allows: result = metric(Z, Z_hat)
        """
        return self.compute(Z, Z_hat)

    @property
    def name(self) -> str:
        """Return the name of the metric class."""
        return self.__class__.__name__

    @property
    def score_range(self) -> Tuple[float, float]:
        """
        Inclusive range [min, max] that this metric's scores are expected to occupy.

        Subclasses can override to widen or shift the range. Defaults to [0, 1].
        """
        return 0.0, 1.0

    def make_result(
        self,
        primary_score: float,
        subscores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Helper to build a MetricResult that respects the metric's score_range.
        """
        score_min, score_max = self.score_range
        return MetricResult(
            primary_score=primary_score,
            subscores=subscores,
            metadata=metadata,
            score_min=score_min,
            score_max=score_max,
        )

    def _validate_result_type(self, result: MetricResult) -> None:
        """Ensure metric implementations return MetricResult."""
        if not isinstance(result, MetricResult):
            raise TypeError(
                f"{self.name}._compute_impl must return MetricResult, got {type(result)}"
            )

    def _validate_result_range(self, result: MetricResult) -> None:
        """Check that the returned score fits within the metric's configured range."""
        score_min, score_max = self.score_range
        if score_min > score_max:
            raise ValueError(
                f"{self.name}.score_range is invalid: ({score_min}, {score_max})"
            )
        if not (score_min <= result.primary_score <= score_max):
            raise ValueError(
                f"{self.name} produced score {result.primary_score} outside "
                f"expected range [{score_min}, {score_max}]"
            )

    def _validate_additional_sample_constraints(
        self, Z: np.ndarray, Z_hat: np.ndarray
    ) -> None:
        """
        Hook for subclasses to implement extra sample validation (e.g., n >= d).
        """
        return None

    @property
    def required_min_samples(self) -> int:
        """
        Minimum number of samples required for this metric.

        Subclasses can override this if they need more samples.
        Default is 2 (minimum for any statistical measure).
        """
        return 2
