"""Explicitness R² metric implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseMetric, MetricResult


class R2Metric(BaseMetric):
    """
    Explicitness R² metric.

    For each ground-truth factor ``Z[:, i]``, fits a linear regression on ``Z_hat``
    to predict ``Z[:, i]`` and computes a variance-normalised R². The final score
    is the mean across all factors. R² can be negative if the linear model
    underperforms the mean predictor.
    """

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        score = self._compute_r2(Z, Z_hat)
        subscores: Dict[str, float] = {"r2": score}
        metadata: Dict[str, Any] = {"metric": "explicitness_r2"}
        return self.make_result(
            primary_score=score, subscores=subscores, metadata=metadata
        )

    def _compute_r2(self, Z: np.ndarray, Z_hat: np.ndarray) -> float:
        n, d = Z.shape
        r2_scores = []

        for i in range(d):
            y = Z[:, i]
            X = Z_hat

            y_pred = self._least_squares(X, y)

            mse = float(np.mean((y - y_pred) ** 2))
            var = float(np.var(y))

            # If variance is zero, treat as perfectly explained to avoid div by zero.
            if var == 0.0:
                r2_i = 1.0
            else:
                r2_i = 1.0 - mse / var
            r2_scores.append(r2_i)

        if not r2_scores:
            return 0.0

        # Replace any non-finite values before averaging to keep metric stable
        r2_array = np.nan_to_num(np.array(r2_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.mean(r2_array))

    @staticmethod
    def _least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve min_w ||Xw - y||^2 with stable lstsq and return predictions."""
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return X @ w

    @property
    def score_range(self) -> tuple[float, float]:
        # R² can be negative; leave upper bound at 1.0.
        return -np.inf, 1.0

    @property
    def required_min_samples(self) -> int:
        # At least one sample needed, but lstsq generally needs n >= m; enforced implicitly.
        return 1

