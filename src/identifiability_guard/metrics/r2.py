"""Explicitness R² metric implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseMetric, MetricResult
from ._numerical import EPS


class R2Metric(BaseMetric):
    """
    Explicitness R² metric.

    For each ground-truth factor ``Z[:, i]``, fits a linear regression on ``Z_hat``
    to predict ``Z[:, i]`` and computes a variance-normalised R². The final score
    is the mean across all factors. R² can be negative if the linear model
    underperforms the mean predictor.
    """

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        score, nan_info = self._compute_r2(Z, Z_hat)
        subscores: Dict[str, float] = {"r2": score}
        metadata: Dict[str, Any] = {
            "metric": "explicitness_r2",
            "nan_info": nan_info,
        }
        return self.make_result(
            primary_score=score, subscores=subscores, metadata=metadata
        )

    def compute_oos(
        self,
        Z_train: np.ndarray,
        Z_hat_train: np.ndarray,
        Z_test: np.ndarray,
        Z_hat_test: np.ndarray,
    ) -> MetricResult:
        """Fit linear regression on train, evaluate R² on held-out test."""
        self._validate_samples(Z_train, Z_hat_train)
        self._validate_samples(Z_test, Z_hat_test)
        score, nan_info = self._compute_r2_oos(
            Z_train,
            Z_hat_train,
            Z_test,
            Z_hat_test,
        )
        result = self.make_result(
            primary_score=score,
            subscores={"r2": score},
            metadata={
                "metric": "explicitness_r2",
                "oos": True,
                "nan_info": nan_info,
            },
        )
        self._validate_result_type(result)
        self._validate_result_range(result)
        return result

    def _compute_r2(self, Z: np.ndarray, Z_hat: np.ndarray) -> tuple:
        n, d = Z.shape

        if d == 0:
            return 0.0, {
                "zero_variance_factors": 0,
                "nonfinite_r2_count": 0,
                "valid_factors": 0,
                "total_factors": 0,
            }

        W, *_ = np.linalg.lstsq(Z_hat, Z, rcond=None)
        Z_pred = Z_hat @ W

        mse = np.mean((Z - Z_pred) ** 2, axis=0)
        var = np.var(Z, axis=0)

        zero_var_mask = var < EPS
        zero_var_count = int(np.sum(zero_var_mask))
        valid_count = d - zero_var_count

        # Zero-variance factors: R² undefined (SS_tot=0) → treat as 0.
        r2_array = np.where(
            zero_var_mask, 0.0, 1.0 - mse / np.maximum(var, EPS)
        )

        nonfinite_count = int(np.sum(~np.isfinite(r2_array)))
        # Clamp to [0, 1]: negative R² means "no predictive power" = 0.
        r2_array = np.nan_to_num(r2_array, nan=0.0, posinf=1.0, neginf=0.0)
        r2_array = np.clip(r2_array, 0.0, 1.0)

        score = float(np.mean(r2_array))

        nan_info = {
            "zero_variance_factors": zero_var_count,
            "nonfinite_r2_count": nonfinite_count,
            "valid_factors": valid_count,
            "total_factors": d,
        }
        return score, nan_info

    def _compute_r2_oos(
        self,
        Z_train: np.ndarray,
        Z_hat_train: np.ndarray,
        Z_test: np.ndarray,
        Z_hat_test: np.ndarray,
    ) -> tuple:
        """Fit on train, score on test."""
        d = Z_train.shape[1]
        if d == 0:
            return 0.0, {"zero_variance_factors": 0, "nonfinite_r2_count": 0}

        # Fit on train
        W, *_ = np.linalg.lstsq(Z_hat_train, Z_train, rcond=None)

        # Predict on test
        Z_pred = Z_hat_test @ W

        mse = np.mean((Z_test - Z_pred) ** 2, axis=0)
        var = np.var(Z_test, axis=0)

        zero_var_mask = var < EPS
        zero_var_count = int(np.sum(zero_var_mask))
        valid_count = d - zero_var_count
        # Zero-variance factors: R² undefined (SS_tot=0) → treat as 0.
        r2_array = np.where(
            zero_var_mask, 0.0, 1.0 - mse / np.maximum(var, EPS)
        )

        nonfinite_count = int(np.sum(~np.isfinite(r2_array)))
        # Clamp to [0, 1]: negative R² means "no predictive power" = 0.
        r2_array = np.nan_to_num(r2_array, nan=0.0, posinf=1.0, neginf=0.0)
        r2_array = np.clip(r2_array, 0.0, 1.0)

        score = float(np.mean(r2_array))

        nan_info = {
            "zero_variance_factors": zero_var_count,
            "nonfinite_r2_count": nonfinite_count,
            "valid_factors": valid_count,
            "total_factors": d,
        }
        return score, nan_info

    @property
    def score_range(self) -> tuple[float, float]:
        return 0.0, 1.0

    @property
    def required_min_samples(self) -> int:
        # At least one sample needed, but lstsq generally needs n >= m; enforced implicitly.
        return 1
