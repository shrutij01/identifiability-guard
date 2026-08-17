"""MCCMetric — BaseMetric-compatible interface for MCC."""

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..base import BaseMetric, MetricResult
from ._legacy import mean_corr_coef
from ._crossfit import mcc_train_test_np, mean_corr_coef_crossfit_np


class MCCMetric(BaseMetric):
    """
    Mean Correlation Coefficient (MCC) metric conforming to BaseMetric interface.

    MCC finds the optimal permutation that maximizes the mean absolute correlation
    between matched pairs of factors and codes.

    Dimension handling:
      - d == m: standard square assignment.
      - d < m (overcomplete): all d true factors are matched; m-d learned
        dimensions are ignored. Legacy and coverage scores coincide.
      - d > m (undercomplete): only m of d true factors are matched.
        Legacy MCC divides by m (the matched count). Coverage MCC divides
        by d, so unmatched factors contribute zero.

    Modes:
      - crossfit=False (default): in-sample MCC using all data for both
        assignment selection and scoring. This is the legacy definition.
      - crossfit=True: K-fold cross-validated MCC. Learns assignment and
        correlation signs on each training fold, scores on the held-out fold,
        returns the weighted average. Removes the in-sample selection bias
        that inflates legacy MCC when m >> d.

    Normalization:
      - "matched" (default): divide by min(d, m). Legacy-compatible.
      - "coverage": divide by d. Penalizes missing factors when d > m.
    """

    def __init__(
        self,
        method: str = "pearson",
        seed: Optional[int] = None,
        normalization: str = "matched",
        crossfit: bool = False,
        n_splits: int = 5,
    ):
        """
        Args:
            method: Correlation method — 'pearson', 'spearman', or 'rdc'.
            seed: Random seed for reproducibility. Used by 'rdc' and for the
                  crossfit K-fold split.
            normalization: 'matched' (divide by min(d,m)) or 'coverage'
                           (divide by d).
            crossfit: If True, use K-fold cross-validated MCC.
            n_splits: Number of cross-validation folds (default 5).
                      Only used when crossfit=True.
        """
        if method not in ("pearson", "spearman", "rdc"):
            raise ValueError(
                f"method must be 'pearson', 'spearman', or 'rdc', got {method}"
            )
        if normalization not in ("matched", "coverage"):
            raise ValueError(
                f"normalization must be 'matched' or 'coverage', got {normalization}"
            )
        if crossfit and method == "rdc":
            raise ValueError(
                "crossfit is not supported with method='rdc' (RDC uses random "
                "projections that are incompatible with sign-corrected cross-fitting)"
            )
        self.method = method
        self.seed = seed
        self.normalization = normalization
        self.crossfit = crossfit
        self.n_splits = n_splits

    @property
    def required_min_samples(self) -> int:
        return 3 * self.n_splits if self.crossfit else 2

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """Compute MCC from samples."""
        z_constant = int(np.sum(np.std(Z, axis=0) == 0))
        zhat_constant = int(np.sum(np.std(Z_hat, axis=0) == 0))

        d, m = Z.shape[1], Z_hat.shape[1]
        k = min(d, m)

        if self.crossfit:
            cf = mean_corr_coef_crossfit_np(
                Z,
                Z_hat,
                method=self.method,
                n_splits=self.n_splits,
                seed=self.seed if self.seed is not None else 0,
                coverage_aware=False,
            )
            matched_score = float(np.clip(cf.score, 0.0, 1.0))
            coverage_score = (
                float(np.clip(matched_score * k / d, 0.0, 1.0)) if d > 0 else 0.0
            )
            extra_meta = {
                "fold_scores": cf.fold_scores.tolist(),
                "n_splits": self.n_splits,
                "raw_score": cf.score,
            }
        else:
            rng = np.random.default_rng(self.seed) if self.seed is not None else None
            matched_score = float(mean_corr_coef(Z, Z_hat, method=self.method, rng=rng))
            if not np.isfinite(matched_score):
                matched_score = 0.0
            matched_score = float(np.clip(matched_score, 0.0, 1.0))
            coverage_score = (
                float(np.clip(matched_score * k / d, 0.0, 1.0)) if d > 0 else 0.0
            )
            extra_meta = {}

        primary = coverage_score if self.normalization == "coverage" else matched_score

        return self.make_result(
            primary_score=primary,
            subscores={
                "mcc": matched_score,
                "mcc_coverage": coverage_score,
            },
            metadata={
                "method": self.method,
                "normalization": self.normalization,
                "crossfit": self.crossfit,
                "nan_info": {
                    "z_constant_columns": z_constant,
                    "zhat_constant_columns": zhat_constant,
                },
                **extra_meta,
            },
        )

    def compute_oos(
        self,
        Z_train: np.ndarray,
        Z_hat_train: np.ndarray,
        Z_test: np.ndarray,
        Z_hat_test: np.ndarray,
    ) -> MetricResult:
        """Out-of-sample MCC.

        When crossfit=True, the provided train/test split is used directly
        via mcc_train_test_np (no K-fold re-splitting).  Otherwise falls
        back to evaluating on the test set only (the legacy default).
        """
        if not self.crossfit:
            return self.compute(Z_test, Z_hat_test)

        self._validate_samples(Z_test, Z_hat_test)
        d, m = Z_train.shape[1], Z_hat_train.shape[1]
        k = min(d, m)

        score, _, _ = mcc_train_test_np(
            Z_train,
            Z_hat_train,
            Z_test,
            Z_hat_test,
            method=self.method,
            coverage_aware=False,
        )
        matched_score = float(np.clip(score, 0.0, 1.0))
        coverage_score = (
            float(np.clip(matched_score * k / d, 0.0, 1.0)) if d > 0 else 0.0
        )
        primary = coverage_score if self.normalization == "coverage" else matched_score

        result = self.make_result(
            primary_score=primary,
            subscores={"mcc": matched_score, "mcc_coverage": coverage_score},
            metadata={
                "method": self.method,
                "normalization": self.normalization,
                "crossfit": True,
                "oos": True,
            },
        )
        self._validate_result_type(result)
        self._validate_result_range(result)
        return result

    def compute_from_matrix(self, R: np.ndarray) -> MetricResult:
        """
        Compute MCC from a precomputed absolute correlation matrix.

        Args:
            R: Absolute correlation matrix of shape (d, m). All entries
                must be non-negative (apply np.abs before calling if needed).
        """
        self._validate_matrix(R)
        if np.any(R < 0):
            raise ValueError(
                "compute_from_matrix expects absolute correlations (all entries >= 0). "
                "Apply np.abs() to signed correlation matrices before calling."
            )

        row_ind, col_ind = linear_sum_assignment(R, maximize=True)
        d = R.shape[0]
        k = len(row_ind)

        matched_score = float(np.clip(R[row_ind, col_ind].mean(), 0.0, 1.0))
        coverage_score = (
            float(np.clip(matched_score * k / d, 0.0, 1.0)) if d > 0 else 0.0
        )
        primary = coverage_score if self.normalization == "coverage" else matched_score

        return self.make_result(
            primary_score=primary,
            subscores={
                "mcc": matched_score,
                "mcc_coverage": coverage_score,
            },
            metadata={
                "computed_from": "matrix",
                "normalization": self.normalization,
            },
        )
