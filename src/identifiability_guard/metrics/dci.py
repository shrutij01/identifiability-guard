# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
import numpy as np
import scipy.stats
from sklearn import ensemble
from typing import Optional

from .base import BaseMetric, MetricResult
from ._numerical import safe_entropy_eps, EPS, clamp


# ============================================================================
# Core DCI functions
# ============================================================================


def compute_importance_gbt(
    x_train,
    y_train,
    x_test,
    y_test,
    discrete_factors=None,
):
    """Compute importance based on gradient boosted trees.

    Supports both discrete and continuous factors:
    - Discrete: Uses GradientBoostingClassifier (informativeness = accuracy)
    - Continuous: Uses GradientBoostingRegressor (informativeness = R²)

    Args:
        x_train: Training codes of shape (num_codes, n_train).
        y_train: Training factors of shape (num_factors, n_train).
        x_test: Test codes of shape (num_codes, n_test).
        y_test: Test factors of shape (num_factors, n_test).
        discrete_factors: Optional list of bools indicating which factors are discrete.
            If None, auto-detects: factors with integer-like values are discrete,
            otherwise continuous.

    Returns:
        importance_matrix: Array of shape (num_codes, num_factors).
        train_score: Mean informativeness on training set.
        test_score: Mean informativeness on test set.
    """
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(
        shape=[num_codes, num_factors], dtype=np.float64
    )
    train_scores = []
    test_scores = []

    for i in range(num_factors):
        # Determine if this factor is discrete or continuous
        if discrete_factors is not None:
            is_discrete = discrete_factors[i]
        else:
            # Auto-detect: check if all values are integer-like
            factor_values = y_train[i, :]
            is_discrete = np.allclose(factor_values, np.round(factor_values))

        if is_discrete:
            # Use classifier for discrete factors
            model = ensemble.GradientBoostingClassifier()
            try:
                model.fit(x_train.T, y_train[i, :])
            except ValueError as e:
                if "continuous" in str(e):
                    raise ValueError(
                        f"Factor {i} was specified as discrete but has continuous values. "
                        f"Either set discrete_factors[{i}]=False or ensure the factor "
                        f"has integer-like values."
                    ) from e
                raise
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            # Informativeness = accuracy
            train_scores.append(
                np.mean(model.predict(x_train.T) == y_train[i, :])
            )
            test_scores.append(
                np.mean(model.predict(x_test.T) == y_test[i, :])
            )
        else:
            # Use regressor for continuous factors
            model = ensemble.GradientBoostingRegressor()
            model.fit(x_train.T, y_train[i, :])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            # Informativeness = R² score
            train_scores.append(model.score(x_train.T, y_train[i, :]))
            test_scores.append(model.score(x_test.T, y_test[i, :]))

    return importance_matrix, np.mean(train_scores), np.mean(test_scores)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    eps = safe_entropy_eps(importance_matrix)
    raw = 1.0 - scipy.stats.entropy(
        importance_matrix.T + eps, base=importance_matrix.shape[1], axis=0
    )
    return np.clip(raw, 0.0, 1.0)


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() < EPS:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return clamp(np.sum(per_code * code_importance))


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    eps = safe_entropy_eps(importance_matrix)
    raw = 1.0 - scipy.stats.entropy(
        importance_matrix + eps, base=importance_matrix.shape[0], axis=0
    )
    return np.clip(raw, 0.0, 1.0)


def completeness(importance_matrix):
    """Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() < EPS:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return clamp(np.sum(per_factor * factor_importance))


# ============================================================================
# Standardised wrapper
# ============================================================================


class DCIMetric(BaseMetric):
    """
    DCI metric conforming to BaseMetric interface.

    Uses the canonical DisentanglementLib implementation above.

    Supports both discrete and continuous factors:
    - Discrete: Uses GradientBoostingClassifier (informativeness = accuracy)
    - Continuous: Uses GradientBoostingRegressor (informativeness = R²)
    - Auto-detection: Factors with integer-like values are treated as discrete
    """

    def __init__(
        self,
        train_test_split: float = 0.8,
        random_state: Optional[int] = None,
        discrete_factors: Optional[list] = None,
    ):
        """
        Args:
            train_test_split: Fraction of data for training.
            random_state: Random seed for reproducibility.
            discrete_factors: Optional list of bools indicating which factors are discrete.
                E.g., [True, False, True] means factors 0 and 2 are discrete.
                If None, auto-detects based on whether values are integer-like.
        """
        if not (0.0 < train_test_split < 1.0):
            raise ValueError(
                f"train_test_split must be in (0, 1), got {train_test_split}"
            )
        self.train_test_split = train_test_split
        self.random_state = random_state
        self.discrete_factors = discrete_factors

    @property
    def required_min_samples(self) -> int:
        """DCI needs at least 10 samples."""
        return 10

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """Compute from samples (n, d) and (n, m)."""
        n = Z.shape[0]
        d = Z.shape[1]
        n_train = int(n * self.train_test_split)

        # Validate discrete_factors if provided
        if self.discrete_factors is not None:
            if len(self.discrete_factors) != d:
                raise ValueError(
                    f"discrete_factors has length {len(self.discrete_factors)}, "
                    f"but Z has {d} factors"
                )
            if not all(isinstance(x, bool) for x in self.discrete_factors):
                raise TypeError("discrete_factors must be a list of bools")

        # Shuffle if random_state provided
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(n)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
        else:
            train_idx = slice(None, n_train)
            test_idx = slice(n_train, None)

        # Transpose to (features, samples) for DisentanglementLib
        mus_train = Z_hat[train_idx].T
        ys_train = Z[train_idx].T
        mus_test = Z_hat[test_idx].T
        ys_test = Z[test_idx].T

        # Use DisentanglementLib functions with auto-detection
        importance_matrix, train_acc, test_acc = compute_importance_gbt(
            mus_train,
            ys_train,
            mus_test,
            ys_test,
            discrete_factors=self.discrete_factors,
        )

        disent = float(disentanglement(importance_matrix))
        complet = float(completeness(importance_matrix))

        # Clamp informativeness to [0, 1] to avoid failures when regressors return negative R²
        train_info = float(np.clip(train_acc, 0.0, 1.0))
        test_info = float(np.clip(test_acc, 0.0, 1.0))

        return self.make_result(
            primary_score=disent,
            subscores={
                "disentanglement": disent,
                "completeness": complet,
                "informativeness_train": train_info,
                "informativeness_test": test_info,
            },
            metadata={
                "train_informativeness_raw": float(train_acc),
                "test_informativeness_raw": float(test_acc),
            },
        )

    def compute_from_matrix(self, R: np.ndarray) -> MetricResult:
        """Compute from importance matrix (num_codes, num_factors)."""
        self._validate_matrix(R)

        disent = float(disentanglement(R))
        complet = float(completeness(R))

        return self.make_result(
            primary_score=disent,
            subscores={
                "disentanglement": disent,
                "completeness": complet,
            },
        )
