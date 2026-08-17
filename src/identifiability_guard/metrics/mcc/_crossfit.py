"""Cross-fitted MCC and null calibration.

Contains:
- cross_correlation_np (fast d×m correlation via BLAS matmul)
- mcc_train_test_np (single train/test split with sign correction)
- make_kfold_splits (K-fold index utility)
- mean_corr_coef_crossfit_np (K-fold cross-fitted MCC)
- legacy_mcc_np (clean in-sample MCC wrapper)
- permutation_null_np (permutation null calibration)
- CrossfitMCCResult / NullCalibration (result containers)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

CorrelationMethod = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class CrossfitMCCResult:
    """Result container for K-fold cross-fitted MCC."""

    score: float
    fold_scores: np.ndarray
    fold_sizes: np.ndarray
    d: int
    m: int
    n: int
    coverage_aware: bool


@dataclass(frozen=True)
class NullCalibration:
    """Result of permutation null calibration for MCC."""

    null_scores: np.ndarray
    null_mean: float
    null_q95: float
    p_value_upper: float
    adjusted_score: Optional[float]


def cross_correlation_np(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the d x m signed cross-correlation matrix via BLAS matmul.

    Faster than np.corrcoef (avoids building the full (d+m)x(d+m) matrix).
    Constant columns receive zero correlation.
    """
    if method == "spearman":
        x = rankdata(x, axis=0, method="average").astype(np.float64)
        y = rankdata(y, axis=0, method="average").astype(np.float64)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    xnorm = np.linalg.norm(xc, axis=0)
    ynorm = np.linalg.norm(yc, axis=0)
    denom = xnorm[:, None] * ynorm[None, :]
    numerator = xc.T @ yc
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > eps)
    return np.clip(out, -1.0, 1.0)


def mcc_train_test_np(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    coverage_aware: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Honest train/test MCC with sign correction.

    Assignment and orientation are learned from train correlations. The test
    score applies the train-learned sign to the held-out signed correlation
    (NOT absolute value). Under an independent null, the expected score is zero.

    coverage_aware=False divides by min(d, m) (legacy).
    coverage_aware=True divides by d (penalizes unmatched factors when d > m).
    """
    r_train = cross_correlation_np(x_train, y_train, method=method)
    r_test = cross_correlation_np(x_test, y_test, method=method)

    rows, cols = linear_sum_assignment(np.abs(r_train), maximize=True)
    orientation = np.sign(r_train[rows, cols])
    numerator = float(np.sum(orientation * r_test[rows, cols]))
    denominator = r_train.shape[0] if coverage_aware else len(rows)
    return numerator / denominator, rows, cols


def make_kfold_splits(
    n: int,
    n_splits: int = 5,
    seed: int = 0,
) -> list:
    """Create K-fold train/test index splits."""
    if not 2 <= n_splits <= n // 3:
        raise ValueError(
            f"n_splits must be in [2, {n // 3}] for {n} samples, got {n_splits}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_splits)
    all_idx = np.arange(n)
    splits = []
    for test in folds:
        mask = np.ones(n, dtype=bool)
        mask[test] = False
        splits.append((all_idx[mask], np.asarray(test)))
    return splits


def mean_corr_coef_crossfit_np(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    n_splits: int = 5,
    seed: int = 0,
    splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    coverage_aware: bool = False,
) -> CrossfitMCCResult:
    """K-fold cross-fitted MCC.

    Learns the assignment permutation and correlation signs on each training
    fold, scores on the corresponding held-out fold, and returns the
    weighted-average score across folds.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if splits is None:
        splits = make_kfold_splits(x.shape[0], n_splits=n_splits, seed=seed)

    fold_scores = []
    fold_sizes = []
    for train_idx, test_idx in splits:
        score, _, _ = mcc_train_test_np(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            y[test_idx],
            method=method,
            coverage_aware=coverage_aware,
        )
        fold_scores.append(score)
        fold_sizes.append(len(test_idx))

    fold_scores_arr = np.asarray(fold_scores, dtype=np.float64)
    fold_sizes_arr = np.asarray(fold_sizes, dtype=np.float64)
    score = float(np.average(fold_scores_arr, weights=fold_sizes_arr))
    return CrossfitMCCResult(
        score=score,
        fold_scores=fold_scores_arr,
        fold_sizes=fold_sizes_arr,
        d=x.shape[1],
        m=y.shape[1],
        n=x.shape[0],
        coverage_aware=coverage_aware,
    )


def legacy_mcc_np(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
) -> float:
    """Legacy in-sample absolute MCC, retained for backward compatibility."""
    r = cross_correlation_np(x, y, method=method)
    rows, cols = linear_sum_assignment(np.abs(r), maximize=True)
    return float(np.mean(np.abs(r[rows, cols])))


def permutation_null_np(
    x: np.ndarray,
    y: np.ndarray,
    *,
    observed_score: float,
    metric: Literal["crossfit", "legacy"] = "crossfit",
    method: CorrelationMethod = "pearson",
    n_permutations: int = 199,
    n_splits: int = 5,
    seed: int = 0,
    coverage_aware: bool = False,
) -> NullCalibration:
    """Calibrate MCC with joint row-permutations of y.

    Preserves y's marginal distributions and inter-dimension dependence.
    Returns the null distribution, p-value, and (for legacy) the
    null-adjusted score (observed - null_mean) / (1 - null_mean).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2-D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if metric not in ("crossfit", "legacy"):
        raise ValueError("metric must be 'crossfit' or 'legacy'")
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int):
        raise TypeError("n_permutations must be an integer")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    rng = np.random.default_rng(seed)
    splits = None
    if metric == "crossfit":
        splits = make_kfold_splits(x.shape[0], n_splits=n_splits, seed=seed)
    null = np.empty(n_permutations, dtype=np.float64)

    for b in range(n_permutations):
        yp = y[rng.permutation(y.shape[0])]
        if metric == "crossfit":
            null[b] = mean_corr_coef_crossfit_np(
                x,
                yp,
                method=method,
                splits=splits,
                coverage_aware=coverage_aware,
            ).score
        elif metric == "legacy":
            null[b] = legacy_mcc_np(x, yp, method=method)

    mu0 = float(np.mean(null))
    adjusted = None
    if metric == "legacy":
        adjusted = float((observed_score - mu0) / (1.0 - mu0)) if mu0 < 1.0 else 0.0
    p = float((1 + np.sum(null >= observed_score)) / (n_permutations + 1))
    return NullCalibration(
        null_scores=null,
        null_mean=mu0,
        null_q95=float(np.quantile(null, 0.95)),
        p_value_upper=p,
        adjusted_score=adjusted,
    )
