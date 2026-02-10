"""
Mutual Information Gap (MIG) metric.

Based on DisentanglementLib:
Apache License, Version 2.0, January 2004
https://github.com/google-research/disentanglement_lib

MIG measures disentanglement by computing the gap between the two highest
mutual information values for each factor. A higher gap indicates that
the factor is captured by a single code dimension rather than spread
across multiple dimensions.
"""

import numpy as np
import sklearn.metrics
from typing import Optional

from .base import BaseMetric, MetricResult


def histogram_discretize(
    target: np.ndarray,
    num_bins: int = 20,
) -> np.ndarray:
    """Discretize continuous values into histogram bins."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(
    codes: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """Compute discrete mutual information between codes and factors."""
    num_codes = codes.shape[0]
    num_factors = factors.shape[0]
    mi_matrix = np.zeros((num_codes, num_factors))
    for i in range(num_codes):
        for j in range(num_factors):
            mi_matrix[i, j] = sklearn.metrics.mutual_info_score(
                factors[j, :], codes[i, :]
            )
    return mi_matrix


def discrete_entropy(factors: np.ndarray) -> np.ndarray:
    """Compute discrete entropy for each factor."""
    num_factors = factors.shape[0]
    entropy = np.zeros(num_factors)
    for j in range(num_factors):
        entropy[j] = sklearn.metrics.mutual_info_score(
            factors[j, :], factors[j, :]
        )
    return entropy


def _compute_mig(
    discretized_mus: np.ndarray,
    ys_train: np.ndarray,
) -> tuple:
    """
    Compute MIG score — follows original ``_compute_mig`` from
    disentanglement_lib.

    Args:
        discretized_mus: Discretized codes of shape (num_codes, num_samples).
        ys_train: Factors of shape (num_factors, num_samples).  In the original
            implementation these are assumed to be discrete already.

    Returns:
        Tuple of (mig_score, nan_info) where nan_info is a dict with
        'zero_entropy_factors' count.
    """
    import warnings

    # m is [num_latents, num_factors]
    m = discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == discretized_mus.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    entropy = discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]

    # Guard: factors with zero entropy (constant after discretization)
    # produce NaN when dividing by entropy.  Exclude them and warn.
    valid_mask = entropy > 0.0
    num_zero_entropy = int(np.sum(~valid_mask))

    if num_zero_entropy > 0:
        warnings.warn(
            f"MIG: {num_zero_entropy} factor(s) have zero entropy "
            f"(constant after discretization) and are excluded from the score."
        )

    if not np.any(valid_mask):
        # All factors constant → score is undefined, return 0.
        return 0.0, {'zero_entropy_factors': num_zero_entropy}

    # Guard: if there is only one code dimension, sorted_m has shape (1, num_factors)
    # and sorted_m[1, ...] would be out of bounds.  The MIG gap is 0 by definition
    # when there is nothing to compare against.
    if sorted_m.shape[0] < 2:
        warnings.warn(
            "MIG: only 1 code dimension — gap is trivially 0."
        )
        return 0.0, {'zero_entropy_factors': num_zero_entropy}

    per_factor = (sorted_m[0, valid_mask] - sorted_m[1, valid_mask]) / entropy[valid_mask]
    return float(np.mean(per_factor)), {'zero_entropy_factors': num_zero_entropy}


def _compute_mig_from_samples(
    codes: np.ndarray,
    factors: np.ndarray,
    num_bins: int = 20,
) -> tuple:
    """
    Compute the Mutual Information Gap (MIG) score.

    Follows the original disentanglement_lib implementation.  The only
    addition is that we also discretize the factors (the original assumes
    they are already discrete).

    Args:
        codes: Learned representations of shape (num_samples, num_codes).
        factors: Ground-truth factors of shape (num_samples, num_factors).
        num_bins: Number of bins for discretizing continuous values.

    Returns:
        Tuple of (mig_score, nan_info).
    """
    # Transpose to (num_dims, num_samples) format used by disentanglement_lib
    mus_train = codes.T   # (num_codes, num_samples)
    ys_train = factors.T  # (num_factors, num_samples)

    # Discretize codes (same as original)
    discretized_mus = histogram_discretize(mus_train, num_bins)

    # Discretize factors — the original assumes discrete factors;
    # we add this step so the metric works with continuous factors too.
    discretized_ys = histogram_discretize(ys_train, num_bins)

    return _compute_mig(discretized_mus, discretized_ys)


class MIGMetric(BaseMetric):
    """
    Mutual Information Gap (MIG) metric.
    
    MIG measures disentanglement by computing the normalized gap between
    the highest and second-highest mutual information values for each
    ground-truth factor. A factor is well-disentangled if it has high MI
    with exactly one code dimension.
    
    MIG = mean_j [(MI_top(j) - MI_second(j)) / H(factor_j)]
    
    Higher scores indicate better disentanglement/identifiability.
    
    Reference:
        Chen et al., "Isolating Sources of Disentanglement in VAEs", NeurIPS 2018.
        https://arxiv.org/abs/1802.04942
    
    Args:
        num_bins: Number of bins for discretizing continuous values.
    """
    
    def __init__(self, num_bins: int = 20):
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        self.num_bins = num_bins
    
    @property
    def required_min_samples(self) -> int:
        """MIG needs enough samples for reliable histogram binning."""
        return max(30, self.num_bins * 2)  # More permissive: 2x bins minimum
    
    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """Compute MIG score from samples."""
        mig_score, nan_info = _compute_mig_from_samples(
            codes=Z_hat,
            factors=Z,
            num_bins=self.num_bins,
        )
        # Clip for the MetricResult interface (original does not clip)
        mig_score = float(np.clip(mig_score, 0.0, 1.0))

        return MetricResult(
            primary_score=mig_score,
            subscores=None,
            metadata={
                'num_bins': self.num_bins,
                'num_factors': Z.shape[1],
                'num_codes': Z_hat.shape[1],
                'nan_info': nan_info,
            },
        )
    
    def compute_from_matrix(self, R: np.ndarray) -> MetricResult:
        """
        Compute MIG from a precomputed mutual information matrix.
        
        This allows computing MIG when you already have the MI matrix
        between codes and factors (e.g., from another metric computation).
        
        Args:
            R: MI matrix of shape (num_factors, num_codes) where R[j, i] is
               the mutual information between factor j and code i.
        
        Returns:
            MetricResult with MIG score.
        
        Note:
            This requires normalized MI values and factor entropies to be
            meaningful. For raw MI matrices, the score may not be in [0, 1].
        """
        # Transpose to (num_codes, num_factors) format
        mi_matrix = R.T
        
        num_factors = R.shape[0]
        
        # Sort MI values for each factor (descending)
        sorted_mi = np.sort(mi_matrix, axis=0)[::-1]
        
        # Compute gap (without entropy normalization since we don't have it)
        # This will give an unnormalized score
        mig_per_factor = np.zeros(num_factors)
        for j in range(num_factors):
            max_mi = sorted_mi[0, j]
            if max_mi > 1e-10:
                gap = sorted_mi[0, j] - sorted_mi[1, j] if sorted_mi.shape[0] > 1 else sorted_mi[0, j]
                mig_per_factor[j] = gap / max_mi  # Normalize by max instead of entropy
            else:
                mig_per_factor[j] = 0.0
        
        mig_score = float(np.clip(np.mean(mig_per_factor), 0.0, 1.0))
        
        return MetricResult(
            primary_score=mig_score,
            metadata={'computed_from_matrix': True},
        )
