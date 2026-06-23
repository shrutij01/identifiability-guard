"""
Mean Correlation Coefficient (MCC) for identifiability evaluation.

MCC measures coordinate-wise identifiability by finding the optimal
permutation that maximizes mean absolute correlation between ground-truth
factors Z (n, d) and learned codes Z_hat (n, m).

Variants
--------
**Via MCCMetric (recommended)**::

    from identifiability_guard.metrics import MCC

    # Legacy in-sample MCC (default)
    MCC().compute(Z, Z_hat)

    # Coverage normalization — divides by d, not min(d,m).
    # Penalizes unmatched factors when d > m (undercomplete).
    MCC(normalization="coverage").compute(Z, Z_hat)

    # K-fold cross-fitted MCC — removes in-sample selection bias.
    # Essential when m >> d (overcomplete) because legacy MCC inflates.
    MCC(crossfit=True, seed=0).compute(Z, Z_hat)

    # Cross-fitted + coverage (strictest)
    MCC(crossfit=True, normalization="coverage", seed=0).compute(Z, Z_hat)

    # Spearman rank correlation instead of Pearson
    MCC(method="spearman").compute(Z, Z_hat)

    # From a precomputed absolute correlation matrix
    MCC().compute_from_matrix(np.abs(R))

**Via low-level functions**::

    from identifiability_guard.metrics.mcc import (
        cross_correlation_np,       # fast d×m correlation via BLAS matmul
        cross_correlation_pt,       # same, on GPU via cuBLAS
        legacy_mcc_np,              # in-sample absolute MCC
        mcc_train_test_np,          # single train/test split with sign correction
        mean_corr_coef_crossfit_np, # K-fold cross-fitted MCC
        permutation_null_np,        # null calibration via row permutations
        sinkhorn_rectangular_plan_pt, # soft matching for d > 10K
        sinkhorn_soft_mcc_pt,       # soft Sinkhorn MCC (GPU, differentiable)
    )

Dimension handling
------------------
- d == m: standard square assignment.
- d < m (overcomplete): all d factors matched; extra codes ignored.
  Legacy and coverage scores coincide.
- d > m (undercomplete): only m of d factors matched.
  Legacy divides by m; coverage divides by d (unmatched = zero).
"""

# MCCMetric class
from ._metric import MCCMetric

# Legacy functions (backward-compatible public API)
from ._legacy import (
    auction_linear_assignment,
    corrcoef_pt,
    cov_pt,
    cross_correlation_pt,
    mean_corr_coef,
    mean_corr_coef_np,
    mean_corr_coef_out_of_sample,
    mean_corr_coef_pt,
    rankdata_pt,
    rdc,
    spearmanr_pt,
)

# Cross-fitted MCC and null calibration
from ._crossfit import (
    CrossfitMCCResult,
    NullCalibration,
    cross_correlation_np,
    legacy_mcc_np,
    make_kfold_splits,
    mcc_train_test_np,
    mean_corr_coef_crossfit_np,
    permutation_null_np,
)

# Sinkhorn (PyTorch, GPU)
from ._sinkhorn import (
    sinkhorn_rectangular_plan_pt,
    sinkhorn_soft_mcc_pt,
)

__all__ = [
    "MCCMetric",
    # Legacy
    "auction_linear_assignment",
    "corrcoef_pt",
    "cov_pt",
    "cross_correlation_pt",
    "mean_corr_coef",
    "mean_corr_coef_np",
    "mean_corr_coef_out_of_sample",
    "mean_corr_coef_pt",
    "rankdata_pt",
    "rdc",
    "spearmanr_pt",
    # Crossfit
    "CrossfitMCCResult",
    "NullCalibration",
    "cross_correlation_np",
    "legacy_mcc_np",
    "make_kfold_splits",
    "mcc_train_test_np",
    "mean_corr_coef_crossfit_np",
    "permutation_null_np",
    # Sinkhorn
    "sinkhorn_rectangular_plan_pt",
    "sinkhorn_soft_mcc_pt",
]
