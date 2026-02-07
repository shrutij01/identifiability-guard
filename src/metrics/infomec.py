"""
InfoMEC (Information-theoretic Modularity, Explicitness, Compactness) metrics.

Based on the latent_quantization implementation:
MIT License - Copyright (c) 2023 Kyle Hsu

InfoM (Modularity): Measures how much each latent code specializes in a single factor.
InfoE (Explicitness): Measures how predictable factors are from the latent codes.
InfoC (Compactness): Measures how concentrated each factor's information is in few codes.

All metrics are based on Normalized Mutual Information (NMI) between factors and codes.
"""

import numpy as np
from sklearn import preprocessing, feature_selection, metrics, linear_model
from typing import Optional, Literal, Dict

from .base import BaseMetric, MetricResult


def _process_sources(sources: np.ndarray, num_bins: int = 20) -> np.ndarray:
    """
    Process source factors by label encoding or discretizing each dimension.
    
    Args:
        sources: Array of shape (n, num_sources).
        num_bins: Number of bins for discretizing continuous sources.
    
    Returns:
        Label-encoded/discretized sources of same shape.
    """
    processed_sources = []
    for i in range(sources.shape[1]):
        col = sources[:, i]
        unique_vals = np.unique(col)
        
        # If already discrete (few unique values), use label encoding
        if len(unique_vals) <= num_bins:
            processed_sources.append(
                preprocessing.LabelEncoder().fit_transform(col)
            )
        else:
            # Discretize continuous sources using histogram binning
            _, bin_edges = np.histogram(col, bins=num_bins)
            discretized = np.digitize(col, bin_edges[:-1])
            processed_sources.append(discretized)
    
    return np.stack(processed_sources, axis=1)


def _compute_nmi_matrix(
    sources: np.ndarray,
    latents: np.ndarray,
    discrete_latents: bool = False,
    n_neighbors: int = 10,
    num_bins: int = 20,
) -> np.ndarray:
    """
    Compute Normalized Mutual Information matrix between sources and latents.
    
    Args:
        sources: Ground-truth factors of shape (n, num_sources).
        latents: Learned codes of shape (n, num_latents).
        discrete_latents: Whether latents are discrete (vs continuous).
        n_neighbors: Number of neighbors for MI estimation with continuous latents.
        num_bins: Number of bins for discretizing continuous sources.
    
    Returns:
        NMI matrix of shape (num_sources, num_latents), normalized by source entropy.
    """
    # Process sources (discretize if continuous)
    processed_sources = _process_sources(sources, num_bins=num_bins)
    
    # Process latents based on whether they're discrete
    # Following original: per-column LabelEncoder (discrete) or per-column StandardScaler (continuous)
    processed_latents = []
    if discrete_latents:
        for j in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.LabelEncoder().fit_transform(latents[:, j])
            )
        processed_latents = np.stack(processed_latents, axis=1)
    else:
        for j in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.StandardScaler().fit_transform(
                    latents[:, j][:, None]
                )
            )
        processed_latents = np.concatenate(processed_latents, axis=1)
    
    num_sources = sources.shape[1]
    num_latents = latents.shape[1]
    
    nmi = np.empty((num_sources, num_latents))
    
    for i in range(num_sources):
        # Compute entropy of source i (for normalization)
        entropy_i = metrics.mutual_info_score(
            processed_sources[:, i], processed_sources[:, i]
        )
        
        for j in range(num_latents):
            if discrete_latents:
                mi_ij = metrics.mutual_info_score(
                    processed_sources[:, i], processed_latents[:, j]
                )
            else:
                # try:
                mi_ij = feature_selection.mutual_info_classif(
                    processed_latents[:, j][:, None],
                    processed_sources[:, i],
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                )[0]
                # except Exception:
                #     # Fallback: use discrete MI after discretizing latents
                #     _, bin_edges = np.histogram(processed_latents[:, j], bins=num_bins)
                #     discretized_latent = np.digitize(processed_latents[:, j], bin_edges[:-1])
                #     mi_ij = metrics.mutual_info_score(
                #         processed_sources[:, i], discretized_latent
                #     )
            
            # Normalize by source entropy
            nmi[i, j] = mi_ij / entropy_i if entropy_i > 0 else 0.0
    
    return nmi


def _logistic_regression_entropy(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute conditional entropy H(Y|X) using logistic regression.
    
    Args:
        X: Features of shape (n, d).
        y: Discrete labels of shape (n,).
    
    Returns:
        Cross-entropy loss (approximation of conditional entropy).
    """
    import warnings
    
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.int32, np.int64]
        
    model = linear_model.LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=100,  # Original uses 100, increased for convergence
        n_jobs=-1,
    )
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')
            model.fit(X, y)
        y_pred = model.predict_proba(X)
        return metrics.log_loss(y, y_pred)
    except Exception:
        # Fallback: return NaN so the caller can exclude this factor
        # from averaging rather than silently inflating the score.
        return np.nan


def _compute_infoe(
    sources: np.ndarray,
    latents: np.ndarray,
    discrete_latents: bool = False,
) -> tuple:
    """
    Compute InfoE (Explicitness) score.
    
    InfoE measures how well the ground-truth factors can be predicted from
    the latent codes using logistic regression. Higher values indicate
    that factors are more explicitly represented.
    
    Args:
        sources: Ground-truth factors of shape (n, num_sources).
        latents: Learned codes of shape (n, num_latents).
        discrete_latents: Whether latents are discrete.
    
    Returns:
        Tuple of (infoe_score, nan_info) where nan_info tracks NaN counts.
    """
    import warnings as _warnings

    normalized_predictive_information = []
    processed_sources = _process_sources(sources)
    
    if discrete_latents:
        try:
            processed_latents = preprocessing.OneHotEncoder(
                sparse_output=False
            ).fit_transform(latents)
        except TypeError:
            # Older sklearn versions
            processed_latents = preprocessing.OneHotEncoder(
                sparse=False
            ).fit_transform(latents)
    else:
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)
    
    nan_entropy_count = 0

    for i in range(processed_sources.shape[1]):
        # Conditional entropy H(S_i | Z)
        h_si_given_z = _logistic_regression_entropy(
            processed_latents, processed_sources[:, i]
        )
        
        # Marginal entropy H(S_i) using null predictor
        null_features = np.zeros_like(processed_latents)
        h_si = _logistic_regression_entropy(null_features, processed_sources[:, i])
        
        # If either entropy is NaN (logistic regression failed), skip this factor
        if np.isnan(h_si_given_z) or np.isnan(h_si):
            nan_entropy_count += 1
            normalized_predictive_information.append(np.nan)
            continue
        
        # Normalized predictive information: I(S_i; Z) / H(S_i) = (H(S_i) - H(S_i|Z)) / H(S_i)
        if h_si > 0:
            npi = (h_si - h_si_given_z) / h_si
            npi = max(0.0, min(1.0, npi))  # Clip to [0, 1]
        else:
            npi = 0.0
        
        normalized_predictive_information.append(npi)
    
    if nan_entropy_count > 0:
        _warnings.warn(
            f"InfoE: logistic regression failed for {nan_entropy_count} factor(s); "
            f"they are excluded from the average."
        )

    npi_array = np.array(normalized_predictive_information)
    valid = npi_array[~np.isnan(npi_array)]
    score = float(np.mean(valid)) if len(valid) > 0 else 0.0

    nan_info = {
        'logistic_regression_failures': nan_entropy_count,
        'factors_used': int(len(valid)),
        'factors_total': int(len(normalized_predictive_information)),
    }
    return score, nan_info


def _compute_infomec(
    sources: np.ndarray,
    latents: np.ndarray,
    discrete_latents: bool = False,
    n_neighbors: int = 10,
) -> Dict[str, float]:
    """
    Compute all InfoMEC metrics.
        
    Args:
        sources: Ground-truth factors of shape (n, num_sources).
        latents: Learned codes of shape (n, num_latents).
        discrete_latents: Whether latents are discrete.
        n_neighbors: Number of neighbors for MI estimation.
    
    Returns:
        Dictionary with keys 'infom', 'infoe', 'infoc', 'nmi', 'active_latents'.
    """
    # Compute NMI matrix (shape: num_sources x num_latents)
    nmi = _compute_nmi_matrix(
        sources, latents, discrete_latents, n_neighbors
    )
    
    # Determine active latents (with non-trivial range)
    # This follows the original implementation exactly
    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)
    if discrete_latents:
        active_latents = latent_ranges > 0
    else:
        active_latents = latent_ranges > np.max(latent_ranges) / 20
    
    num_sources = sources.shape[1]
    num_active_latents = np.sum(active_latents)
    
    # Use only active latents for InfoM/C computation
    pruned_nmi = nmi[:, active_latents] if num_active_latents > 0 else nmi
    
    # Handle edge cases
    if pruned_nmi.size == 0 or num_active_latents == 0:
        infoe_score, infoe_nan_info = _compute_infoe(sources, latents, discrete_latents)
        return {
            'infom': 0.0,
            'infoe': infoe_score,
            'infoc': 0.0,
            'nmi': nmi,
            'active_latents': active_latents,
            'nan_info': {
                'infom_zero_sum_latents': 0,
                'infoc_zero_sum_factors': 0,
                'infoe': infoe_nan_info,
                'edge_case': 'no_active_latents',
            },
        }
    
    # # Original implementation formulas:
    # infom = (np.mean(np.max(pruned_nmi, axis=0) / np.sum(pruned_nmi, axis=0)) - 1 / num_sources) / (1 - 1 / num_sources)
    # infoc = (np.mean(np.max(pruned_nmi, axis=1) / np.sum(pruned_nmi, axis=1)) - 1 / num_active_latents) / (1 - 1 / num_active_latents)

    
    # InfoM: Modularity - each code should specialize in one factor
    # Exclude latents with zero total NMI (they carry no information)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_sums = np.sum(pruned_nmi, axis=0)
        valid_cols = col_sums > 0
        infom_zero_sum = int(np.sum(~valid_cols))
        
        if np.any(valid_cols):
            modularity_ratios = np.max(pruned_nmi[:, valid_cols], axis=0) / col_sums[valid_cols]
            if num_sources > 1:
                infom = (np.mean(modularity_ratios) - 1 / num_sources) / (1 - 1 / num_sources)
            else:
                infom = np.mean(modularity_ratios)
        else:
            infom = 0.0
        infom = float(np.clip(infom, 0.0, 1.0))
    
    # InfoC: Compactness - each factor should be concentrated in few codes
    # Exclude factors with zero total NMI (they have no information in any code)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = np.sum(pruned_nmi, axis=1)
        valid_rows = row_sums > 0
        infoc_zero_sum = int(np.sum(~valid_rows))
        
        if np.any(valid_rows):
            compactness_ratios = np.max(pruned_nmi[valid_rows, :], axis=1) / row_sums[valid_rows]
            if num_active_latents > 1:
                infoc = (np.mean(compactness_ratios) - 1 / num_active_latents) / (1 - 1 / num_active_latents)
            else:
                infoc = np.mean(compactness_ratios)
        else:
            infoc = 0.0
        infoc = float(np.clip(infoc, 0.0, 1.0))
    
    # InfoE: Explicitness - factors should be predictable from codes
    infoe_score, infoe_nan_info = _compute_infoe(sources, latents, discrete_latents)
    
    return {
        'infom': infom,
        'infoe': infoe_score,
        'infoc': infoc,
        'nmi': nmi,
        'active_latents': active_latents,
        'nan_info': {
            'infom_zero_sum_latents': infom_zero_sum,
            'infoc_zero_sum_factors': infoc_zero_sum,
            'infoe': infoe_nan_info,
        },
    }


class InfoMECMetric(BaseMetric):
    """
    InfoMEC metric computing Modularity, Explicitness, and Compactness.
    
    Based on Normalized Mutual Information between factors and codes:
    - InfoM: How much each code specializes in a single factor (modularity).
    - InfoE: How predictable factors are from codes (explicitness).
    - InfoC: How concentrated each factor's info is in few codes (compactness).
    
    Higher scores indicate better disentanglement/identifiability.
    
    Args:
        discrete_latents: Whether learned representations are discrete.
        n_neighbors: Number of neighbors for MI estimation (continuous case).
    """
    
    def __init__(
        self,
        discrete_latents: bool = False,
        n_neighbors: int = 10,
    ):
        self.discrete_latents = discrete_latents
        self.n_neighbors = n_neighbors
    
    @property
    def required_min_samples(self) -> int:
        """InfoMEC needs sufficient samples for reliable MI estimation."""
        return max(50, self.n_neighbors * 5)
    
    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """Compute InfoMEC scores from samples."""
        # Discretize Z for MI computation (treat as factors)
        # Round continuous factors to make them discrete for MI computation
        Z_discrete = np.round(Z * 10) / 10  # Simple discretization
        
        results = _compute_infomec(
            sources=Z_discrete,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
            n_neighbors=self.n_neighbors,
        )
        
        # Primary score is average of the three
        primary_score = (results['infom'] + results['infoe'] + results['infoc']) / 3
        
        return MetricResult(
            primary_score=float(primary_score),
            subscores={
                'modularity': results['infom'],
                'explicitness': results['infoe'],
                'compactness': results['infoc'],
            },
            metadata={
                'discrete_latents': self.discrete_latents,
                'num_active_latents': int(np.sum(results['active_latents'])),
                'nan_info': results.get('nan_info', {}),
            },
        )


class InfoMMetric(BaseMetric):
    """
    InfoM (Modularity) metric.
    
    Measures how much each latent code specializes in representing a single
    ground-truth factor. Based on NMI matrix between factors and codes.
    
    Higher scores indicate better modularity (each code captures one factor).
    """
    
    def __init__(
        self,
        discrete_latents: bool = False,
        n_neighbors: int = 10,
    ):
        self.discrete_latents = discrete_latents
        self.n_neighbors = n_neighbors
    
    @property
    def required_min_samples(self) -> int:
        return max(50, self.n_neighbors * 5)
    
    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        Z_discrete = np.round(Z * 10) / 10
        results = _compute_infomec(
            sources=Z_discrete,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
            n_neighbors=self.n_neighbors,
        )
        return MetricResult(
            primary_score=float(results['infom']),
            metadata={'nan_info': results.get('nan_info', {})},
        )


class InfoEMetric(BaseMetric):
    """
    InfoE (Explicitness) metric.
    
    Measures how well ground-truth factors can be predicted from the latent
    codes using logistic regression. Based on normalized predictive information.
    
    Higher scores indicate factors are more explicitly represented.
    """
    
    def __init__(
        self,
        discrete_latents: bool = False,
    ):
        self.discrete_latents = discrete_latents
    
    @property
    def required_min_samples(self) -> int:
        return 50
    
    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        Z_discrete = np.round(Z * 10) / 10
        infoe, nan_info = _compute_infoe(
            sources=Z_discrete,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
        )
        return MetricResult(
            primary_score=float(infoe),
            metadata={'nan_info': nan_info},
        )


class InfoCMetric(BaseMetric):
    """
    InfoC (Compactness) metric.
    
    Measures how concentrated each factor's information is across the latent
    codes. A factor is compact if its information is in few codes.
    
    Higher scores indicate better compactness (factors map to few codes).
    """
    
    def __init__(
        self,
        discrete_latents: bool = False,
        n_neighbors: int = 10,
    ):
        self.discrete_latents = discrete_latents
        self.n_neighbors = n_neighbors
    
    @property
    def required_min_samples(self) -> int:
        return max(50, self.n_neighbors * 5)
    
    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        Z_discrete = np.round(Z * 10) / 10
        results = _compute_infomec(
            sources=Z_discrete,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
            n_neighbors=self.n_neighbors,
        )
        return MetricResult(
            primary_score=float(results['infoc']),
            metadata={'nan_info': results.get('nan_info', {})},
        )
