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
from ._numerical import EPS, sanitize_mi, clamp


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
    random_state: Optional[int] = None,
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
    if discrete_latents:
        processed_latents = []
        for j in range(latents.shape[1]):
            processed_latents.append(
                preprocessing.LabelEncoder().fit_transform(latents[:, j])
            )
        processed_latents = np.stack(processed_latents, axis=1)
    else:
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)

    num_sources = sources.shape[1]
    num_latents = latents.shape[1]

    nmi = np.empty((num_sources, num_latents))

    for i in range(num_sources):
        entropy_i = metrics.mutual_info_score(
            processed_sources[:, i], processed_sources[:, i]
        )

        if entropy_i < EPS:
            nmi[i, :] = 0.0
            continue

        if discrete_latents:
            for j in range(num_latents):
                mi_ij = metrics.mutual_info_score(
                    processed_sources[:, i], processed_latents[:, j]
                )
                mi_ij = sanitize_mi(mi_ij)
                nmi[i, j] = clamp(mi_ij / entropy_i)
        else:
            mi_row = feature_selection.mutual_info_classif(
                processed_latents,
                processed_sources[:, i],
                discrete_features=False,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            mi_row = np.array([sanitize_mi(v) for v in mi_row], dtype=float)
            nmi[i, :] = np.clip(mi_row / entropy_i, 0.0, 1.0)

    return nmi


def _logistic_regression_entropy(
    X: np.ndarray,
    y: np.ndarray,
    penalty=None,
    C: float = 1.0,
    random_state: Optional[int] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> float:
    """
    Compute conditional entropy H(Y|X) using logistic regression.

    When *X_test* and *y_test* are provided the model is fit on (X, y) and
    the log-loss is evaluated on the held-out (X_test, y_test).  Otherwise
    the in-sample loss is returned (original behaviour).

    Args:
        X: Features of shape (n, d) – training data.
        y: Discrete labels of shape (n,) – training labels.
        penalty: Regularization penalty (None, 'l2', etc.).
        C: Inverse regularization strength (only used when penalty is not None).
        random_state: Random seed for reproducibility.
        X_test: Optional held-out features of shape (n_test, d).
        y_test: Optional held-out labels of shape (n_test,).

    Returns:
        Cross-entropy loss (approximation of conditional entropy).
    """
    import warnings

    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.int32, np.int64]

    kwargs = dict(
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=100,
    )
    if penalty is not None:
        kwargs["l1_ratio"] = 0 if penalty == "l2" else 1
        kwargs["C"] = C

    kwargs["random_state"] = random_state
    model = linear_model.LogisticRegression(**kwargs)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings(
                "ignore", message=".*lbfgs failed to converge.*"
            )
            model.fit(X, y)

        # Evaluate on held-out test set when provided
        if X_test is not None and y_test is not None:
            y_pred = model.predict_proba(X_test)
            return metrics.log_loss(y_test, y_pred)

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
    random_state: Optional[int] = None,
    sources_test: Optional[np.ndarray] = None,
    latents_test: Optional[np.ndarray] = None,
) -> tuple:
    """
    Compute InfoE (Explicitness) score.

    InfoE measures how well the ground-truth factors can be predicted from
    the latent codes using logistic regression. Higher values indicate
    that factors are more explicitly represented.

    When *sources_test* and *latents_test* are provided, logistic regression is
    fit on (sources, latents) and evaluated on the held-out test arrays.

    Args:
        sources: Ground-truth factors of shape (n, num_sources).
        latents: Learned codes of shape (n, num_latents).
        discrete_latents: Whether latents are discrete.
        random_state: Random seed for reproducibility.
        sources_test: Optional held-out ground-truth factors.
        latents_test: Optional held-out learned codes.

    Returns:
        Tuple of (infoe_score, nan_info) where nan_info tracks NaN counts.
    """
    import warnings as _warnings

    oos = sources_test is not None and latents_test is not None

    normalized_predictive_information = []
    processed_sources = _process_sources(sources)

    if discrete_latents:
        try:
            encoder = preprocessing.OneHotEncoder(sparse_output=False)
        except TypeError:
            encoder = preprocessing.OneHotEncoder(sparse=False)
        processed_latents = encoder.fit_transform(latents)
        if oos:
            processed_latents_test = encoder.transform(latents_test)
    else:
        scaler = preprocessing.StandardScaler().fit(latents)
        processed_latents = scaler.transform(latents)
        if oos:
            processed_latents_test = scaler.transform(latents_test)

    if oos:
        processed_sources_test = _process_sources(sources_test)

    nan_entropy_count = 0

    for i in range(processed_sources.shape[1]):
        # Build test kwargs for _logistic_regression_entropy
        lr_test_kwargs: Dict = {}
        if oos:
            lr_test_kwargs["X_test"] = processed_latents_test
            lr_test_kwargs["y_test"] = processed_sources_test[:, i]

        # Conditional entropy H(S_i | Z)
        h_si_given_z = _logistic_regression_entropy(
            processed_latents, processed_sources[:, i],
            random_state=random_state,
            **lr_test_kwargs,
        )

        # Marginal entropy H(S_i) — use test labels when doing OOS
        if oos:
            labels_i = processed_sources_test[:, i]
        else:
            labels_i = processed_sources[:, i]
        _, counts = np.unique(labels_i, return_counts=True)
        probs = counts / counts.sum()
        h_si = float(-np.sum(probs * np.log(probs + EPS)))

        if np.isnan(h_si_given_z) or np.isnan(h_si):
            nan_entropy_count += 1
            normalized_predictive_information.append(np.nan)
            continue

        # Normalized predictive information: I(S_i; Z) / H(S_i) = (H(S_i) - H(S_i|Z)) / H(S_i)
        if h_si < EPS:
            npi = 0.0
        else:
            npi = (h_si - h_si_given_z) / h_si
            npi = max(0.0, min(1.0, npi))  # Clip to [0, 1]

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
        "logistic_regression_failures": nan_entropy_count,
        "factors_used": int(len(valid)),
        "factors_total": int(len(normalized_predictive_information)),
    }
    return score, nan_info


def _compute_infomec(
    sources: np.ndarray,
    latents: np.ndarray,
    discrete_latents: bool = False,
    n_neighbors: int = 10,
    compute_infoe: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute all InfoMEC metrics.

    Args:
        sources: Ground-truth factors of shape (n, num_sources).
        latents: Learned codes of shape (n, num_latents).
        discrete_latents: Whether latents are discrete.
        n_neighbors: Number of neighbors for MI estimation.
        compute_infoe: Whether to compute InfoE (expensive). If False, infoe=0.0.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'infom', 'infoe', 'infoc', 'nmi', 'active_latents'.
    """
    # Compute NMI matrix (shape: num_sources x num_latents)
    nmi = _compute_nmi_matrix(
        sources, latents, discrete_latents, n_neighbors,
        random_state=random_state,
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
        if compute_infoe:
            infoe_score, infoe_nan_info = _compute_infoe(
                sources, latents, discrete_latents,
                random_state=random_state,
            )
        else:
            infoe_score, infoe_nan_info = 0.0, {}
        return {
            "infom": 0.0,
            "infoe": infoe_score,
            "infoc": 0.0,
            "nmi": nmi,
            "active_latents": active_latents,
            "nan_info": {
                "infom_zero_sum_latents": 0,
                "infoc_zero_sum_factors": 0,
                "infoe": infoe_nan_info,
                "edge_case": "no_active_latents",
            },
        }

    # # Original implementation formulas:
    # infom = (np.mean(np.max(pruned_nmi, axis=0) / np.sum(pruned_nmi, axis=0)) - 1 / num_sources) / (1 - 1 / num_sources)
    # infoc = (np.mean(np.max(pruned_nmi, axis=1) / np.sum(pruned_nmi, axis=1)) - 1 / num_active_latents) / (1 - 1 / num_active_latents)

    # InfoM: Modularity - each code should specialize in one factor
    # Exclude latents with zero total NMI (they carry no information)
    with np.errstate(divide="ignore", invalid="ignore"):
        col_sums = np.sum(pruned_nmi, axis=0)
        valid_cols = col_sums > 0
        infom_zero_sum = int(np.sum(~valid_cols))

        if np.any(valid_cols):
            modularity_ratios = (
                np.max(pruned_nmi[:, valid_cols], axis=0)
                / col_sums[valid_cols]
            )
            if num_sources > 1:
                infom = (np.mean(modularity_ratios) - 1 / num_sources) / (
                    1 - 1 / num_sources
                )
            else:
                infom = np.mean(modularity_ratios)
        else:
            infom = 0.0
        infom = float(np.clip(infom, 0.0, 1.0))

    # InfoC: Compactness - each factor should be concentrated in few codes
    # Exclude factors with zero total NMI (they have no information in any code)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = np.sum(pruned_nmi, axis=1)
        valid_rows = row_sums > 0
        infoc_zero_sum = int(np.sum(~valid_rows))

        if np.any(valid_rows):
            compactness_ratios = (
                np.max(pruned_nmi[valid_rows, :], axis=1)
                / row_sums[valid_rows]
            )
            if num_active_latents > 1:
                infoc = (
                    np.mean(compactness_ratios) - 1 / num_active_latents
                ) / (1 - 1 / num_active_latents)
            else:
                infoc = np.mean(compactness_ratios)
        else:
            infoc = 0.0
        infoc = float(np.clip(infoc, 0.0, 1.0))

    # InfoE: Explicitness - factors should be predictable from codes
    if compute_infoe:
        infoe_score, infoe_nan_info = _compute_infoe(
            sources, latents, discrete_latents,
            random_state=random_state,
        )
    else:
        infoe_score, infoe_nan_info = 0.0, {}

    return {
        "infom": infom,
        "infoe": infoe_score,
        "infoc": infoc,
        "nmi": nmi,
        "active_latents": active_latents,
        "nan_info": {
            "infom_zero_sum_latents": infom_zero_sum,
            "infoc_zero_sum_factors": infoc_zero_sum,
            "infoe": infoe_nan_info,
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
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        discrete_latents: bool = False,
        n_neighbors: int = 10,
        random_state: Optional[int] = None,
    ):
        self.discrete_latents = discrete_latents
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    @property
    def required_min_samples(self) -> int:
        """InfoMEC needs sufficient samples for reliable MI estimation."""
        return max(50, self.n_neighbors * 5)

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """Compute InfoMEC scores from samples."""
        results = _compute_infomec(
            sources=Z,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )

        primary_score = (
            results["infom"] + results["infoe"] + results["infoc"]
        ) / 3

        return MetricResult(
            primary_score=float(primary_score),
            subscores={
                "modularity": results["infom"],
                "explicitness": results["infoe"],
                "compactness": results["infoc"],
            },
            metadata={
                "discrete_latents": self.discrete_latents,
                "num_active_latents": int(np.sum(results["active_latents"])),
                "nan_info": results.get("nan_info", {}),
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
        random_state: Optional[int] = None,
    ):
        self.discrete_latents = discrete_latents
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    @property
    def required_min_samples(self) -> int:
        return max(50, self.n_neighbors * 5)

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        results = _compute_infomec(
            sources=Z,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
            n_neighbors=self.n_neighbors,
            compute_infoe=False,
            random_state=self.random_state,
        )
        return MetricResult(
            primary_score=float(results["infom"]),
            metadata={"nan_info": results.get("nan_info", {})},
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
        random_state: Optional[int] = None,
    ):
        self.discrete_latents = discrete_latents
        self.random_state = random_state

    @property
    def required_min_samples(self) -> int:
        return 50

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        infoe, nan_info = _compute_infoe(
            sources=Z,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
            random_state=self.random_state,
        )
        return MetricResult(
            primary_score=float(infoe),
            metadata={"nan_info": nan_info},
        )

    def compute_oos(
        self,
        Z_train: np.ndarray,
        Z_hat_train: np.ndarray,
        Z_test: np.ndarray,
        Z_hat_test: np.ndarray,
    ) -> MetricResult:
        """Fit logistic regression on train, evaluate InfoE on held-out test."""
        self._validate_samples(Z_train, Z_hat_train)
        self._validate_samples(Z_test, Z_hat_test)
        infoe, nan_info = _compute_infoe(
            sources=Z_train,
            latents=Z_hat_train,
            discrete_latents=self.discrete_latents,
            random_state=self.random_state,
            sources_test=Z_test,
            latents_test=Z_hat_test,
        )
        nan_info["oos"] = True
        result = MetricResult(
            primary_score=float(infoe),
            metadata={"nan_info": nan_info},
        )
        self._validate_result_type(result)
        self._validate_result_range(result)
        return result


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
        random_state: Optional[int] = None,
    ):
        self.discrete_latents = discrete_latents
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    @property
    def required_min_samples(self) -> int:
        return max(50, self.n_neighbors * 5)

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        results = _compute_infomec(
            sources=Z,
            latents=Z_hat,
            discrete_latents=self.discrete_latents,
            n_neighbors=self.n_neighbors,
            compute_infoe=False,
            random_state=self.random_state,
        )
        return MetricResult(
            primary_score=float(results["infoc"]),
            metadata={"nan_info": results.get("nan_info", {})},
        )
