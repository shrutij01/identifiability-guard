"""
Shared evaluation helper functions.

Provides reusable utilities for DGP/encoder evaluation across different scripts,
including class mappings, metric extraction, and common evaluation patterns.
"""

from typing import Dict, Any, Optional, Set
import warnings

import numpy as np

from ..dgp import (
    D1Independent, 
    D2Correlated, 
    D3SingleRedundant, 
    D4MultiRedundant,
)
from ..encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
    E4UndercompleteLinear,
    E5OvercompleteLinear,
    E6OvercompleteMulticodes,
    E7OvercompleteEntangled,
    E8OvercompleteDisjoint,
    E9RandomGaussian,
    E10RandomUniform,
)
from ..metrics import MetricRegistry


# Mapping of DGP names to classes
DGP_CLASSES = {
    'D1': D1Independent,
    'D2': D2Correlated,
    'D3': D3SingleRedundant,
    'D4': D4MultiRedundant,
}

# Mapping of encoder names to classes
ENCODER_CLASSES = {
    'E1': E1ElementwiseLinear,
    'E2': E2ElementwiseNonlinear,
    'E3': E3LinearlyEntangled,
    'E4': E4UndercompleteLinear,
    'E5': E5OvercompleteLinear,
    'E6': E6OvercompleteMulticodes,
    'E7': E7OvercompleteEntangled,
    'E8': E8OvercompleteDisjoint,
    'E9': E9RandomGaussian,
    'E10': E10RandomUniform,
}

# All available metrics with descriptions
ALL_METRICS = {
    'dci_disentanglement': 'DCI Disentanglement score',
    'dci_completeness': 'DCI Completeness score',
    'dci_informativeness': 'DCI Informativeness score',
    'mcc_pearson': 'MCC with Pearson correlation',
    'mcc_spearman': 'MCC with Spearman correlation',
    'mcc_rdc': 'MCC with Randomized Dependence Coefficient',
    'r2': 'R² coefficient of determination',
    'mig': 'Mutual Information Gap',
    'tmex': 'Testing for Measurement Exchangeability',
    'infom': 'InfoM (Modularity)',
    'infoe': 'InfoE (Explicitness)',
    'infoc': 'InfoC (Compactness)',
}

# Metric display names for visualization
METRIC_DISPLAY_NAMES = {
    'dci_disentanglement': 'DCI-D',
    'dci_completeness': 'DCI-C',
    'dci_informativeness': 'DCI-I',
    'mcc_pearson': 'MCC-P',
    'mcc_spearman': 'MCC-S',
    'mcc_rdc': 'MCC-RDC',
    'r2': 'R²',
    'mig': 'MIG',
    'tmex': 'T-MEX',
    'infom': 'InfoM',
    'infoe': 'InfoE',
    'infoc': 'InfoC',
}

# Default subset of metrics to track (for faster evaluation)
DEFAULT_METRICS = {
    'dci_disentanglement', 
    'dci_completeness', 
    'dci_informativeness', 
    'mcc_pearson',
    'mcc_spearman',
    'mcc_rdc',
    'r2',
    'mig',
    'tmex',
    'infom',
    'infoe',
    'infoc',
}


def get_dgp_class(name: str):
    """Get DGP class by name (e.g., 'D1', 'D2')."""
    if name not in DGP_CLASSES:
        raise ValueError(f"Unknown DGP: {name}. Available: {list(DGP_CLASSES.keys())}")
    return DGP_CLASSES[name]


def get_encoder_class(name: str):
    """Get encoder class by name (e.g., 'E1', 'E2')."""
    if name not in ENCODER_CLASSES:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODER_CLASSES.keys())}")
    return ENCODER_CLASSES[name]


def extract_metric_scores(
    all_results: Dict[str, Any],
    metrics_to_extract: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Extract metric scores from MetricRegistry.compute_all() results.
    
    Args:
        all_results: Dictionary from registry.compute_all(Z, Z_hat).
        metrics_to_extract: Set of metric names to extract. If None, extracts all.
    
    Returns:
        Dictionary of metric_name -> score. Missing metrics will have np.nan.
    """
    if metrics_to_extract is None:
        metrics_to_extract = set(ALL_METRICS.keys())
    
    results = {}
    
    # Initialize all requested metrics with NaN (will be overwritten if computed)
    for metric_name in metrics_to_extract:
        results[metric_name] = np.nan
    
    # DCI subscores
    dci_metrics = {'dci_disentanglement', 'dci_completeness', 'dci_informativeness'}
    if 'dci' in all_results and dci_metrics & metrics_to_extract:
        dci_result = all_results['dci']
        if 'dci_disentanglement' in metrics_to_extract and dci_result is not None:
            results['dci_disentanglement'] = dci_result.subscores.get('disentanglement', np.nan)
        if 'dci_completeness' in metrics_to_extract and dci_result is not None:
            results['dci_completeness'] = dci_result.subscores.get('completeness', np.nan)
        if 'dci_informativeness' in metrics_to_extract and dci_result is not None:
            results['dci_informativeness'] = dci_result.subscores.get('informativeness_test', np.nan)
    
    # MCC variants
    for mcc_type in ['mcc_pearson', 'mcc_spearman', 'mcc_rdc']:
        if mcc_type in metrics_to_extract and mcc_type in all_results and all_results[mcc_type] is not None:
            results[mcc_type] = all_results[mcc_type].primary_score
    
    # R²
    if 'r2' in metrics_to_extract and 'r2' in all_results and all_results['r2'] is not None:
        results['r2'] = all_results['r2'].primary_score
    
    # MIG
    if 'mig' in metrics_to_extract and 'mig' in all_results and all_results['mig'] is not None:
        results['mig'] = all_results['mig'].primary_score
    
    # T-MEX
    if 'tmex' in metrics_to_extract and 'tmex' in all_results and all_results['tmex'] is not None:
        results['tmex'] = all_results['tmex'].primary_score
    
    # InfoMEC metrics
    if 'infom' in metrics_to_extract and 'infom' in all_results and all_results['infom'] is not None:
        results['infom'] = all_results['infom'].primary_score
    if 'infoe' in metrics_to_extract and 'infoe' in all_results and all_results['infoe'] is not None:
        results['infoe'] = all_results['infoe'].primary_score
    if 'infoc' in metrics_to_extract and 'infoc' in all_results and all_results['infoc'] is not None:
        results['infoc'] = all_results['infoc'].primary_score
    
    return results


def validate_array(arr: np.ndarray, name: str = "array") -> None:
    """
    Validate that an array is suitable for metric computation.

    Raises ``ValueError`` if the array contains NaN, Inf, or extreme values
    that would cause numerical problems.

    Args:
        arr: Input array to validate.
        name: Descriptive name for error messages.

    Raises:
        ValueError: If the array contains NaN, Inf, or values outside float64
            representable range.
    """
    nan_count = int(np.sum(np.isnan(arr)))
    if nan_count > 0:
        raise ValueError(
            f"{name} contains {nan_count} NaN value(s). "
            f"This indicates a problem in the data generation or encoding pipeline."
        )

    inf_count = int(np.sum(np.isinf(arr)))
    if inf_count > 0:
        raise ValueError(
            f"{name} contains {inf_count} Inf value(s). "
            f"This indicates a numerical overflow in the data pipeline."
        )


def evaluate_combination(
    dgp_name: str,
    encoder_name: str,
    n_samples: int,
    n_factors: int,
    seed: int,
    dgp_kwargs: Optional[Dict[str, Any]] = None,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    metrics_to_compute: Optional[Set[str]] = None,
    registry: Optional[MetricRegistry] = None,
    sanitize_inputs: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a single DGP/encoder combination.
    
    This is the primary evaluation function used across different scripts.
    
    Args:
        dgp_name: DGP class name ('D1', 'D2', etc.)
        encoder_name: Encoder class name ('E1', 'E2', etc.)
        n_samples: Number of samples to generate
        n_factors: Number of latent factors
        seed: Random seed for reproducibility
        dgp_kwargs: Additional keyword arguments for DGP constructor
        encoder_kwargs: Additional keyword arguments for encoder constructor
        metrics_to_compute: Set of metric names to compute (default: all)
        registry: Optional pre-initialized MetricRegistry
        sanitize_inputs: If True, sanitize Z and Z_hat to prevent NaN/Inf issues
    
    Returns:
        Dictionary of metric scores.
    
    Example:
        >>> scores = evaluate_combination('D1', 'E1', n_samples=1000, n_factors=5, seed=42)
        >>> print(scores['mcc_pearson'])
    """
    if dgp_kwargs is None:
        dgp_kwargs = {}
    if encoder_kwargs is None:
        encoder_kwargs = {}
    
    # Create DGP
    dgp_cls = get_dgp_class(dgp_name)
    dgp = dgp_cls(d=n_factors, seed=seed, **dgp_kwargs)
    Z = dgp.sample(n_samples)
    
    # Create encoder
    encoder_cls = get_encoder_class(encoder_name)
    encoder = encoder_cls(d=n_factors, seed=seed, **encoder_kwargs)
    Z_hat = encoder.encode(Z)
    
    # Validate inputs — NaN/Inf in data is a pipeline bug, not something to
    # paper over.  BaseMetric._validate_samples does this too, but checking
    # early gives clearer error messages with the array name.
    if sanitize_inputs:
        validate_array(Z, "Z (ground-truth)")
        validate_array(Z_hat, "Z_hat (encoded)")
    
    # Compute metrics
    if registry is None:
        registry = MetricRegistry()
        registry.register_defaults()
    
    all_results = registry.compute_all(Z, Z_hat)
    
    # Extract scores
    return extract_metric_scores(all_results, metrics_to_compute)


def create_dgp_with_params(
    dgp_name: str,
    n_factors: int,
    seed: int,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Create a DGP instance with appropriate parameters based on DGP type.
    
    Args:
        dgp_name: DGP class name ('D1', 'D2', etc.)
        n_factors: Number of latent factors
        seed: Random seed
        params: Optional additional parameters (e.g., 'correlation' for D2)
    
    Returns:
        DGP instance
    """
    if params is None:
        params = {}
    
    dgp_cls = get_dgp_class(dgp_name)
    dgp_kwargs = {'d': n_factors, 'seed': seed}
    
    # Add DGP-specific parameters
    if dgp_name == 'D2' and 'correlation' in params:
        dgp_kwargs['correlation'] = params['correlation']
    if dgp_name == 'D4':
        if 'redundancy_strength' in params:
            dgp_kwargs['redundancy_strength'] = params['redundancy_strength']
        if 'r' in params:
            dgp_kwargs['r'] = params['r']
        else:
            # Default r=1 to avoid constraint issues
            dgp_kwargs['r'] = 1
    
    return dgp_cls(**dgp_kwargs)


def create_encoder_with_params(
    encoder_name: str,
    n_factors: int,
    seed: int,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Create an encoder instance with appropriate parameters based on encoder type.
    
    Args:
        encoder_name: Encoder class name ('E1', 'E2', etc.)
        n_factors: Number of latent factors
        seed: Random seed
        params: Optional additional parameters (e.g., 'nonlinearity_strength' for E2)
    
    Returns:
        Encoder instance
    """
    if params is None:
        params = {}
    
    encoder_cls = get_encoder_class(encoder_name)
    encoder_kwargs = {'d': n_factors, 'seed': seed}
    
    # Add encoder-specific parameters
    if encoder_name == 'E2' and 'nonlinearity_strength' in params:
        encoder_kwargs['nonlinearity_strength'] = params['nonlinearity_strength']
    if encoder_name == 'E3' and 'condition_number' in params:
        encoder_kwargs['condition_number'] = params['condition_number']
    if encoder_name == 'E4' and 'm' in params:
        encoder_kwargs['m'] = params['m']
    if encoder_name == 'E5' and 'm' in params:
        encoder_kwargs['m'] = params['m']
    if encoder_name == 'E6' and 'm' in params:
        encoder_kwargs['m'] = params['m']
    if encoder_name == 'E7':
        if 'condition_number' in params:
            encoder_kwargs['condition_number'] = params['condition_number']
        if 'm' in params:
            encoder_kwargs['m'] = params['m']
    if encoder_name == 'E8' and 'codes_per_factor' in params:
        encoder_kwargs['codes_per_factor'] = params['codes_per_factor']
    
    return encoder_cls(**encoder_kwargs)
