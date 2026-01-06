"""
Multi-seed evaluation utilities.

Provides functions to run evaluations over multiple random seeds and
compute statistics (mean, std, confidence intervals) over the results.
"""

from typing import Callable, Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats


def run_with_seeds(
    evaluation_fn: Callable[[int], Dict[str, float]],
    seeds: List[int],
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Run evaluation function with multiple seeds and collect results.
    
    Args:
        evaluation_fn: Function that takes a seed and returns a dict of metrics.
        seeds: List of random seeds to use.
        verbose: If True, prints progress.
        
    Returns:
        Dictionary mapping metric names to lists of values (one per seed).
        
    Example:
        >>> def eval_fn(seed):
        ...     dgp = D1Independent(d=5, seed=seed)
        ...     Z = dgp.sample(1000)
        ...     encoder = E1ElementwiseLinear(d=5, seed=seed)
        ...     Z_hat = encoder.encode(Z)
        ...     return {"mcc": compute_mcc(Z, Z_hat)}
        >>> results = run_with_seeds(eval_fn, seeds=[42, 43, 44])
    """
    # Initialize results dict
    all_results: Dict[str, List[float]] = {}
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"Running with seed {seed} ({i+1}/{len(seeds)})...")
        
        try:
            result = evaluation_fn(seed)
            
            # Add results to collection
            for metric_name, value in result.items():
                if metric_name not in all_results:
                    all_results[metric_name] = []
                all_results[metric_name].append(value)
        
        except Exception as e:
            if verbose:
                print(f"  Warning: Evaluation failed for seed {seed}: {e}")
            # Add NaN for failed runs
            if all_results:
                for metric_name in all_results.keys():
                    all_results[metric_name].append(np.nan)
    
    return all_results


def compute_statistics(
    values: List[float],
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compute statistics over a list of values.
    
    Args:
        values: List of numeric values.
        confidence_level: Confidence level for confidence intervals (default: 0.95).
        
    Returns:
        Dictionary with keys: mean, std, sem, ci_lower, ci_upper, min, max, median.
        
    Example:
        >>> values = [0.85, 0.87, 0.83, 0.86, 0.84]
        >>> stats = compute_statistics(values)
        >>> print(f"Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
    """
    values_array = np.array(values)
    
    # Filter out NaN values for statistics
    valid_values = values_array[~np.isnan(values_array)]
    
    if len(valid_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'sem': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'n_valid': 0,
            'n_total': len(values),
        }
    
    mean = np.mean(valid_values)
    std = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
    sem = std / np.sqrt(len(valid_values)) if len(valid_values) > 1 else 0.0
    
    # Compute confidence interval using t-distribution
    if len(valid_values) > 1:
        ci = stats.t.interval(
            confidence_level,
            df=len(valid_values) - 1,
            loc=mean,
            scale=sem,
        )
        ci_lower, ci_upper = ci
    else:
        ci_lower = ci_upper = mean
    
    return {
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'min': np.min(valid_values),
        'max': np.max(valid_values),
        'median': np.median(valid_values),
        'n_valid': len(valid_values),
        'n_total': len(values),
    }


def aggregate_results(
    results: Dict[str, List[float]],
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for all metrics in a results dictionary.
    
    Args:
        results: Dictionary mapping metric names to lists of values.
        confidence_level: Confidence level for confidence intervals.
        
    Returns:
        Dictionary mapping metric names to statistics dictionaries.
        
    Example:
        >>> results = {"mcc": [0.85, 0.87, 0.83], "dci": [0.92, 0.91, 0.93]}
        >>> aggregated = aggregate_results(results)
        >>> print(f"MCC: {aggregated['mcc']['mean']:.3f} ± {aggregated['mcc']['std']:.3f}")
    """
    aggregated = {}
    
    for metric_name, values in results.items():
        aggregated[metric_name] = compute_statistics(values, confidence_level)
    
    return aggregated


def run_multi_seed_evaluation(
    evaluation_fn: Callable[[int], Dict[str, float]],
    n_seeds: int = 10,
    base_seed: int = 42,
    confidence_level: float = 0.95,
    verbose: bool = True,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """
    Run evaluation over multiple seeds and return raw and aggregated results.
    
    Args:
        evaluation_fn: Function that takes a seed and returns a dict of metrics.
        n_seeds: Number of seeds to use.
        base_seed: Starting seed value.
        confidence_level: Confidence level for confidence intervals.
        verbose: If True, prints progress and summary.
        
    Returns:
        Tuple of (raw_results, aggregated_results).
        raw_results: Dict mapping metric names to lists of values.
        aggregated_results: Dict mapping metric names to statistics dicts.
        
    Example:
        >>> def eval_fn(seed):
        ...     # ... evaluation code ...
        ...     return {"mcc": score}
        >>> raw, agg = run_multi_seed_evaluation(eval_fn, n_seeds=5)
        >>> print(f"MCC: {agg['mcc']['mean']:.3f} ± {agg['mcc']['std']:.3f}")
    """
    # Generate seeds
    seeds = [base_seed + i for i in range(n_seeds)]
    
    if verbose:
        print(f"Running evaluation with {n_seeds} seeds...")
        print(f"Seeds: {seeds}")
    
    # Run evaluations
    raw_results = run_with_seeds(evaluation_fn, seeds, verbose=verbose)
    
    # Compute statistics
    aggregated_results = aggregate_results(raw_results, confidence_level)
    
    if verbose:
        print("\nAggregated Results:")
        print("=" * 60)
        for metric_name, stats_dict in aggregated_results.items():
            mean = stats_dict['mean']
            std = stats_dict['std']
            ci_lower = stats_dict['ci_lower']
            ci_upper = stats_dict['ci_upper']
            print(f"{metric_name:20s}: {mean:.4f} ± {std:.4f}")
            print(f"{'':20s}  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("=" * 60)
    
    return raw_results, aggregated_results


def format_result_with_ci(
    mean: float,
    ci_lower: float,
    ci_upper: float,
    precision: int = 3,
) -> str:
    """
    Format a result with confidence interval as a string.
    
    Args:
        mean: Mean value.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        precision: Number of decimal places.
        
    Returns:
        Formatted string like "0.850 [0.830, 0.870]".
        
    Example:
        >>> s = format_result_with_ci(0.85, 0.83, 0.87)
        >>> print(s)  # "0.850 [0.830, 0.870]"
    """
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} [{fmt.format(ci_lower)}, {fmt.format(ci_upper)}]"


def format_result_with_std(
    mean: float,
    std: float,
    precision: int = 3,
) -> str:
    """
    Format a result with standard deviation as a string.
    
    Args:
        mean: Mean value.
        std: Standard deviation.
        precision: Number of decimal places.
        
    Returns:
        Formatted string like "0.850 ± 0.020".
        
    Example:
        >>> s = format_result_with_std(0.85, 0.02)
        >>> print(s)  # "0.850 ± 0.020"
    """
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"
