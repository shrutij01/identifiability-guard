"""
Sensitivity analysis utilities.

Provides functions for grid-based parameter sweeps to study how metrics
vary with different parameter values.
"""

from typing import Callable, Dict, List, Any, Optional, Tuple
import numpy as np
from itertools import product
import json


def parameter_sweep(
    evaluation_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    param_grid: Dict[str, List[Any]],
    fixed_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run evaluation over a grid of parameter values.
    
    Args:
        evaluation_fn: Function that takes a parameter dict and returns metrics.
        param_grid: Dictionary mapping parameter names to lists of values to try.
        fixed_params: Dictionary of fixed parameters (not varied).
        verbose: If True, prints progress.
        
    Returns:
        List of result dictionaries, each containing parameters and metrics.
        
    Example:
        >>> def eval_fn(params):
        ...     dgp = D2Correlated(d=params['d'], correlation=params['corr'])
        ...     # ... evaluate ...
        ...     return {"mcc": score}
        >>> results = parameter_sweep(
        ...     eval_fn,
        ...     param_grid={"d": [3, 5, 7], "corr": [0.3, 0.5, 0.7]}
        ... )
    """
    if fixed_params is None:
        fixed_params = {}
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    results = []
    total = len(combinations)
    
    if verbose:
        print(f"Running parameter sweep with {total} combinations...")
    
    for i, combo in enumerate(combinations):
        # Build parameter dict
        params = dict(zip(param_names, combo))
        params.update(fixed_params)
        
        if verbose:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"[{i+1}/{total}] Evaluating: {param_str}")
        
        try:
            # Run evaluation
            metrics = evaluation_fn(params)
            
            # Store results
            result = {
                'params': params.copy(),
                'metrics': metrics,
            }
            results.append(result)
        
        except Exception as e:
            if verbose:
                print(f"  Warning: Evaluation failed: {e}")
            # Store failed result with NaN metrics
            result = {
                'params': params.copy(),
                'metrics': {k: np.nan for k in ['error']},
                'error': str(e),
            }
            results.append(result)
    
    return results


def sensitivity_analysis_1d(
    evaluation_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    param_name: str,
    param_values: List[Any],
    fixed_params: Optional[Dict[str, Any]] = None,
    n_seeds: int = 1,
    base_seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Any], Dict[str, List[List[float]]]]:
    """
    Run 1D sensitivity analysis: vary one parameter while keeping others fixed.
    
    Args:
        evaluation_fn: Function that takes parameters and returns metrics.
        param_name: Name of parameter to vary.
        param_values: List of values for the varying parameter.
        fixed_params: Dictionary of fixed parameters.
        n_seeds: Number of random seeds to use per parameter value.
        base_seed: Starting seed value.
        verbose: If True, prints progress.
        
    Returns:
        Tuple of (param_values, metric_results).
        metric_results: Dict mapping metric names to list of lists
                       (outer list: param values, inner list: seeds).
        
    Example:
        >>> def eval_fn(params):
        ...     dgp = D2Correlated(d=5, correlation=params['corr'], seed=params['seed'])
        ...     # ... evaluate ...
        ...     return {"mcc": score}
        >>> values, results = sensitivity_analysis_1d(
        ...     eval_fn,
        ...     param_name="corr",
        ...     param_values=[0.0, 0.3, 0.5, 0.7, 0.9],
        ...     n_seeds=5
        ... )
    """
    if fixed_params is None:
        fixed_params = {}
    
    # Initialize results storage
    metric_results: Dict[str, List[List[float]]] = {}
    all_metric_names: set = set()  # Track all metric names seen
    
    if verbose:
        print(f"Running 1D sensitivity analysis for parameter '{param_name}'")
        print(f"Values: {param_values}")
        print(f"Seeds per value: {n_seeds}")
    
    for i, param_value in enumerate(param_values):
        if verbose:
            print(f"\n[{i+1}/{len(param_values)}] {param_name} = {param_value}")
        
        # Run with multiple seeds
        seed_results: Dict[str, List[float]] = {}
        
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            
            # Build parameter dict
            params = fixed_params.copy()
            params[param_name] = param_value
            params['seed'] = seed
            
            try:
                metrics = evaluation_fn(params)
                
                # Store results
                for metric_name, value in metrics.items():
                    all_metric_names.add(metric_name)
                    if metric_name not in seed_results:
                        # Initialize with NaN for any prior seeds
                        seed_results[metric_name] = [np.nan] * seed_idx
                    seed_results[metric_name].append(value)
                
                # For any metric we've seen before but not in this run, add NaN
                for metric_name in all_metric_names:
                    if metric_name not in seed_results:
                        seed_results[metric_name] = [np.nan] * seed_idx
                    elif len(seed_results[metric_name]) < seed_idx + 1:
                        seed_results[metric_name].append(np.nan)
            
            except Exception as e:
                if verbose:
                    print(f"  Warning: Evaluation failed for seed {seed}: {e}")
                # Add NaN for failed runs for all known metrics
                for metric_name in all_metric_names:
                    if metric_name not in seed_results:
                        seed_results[metric_name] = [np.nan] * seed_idx
                    seed_results[metric_name].append(np.nan)
        
        # Ensure all seed_results have n_seeds values
        for metric_name in list(seed_results.keys()):
            while len(seed_results[metric_name]) < n_seeds:
                seed_results[metric_name].append(np.nan)
        
        # Add to overall results
        for metric_name, values in seed_results.items():
            if metric_name not in metric_results:
                # Back-fill with NaN lists for prior param values
                metric_results[metric_name] = [[np.nan] * n_seeds for _ in range(i)]
            metric_results[metric_name].append(values)
    
    # Ensure all metrics have entries for all param values (back-fill any missing)
    for metric_name in metric_results:
        while len(metric_results[metric_name]) < len(param_values):
            metric_results[metric_name].append([np.nan] * n_seeds)
    
    return param_values, metric_results


def compute_sensitivity_statistics(
    param_values: List[Any],
    metric_results: Dict[str, List[List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute statistics (mean, std) for sensitivity analysis results.
    
    Args:
        param_values: List of parameter values.
        metric_results: Dict mapping metric names to lists of lists of values.
        
    Returns:
        Dictionary mapping metric names to statistics dicts with keys:
        'values' (param values), 'mean', 'std', 'ci_lower', 'ci_upper'.
        
    Example:
        >>> values, results = sensitivity_analysis_1d(...)
        >>> stats = compute_sensitivity_statistics(values, results)
        >>> mcc_stats = stats.get("mcc")
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(mcc_stats.get("values"), mcc_stats.get("mean"))
        >>> plt.fill_between(mcc_stats.get("values"),
        ...                  mcc_stats.get("ci_lower"),
        ...                  mcc_stats.get("ci_upper"),
        ...                  alpha=0.3)
    """
    from scipy import stats as scipy_stats
    
    statistics = {}
    
    for metric_name, results_list in metric_results.items():
        means = []
        stds = []
        ci_lowers = []
        ci_uppers = []
        
        for values in results_list:
            values_array = np.array(values)
            valid_values = values_array[~np.isnan(values_array)]
            
            if len(valid_values) == 0:
                means.append(np.nan)
                stds.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
            else:
                mean = np.mean(valid_values)
                std = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                sem = std / np.sqrt(len(valid_values)) if len(valid_values) > 1 else 0.0
                
                # 95% confidence interval
                if len(valid_values) > 1:
                    ci = scipy_stats.t.interval(
                        0.95,
                        df=len(valid_values) - 1,
                        loc=mean,
                        scale=sem,
                    )
                    ci_lower, ci_upper = ci
                else:
                    ci_lower = ci_upper = mean
                
                means.append(mean)
                stds.append(std)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
        
        statistics[metric_name] = {
            'values': param_values,
            'mean': means,
            'std': stds,
            'ci_lower': ci_lowers,
            'ci_upper': ci_uppers,
        }
    
    return statistics


def save_sensitivity_results(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save sensitivity analysis results to JSON file.
    
    Args:
        results: List of result dictionaries.
        output_path: Path to output JSON file.
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_sensitivity_results(input_path: str) -> List[Dict[str, Any]]:
    """
    Load sensitivity analysis results from JSON file.
    
    Args:
        input_path: Path to input JSON file.
        
    Returns:
        List of result dictionaries.
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    return results
