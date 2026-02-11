"""
Enhanced Sensitivity Analysis Evaluation Script.

This script provides comprehensive sensitivity analysis capabilities for studying
how identifiability metrics vary with different parameters of DGPs and encoders.

Features:
- Parameter sweeps: samples, factors, correlation, redundancy, nonlinearity
- Multi-seed evaluations with statistical aggregation
- CSV/JSON output with raw and aggregated results
- Camera-ready sensitivity plots with error bands

Usage:
    python examples/sensitivity.py --sweep-samples 1000,5000,10000
    python examples/sensitivity.py --sweep-correlation 0.0,0.3,0.5,0.7,0.9
    python examples/sensitivity.py --sweep-factors 3,5,7,10 --n-seeds 10
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from identifiability_guard.metrics import MetricRegistry
from identifiability_guard.evaluation import (
    sensitivity_analysis_1d,
    compute_sensitivity_statistics,
    save_sensitivity_results,
    DGP_CLASSES,
    ENCODER_CLASSES,
    ALL_METRICS,
    DEFAULT_METRICS,
    METRIC_DISPLAY_NAMES,
    create_dgp_with_params,
    create_encoder_with_params,
    extract_metric_scores,
)


# Default configuration
DEFAULT_N_SAMPLES = 1000
DEFAULT_N_FACTORS = 5
DEFAULT_N_SEEDS = 5
DEFAULT_BASE_SEED = 42
DEFAULT_OUTPUT_DIR = "results/sensitivity"


def setup_plot_style():
    """Set up matplotlib style for camera-ready plots."""
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.titlesize'] = 14
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 6


def evaluate_dgp_encoder_combination(
    params: Dict[str, Any],
    metrics_to_compute: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate a single DGP/encoder combination with given parameters.
    
    This function uses the shared helpers from identifiability_guard.evaluation for DGP/encoder
    creation and metric extraction.
    
    Args:
        params: Dictionary with keys:
            - dgp: DGP class name ('D1', 'D2', etc.)
            - encoder: Encoder class name ('E1', 'E2', etc.)
            - n_samples: Number of samples
            - n_factors: Number of factors
            - seed: Random seed
            - Additional DGP/encoder-specific parameters
        metrics_to_compute: Set of metric names to compute. If None, uses DEFAULT_METRICS.
            Use set(ALL_METRICS.keys()) for all metrics.
    
    Returns:
        Dictionary of metric scores.
    """
    if metrics_to_compute is None:
        metrics_to_compute = DEFAULT_METRICS
    
    # Extract parameters
    dgp_name = params['dgp']
    encoder_name = params['encoder']
    n_samples = params['n_samples']
    n_factors = params['n_factors']
    seed = params['seed']
    
    # Create DGP and encoder using shared helpers
    dgp = create_dgp_with_params(dgp_name, n_factors, seed, params)
    Z = dgp.sample(n_samples)
    
    encoder = create_encoder_with_params(encoder_name, n_factors, seed, params)
    Z_hat = encoder.encode(Z)
    
    # Compute metrics
    registry = MetricRegistry()
    registry.register_defaults()
    all_results = registry.compute_all(Z, Z_hat)
    
    # Extract scores using shared helper
    return extract_metric_scores(all_results, metrics_to_compute)


def sweep_samples(
    dgp: str,
    encoder: str,
    sample_values: List[int],
    n_factors: int,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
    metrics_to_compute: Optional[Set[str]] = None,
):
    """Run sensitivity analysis sweeping over number of samples."""
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: Varying Number of Samples")
    print(f"DGP: {dgp}, Encoder: {encoder}")
    print("=" * 80)
    
    def eval_fn(params):
        params['dgp'] = dgp
        params['encoder'] = encoder
        params['n_factors'] = n_factors
        return evaluate_dgp_encoder_combination(params, metrics_to_compute)
    
    param_values, metric_results = sensitivity_analysis_1d(
        eval_fn,
        param_name='n_samples',
        param_values=sample_values,
        fixed_params={},
        n_seeds=n_seeds,
        base_seed=base_seed,
        verbose=True,
    )
    
    # Compute statistics
    stats = compute_sensitivity_statistics(param_values, metric_results)
    
    # Save results
    output_file = output_dir / f"sweep_samples_{dgp}_{encoder}.json"
    save_sensitivity_results(
        [{'param_values': param_values, 'metric_results': metric_results, 'statistics': stats}],
        str(output_file)
    )
    
    # Create plot
    plot_sensitivity(
        stats,
        param_name="Number of Samples",
        title=f"Sensitivity to Sample Size\n{dgp} + {encoder}",
        output_path=output_dir / f"sweep_samples_{dgp}_{encoder}.png"
    )
    
    return stats


def sweep_factors(
    dgp: str,
    encoder: str,
    factor_values: List[int],
    n_samples: int,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
    metrics_to_compute: Optional[Set[str]] = None,
    n_factors_ground_truth: Optional[int] = None,
):
    """Run sensitivity analysis sweeping over number of factors.

    Parameters
    ----------
    n_factors_ground_truth : int, optional
        The ground-truth number of factors used in the experiment.  When
        provided it is shown in the plot title so the reader knows which
        ``d`` value the experiment was designed around.
    """
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: Varying Number of Factors")
    print(f"DGP: {dgp}, Encoder: {encoder}")
    print("=" * 80)
    
    def eval_fn(params):
        params['dgp'] = dgp
        params['encoder'] = encoder
        params['n_samples'] = n_samples
        return evaluate_dgp_encoder_combination(params, metrics_to_compute)
    
    param_values, metric_results = sensitivity_analysis_1d(
        eval_fn,
        param_name='n_factors',
        param_values=factor_values,
        fixed_params={},
        n_seeds=n_seeds,
        base_seed=base_seed,
        verbose=True,
    )
    
    # Compute statistics
    stats = compute_sensitivity_statistics(param_values, metric_results)
    
    # Save results
    output_file = output_dir / f"sweep_factors_{dgp}_{encoder}.json"
    save_sensitivity_results(
        [{'param_values': param_values, 'metric_results': metric_results, 'statistics': stats}],
        str(output_file)
    )
    
    # Create plot
    gt_str = f", d={n_factors_ground_truth}" if n_factors_ground_truth is not None else ""
    plot_sensitivity(
        stats,
        param_name="Number of Factors",
        title=f"Sensitivity to Factor Dimensionality\n{dgp} + {encoder}{gt_str}",
        output_path=output_dir / f"sweep_factors_{dgp}_{encoder}.png"
    )
    
    return stats


def sweep_correlation(
    encoder: str,
    correlation_values: List[float],
    n_samples: int,
    n_factors: int,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
    metrics_to_compute: Optional[Set[str]] = None,
):
    """Run sensitivity analysis sweeping over correlation (D2 only)."""
    dgp = 'D2'
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: Varying Correlation")
    print(f"DGP: {dgp}, Encoder: {encoder}")
    print("=" * 80)
    
    def eval_fn(params):
        params['dgp'] = dgp
        params['encoder'] = encoder
        params['n_samples'] = n_samples
        params['n_factors'] = n_factors
        return evaluate_dgp_encoder_combination(params, metrics_to_compute)
    
    param_values, metric_results = sensitivity_analysis_1d(
        eval_fn,
        param_name='correlation',
        param_values=correlation_values,
        fixed_params={},
        n_seeds=n_seeds,
        base_seed=base_seed,
        verbose=True,
    )
    
    # Compute statistics
    stats = compute_sensitivity_statistics(param_values, metric_results)
    
    # Save results
    output_file = output_dir / f"sweep_correlation_{encoder}.json"
    save_sensitivity_results(
        [{'param_values': param_values, 'metric_results': metric_results, 'statistics': stats}],
        str(output_file)
    )
    
    # Create plot
    plot_sensitivity(
        stats,
        param_name="Correlation Coefficient",
        title=f"Sensitivity to Factor Correlation\n{dgp} + {encoder}",
        output_path=output_dir / f"sweep_correlation_{encoder}.png"
    )
    
    return stats


def sweep_nonlinearity(
    dgp: str,
    nonlinearity_values: List[float],
    n_samples: int,
    n_factors: int,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
    metrics_to_compute: Optional[Set[str]] = None,
):
    """Run sensitivity analysis sweeping over nonlinearity strength (E2 only)."""
    encoder = 'E2'
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: Varying Nonlinearity Strength")
    print(f"DGP: {dgp}, Encoder: {encoder}")
    print("=" * 80)
    
    def eval_fn(params):
        params['dgp'] = dgp
        params['encoder'] = encoder
        params['n_samples'] = n_samples
        params['n_factors'] = n_factors
        return evaluate_dgp_encoder_combination(params, metrics_to_compute)
    
    param_values, metric_results = sensitivity_analysis_1d(
        eval_fn,
        param_name='nonlinearity_strength',
        param_values=nonlinearity_values,
        fixed_params={},
        n_seeds=n_seeds,
        base_seed=base_seed,
        verbose=True,
    )
    
    # Compute statistics
    stats = compute_sensitivity_statistics(param_values, metric_results)
    
    # Save results
    output_file = output_dir / f"sweep_nonlinearity_{dgp}.json"
    save_sensitivity_results(
        [{'param_values': param_values, 'metric_results': metric_results, 'statistics': stats}],
        str(output_file)
    )
    
    # Create plot
    plot_sensitivity(
        stats,
        param_name="Nonlinearity Strength",
        title=f"Sensitivity to Nonlinearity\n{dgp} + {encoder}",
        output_path=output_dir / f"sweep_nonlinearity_{dgp}.png"
    )
    
    return stats


def sweep_encoder_nonlinearity(
    dgp: str,
    nonlinearity_values: List[float],
    n_samples: int,
    n_factors: int,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
    metrics_to_compute: Optional[Set[str]] = None,
    nonlinearity_type: str = "tanh_modified",
):
    """Run sensitivity analysis sweeping over encoder E2 nonlinearity strength.

    This is similar to ``sweep_nonlinearity`` but explicitly documented as
    sweeping the **encoder** nonlinearity, keeping the DGP fixed.

    Parameters
    ----------
    dgp : str
        DGP to pair with E2 (e.g. ``'D1'``).
    nonlinearity_values : list of float
        Strength values in [0, 1] (0 = linear, 1 = fully nonlinear).
    nonlinearity_type : str
        Label used in filenames / titles (default ``'tanh_modified'``).
    """
    encoder = 'E2'
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: Varying Encoder Nonlinearity Strength")
    print(f"DGP: {dgp}, Encoder: {encoder}, type: {nonlinearity_type}")
    print("=" * 80)

    def eval_fn(params):
        params['dgp'] = dgp
        params['encoder'] = encoder
        params['n_samples'] = n_samples
        params['n_factors'] = n_factors
        return evaluate_dgp_encoder_combination(params, metrics_to_compute)

    param_values, metric_results = sensitivity_analysis_1d(
        eval_fn,
        param_name='nonlinearity_strength',
        param_values=nonlinearity_values,
        fixed_params={},
        n_seeds=n_seeds,
        base_seed=base_seed,
        verbose=True,
    )

    # Compute statistics
    stats = compute_sensitivity_statistics(param_values, metric_results)

    # Save results
    output_file = output_dir / f"sweep_encoder_nl_{dgp}_{nonlinearity_type}.json"
    save_sensitivity_results(
        [{'param_values': param_values, 'metric_results': metric_results, 'statistics': stats}],
        str(output_file)
    )

    # Create plot
    plot_sensitivity(
        stats,
        param_name="Encoder Nonlinearity Strength",
        title=f"Sensitivity to Encoder Nonlinearity ({nonlinearity_type})\n{dgp} + {encoder}",
        output_path=output_dir / f"sweep_encoder_nl_{dgp}_{nonlinearity_type}.png"
    )

    return stats


def plot_sensitivity(
    stats: Dict[str, Dict[str, List[float]]],
    param_name: str,
    title: str,
    output_path: Path,
    metrics_to_plot: Optional[List[str]] = None,
):
    """
    Create camera-ready sensitivity plot with error bands.
    
    Args:
        stats: Statistics dictionary from compute_sensitivity_statistics.
        param_name: Name of the parameter being varied.
        title: Plot title.
        output_path: Path to save the plot.
        metrics_to_plot: List of metric names to plot. If None, uses all DEFAULT_METRICS.
    """
    setup_plot_style()
    
    # Select metrics to plot - use DEFAULT_METRICS by default
    if metrics_to_plot is None:
        metrics_to_plot = list(DEFAULT_METRICS)
    
    # Filter to metrics that exist in stats
    metrics_to_plot = [m for m in metrics_to_plot if m in stats]
    
    if not metrics_to_plot:
        print(f"Warning: No metrics to plot for {output_path}")
        return
    
    # Create figure with appropriate layout for 7 metrics
    n_metrics = len(metrics_to_plot)
    # Use 2 rows if more than 4 metrics for better readability
    if n_metrics > 4:
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
    else:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
    
    # Plot each metric
    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        metric_stats = stats[metric_name]
        
        param_values = metric_stats['values']
        means = metric_stats['mean']
        ci_lowers = metric_stats['ci_lower']
        ci_uppers = metric_stats['ci_upper']
        
        # Plot line with error band
        ax.plot(param_values, means, 'o-', label=metric_name, linewidth=2, markersize=6)
        ax.fill_between(param_values, ci_lowers, ci_uppers, alpha=0.2)
        
        # Formatting - use display name if available
        display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace('_', ' ').title())
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(display_name, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved plot to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enhanced sensitivity analysis for identifiability metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Sweep types
    parser.add_argument(
        '--sweep-samples',
        type=str,
        help='Comma-separated list of sample sizes (e.g., "1000,5000,10000")',
    )
    parser.add_argument(
        '--sweep-factors',
        type=str,
        help='Comma-separated list of factor counts (e.g., "3,5,7,10")',
    )
    parser.add_argument(
        '--sweep-correlation',
        type=str,
        help='Comma-separated list of correlation values for D2 (e.g., "0.0,0.3,0.5,0.7,0.9")',
    )
    parser.add_argument(
        '--sweep-nonlinearity',
        type=str,
        help='Comma-separated list of nonlinearity strengths for E2 (e.g., "0.0,0.25,0.5,0.75,1.0")',
    )
    parser.add_argument(
        '--sweep-encoder-nonlinearity',
        type=str,
        help='Comma-separated list of encoder nonlinearity strengths for E2 (e.g., "0.0,0.25,0.5,0.75,1.0")',
    )
    
    # DGP and encoder selection
    parser.add_argument(
        '--dgp',
        type=str,
        default='D1',
        choices=['D1', 'D2', 'D3', 'D4'],
        help='DGP to use (default: D1)',
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='E1',
        choices=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'],
        help='Encoder to use (default: E1)',
    )
    
    # Default parameters
    parser.add_argument(
        '--n-samples',
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f'Default number of samples (default: {DEFAULT_N_SAMPLES})',
    )
    parser.add_argument(
        '--n-factors',
        type=int,
        default=DEFAULT_N_FACTORS,
        help=f'Default number of factors (default: {DEFAULT_N_FACTORS})',
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f'Number of random seeds per configuration (default: {DEFAULT_N_SEEDS})',
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f'Base random seed (default: {DEFAULT_BASE_SEED})',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})',
    )
    
    # Metrics selection
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help=(
            'Comma-separated list of metrics to compute. '
            f'Available: {",".join(ALL_METRICS.keys())}. '
            f'Default (subset): {",".join(DEFAULT_METRICS)}'
        ),
    )
    parser.add_argument(
        '--all-metrics',
        action='store_true',
        help='Compute all available metrics (overrides --metrics)',
    )
    
    args = parser.parse_args()
    
    # Parse metrics selection
    if args.all_metrics:
        metrics_to_compute = set(ALL_METRICS.keys())
    elif args.metrics:
        metrics_to_compute = set(m.strip() for m in args.metrics.split(','))
        invalid_metrics = metrics_to_compute - set(ALL_METRICS.keys())
        if invalid_metrics:
            parser.error(f"Invalid metrics: {invalid_metrics}. Available: {list(ALL_METRICS.keys())}")
    else:
        metrics_to_compute = DEFAULT_METRICS
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ENHANCED SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  DGP: {args.dgp}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Default samples: {args.n_samples}")
    print(f"  Default factors: {args.n_factors}")
    print(f"  Seeds per config: {args.n_seeds}")
    print(f"  Metrics: {sorted(metrics_to_compute)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 80)
    
    # Run requested sweeps
    if args.sweep_samples:
        sample_values = [int(x) for x in args.sweep_samples.split(',')]
        sweep_samples(
            args.dgp,
            args.encoder,
            sample_values,
            args.n_factors,
            args.n_seeds,
            args.base_seed,
            output_dir,
            metrics_to_compute,
        )
    
    if args.sweep_factors:
        factor_values = [int(x) for x in args.sweep_factors.split(',')]
        sweep_factors(
            args.dgp,
            args.encoder,
            factor_values,
            args.n_samples,
            args.n_seeds,
            args.base_seed,
            output_dir,
            metrics_to_compute,
        )
    
    if args.sweep_correlation:
        correlation_values = [float(x) for x in args.sweep_correlation.split(',')]
        sweep_correlation(
            args.encoder,
            correlation_values,
            args.n_samples,
            args.n_factors,
            args.n_seeds,
            args.base_seed,
            output_dir,
            metrics_to_compute,
        )
    
    if args.sweep_nonlinearity:
        nonlinearity_values = [float(x) for x in args.sweep_nonlinearity.split(',')]
        sweep_nonlinearity(
            args.dgp,
            nonlinearity_values,
            args.n_samples,
            args.n_factors,
            args.n_seeds,
            args.base_seed,
            output_dir,
            metrics_to_compute,
        )
    
    if args.sweep_encoder_nonlinearity:
        nl_values = [float(x) for x in args.sweep_encoder_nonlinearity.split(',')]
        sweep_encoder_nonlinearity(
            args.dgp,
            nl_values,
            args.n_samples,
            args.n_factors,
            args.n_seeds,
            args.base_seed,
            output_dir,
            metrics_to_compute,
        )
    
    # If no sweeps specified, show help
    if not any([args.sweep_samples, args.sweep_factors, args.sweep_correlation,
                args.sweep_nonlinearity, args.sweep_encoder_nonlinearity]):
        parser.print_help()
        print("\nNo sweep specified. Use --sweep-* flags to run sensitivity analysis.")
    else:
        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
