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
    python examples/evaluate_sensitivity.py --sweep-samples 1000,5000,10000
    python examples/evaluate_sensitivity.py --sweep-correlation 0.0,0.3,0.5,0.7,0.9
    python examples/evaluate_sensitivity.py --sweep-factors 3,5,7,10 --n-seeds 10
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dgp import D1Independent, D2Correlated, D3SingleRedundant, D4MultiRedundant
from src.encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
    E7OvercompleteEntangled,
    E8OvercompleteDisjoint,
)
from src.metrics import MetricRegistry
from src.evaluation import (
    sensitivity_analysis_1d,
    compute_sensitivity_statistics,
    save_sensitivity_results,
)


# Default configuration
DEFAULT_N_SAMPLES = 5000
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


def evaluate_dgp_encoder_combination(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate a single DGP/encoder combination with given parameters.
    
    Args:
        params: Dictionary with keys:
            - dgp: DGP class name ('D1', 'D2', etc.)
            - encoder: Encoder class name ('E1', 'E2', etc.)
            - n_samples: Number of samples
            - n_factors: Number of factors
            - seed: Random seed
            - Additional DGP/encoder-specific parameters
    
    Returns:
        Dictionary of metric scores.
    """
    # Extract parameters
    dgp_name = params['dgp']
    encoder_name = params['encoder']
    n_samples = params['n_samples']
    n_factors = params['n_factors']
    seed = params['seed']
    
    # Map names to classes
    dgp_classes = {
        'D1': D1Independent,
        'D2': D2Correlated,
        'D3': D3SingleRedundant,
        'D4': D4MultiRedundant,
    }
    
    encoder_classes = {
        'E1': E1ElementwiseLinear,
        'E2': E2ElementwiseNonlinear,
        'E3': E3LinearlyEntangled,
        'E7': E7OvercompleteEntangled,
        'E8': E8OvercompleteDisjoint,
    }
    
    # Create DGP
    dgp_cls = dgp_classes[dgp_name]
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
            # Set r=1 by default for D4 to avoid constraint issues
            dgp_kwargs['r'] = 1
    
    dgp = dgp_cls(**dgp_kwargs)
    Z = dgp.sample(n_samples)
    
    # Create encoder
    encoder_cls = encoder_classes[encoder_name]
    encoder_kwargs = {'d': n_factors, 'seed': seed}
    
    # Add encoder-specific parameters
    if encoder_name == 'E2' and 'nonlinearity_strength' in params:
        encoder_kwargs['nonlinearity_strength'] = params['nonlinearity_strength']
    if encoder_name == 'E3' and 'condition_number' in params:
        encoder_kwargs['condition_number'] = params['condition_number']
    if encoder_name == 'E7':
        if 'condition_number' in params:
            encoder_kwargs['condition_number'] = params['condition_number']
        if 'm' in params:
            encoder_kwargs['m'] = params['m']
    if encoder_name == 'E8' and 'codes_per_factor' in params:
        encoder_kwargs['codes_per_factor'] = params['codes_per_factor']
    
    encoder = encoder_cls(**encoder_kwargs)
    Z_hat = encoder.encode(Z)
    
    # Compute metrics
    registry = MetricRegistry()
    registry.register_defaults()
    
    all_results = registry.compute_all(Z, Z_hat)
    
    # Extract scores
    results = {}
    
    # DCI subscores
    if 'dci' in all_results:
        dci_result = all_results['dci']
        results['dci_disentanglement'] = dci_result.subscores.get('disentanglement', np.nan)
        results['dci_completeness'] = dci_result.subscores.get('completeness', np.nan)
        results['dci_informativeness'] = dci_result.subscores.get('informativeness_test', np.nan)
    
    # MCC variants
    for mcc_type in ['mcc_pearson', 'mcc_spearman', 'mcc_rdc']:
        if mcc_type in all_results:
            results[mcc_type] = all_results[mcc_type].primary_score
    
    # R²
    if 'r2' in all_results:
        results['r2'] = all_results['r2'].primary_score
    
    return results


def sweep_samples(
    dgp: str,
    encoder: str,
    sample_values: List[int],
    n_factors: int,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
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
        return evaluate_dgp_encoder_combination(params)
    
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
):
    """Run sensitivity analysis sweeping over number of factors."""
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: Varying Number of Factors")
    print(f"DGP: {dgp}, Encoder: {encoder}")
    print("=" * 80)
    
    def eval_fn(params):
        params['dgp'] = dgp
        params['encoder'] = encoder
        params['n_samples'] = n_samples
        return evaluate_dgp_encoder_combination(params)
    
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
    plot_sensitivity(
        stats,
        param_name="Number of Factors",
        title=f"Sensitivity to Factor Dimensionality\n{dgp} + {encoder}",
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
        return evaluate_dgp_encoder_combination(params)
    
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
        return evaluate_dgp_encoder_combination(params)
    
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


def plot_sensitivity(
    stats: Dict[str, Dict[str, List[float]]],
    param_name: str,
    title: str,
    output_path: Path,
):
    """
    Create camera-ready sensitivity plot with error bands.
    
    Args:
        stats: Statistics dictionary from compute_sensitivity_statistics.
        param_name: Name of the parameter being varied.
        title: Plot title.
        output_path: Path to save the plot.
    """
    setup_plot_style()
    
    # Select important metrics to plot
    important_metrics = [
        'dci_disentanglement',
        'dci_completeness',
        'mcc_pearson',
        'r2',
    ]
    
    # Filter to metrics that exist
    metrics_to_plot = [m for m in important_metrics if m in stats]
    
    if not metrics_to_plot:
        print(f"Warning: No metrics to plot for {output_path}")
        return
    
    # Create figure
    n_metrics = len(metrics_to_plot)
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
        
        # Formatting
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(metric_name.replace('_', ' ').title(), fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
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
        choices=['E1', 'E2', 'E3', 'E7', 'E8'],
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
    
    args = parser.parse_args()
    
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
        )
    
    # If no sweeps specified, show help
    if not any([args.sweep_samples, args.sweep_factors, args.sweep_correlation, args.sweep_nonlinearity]):
        parser.print_help()
        print("\nNo sweep specified. Use --sweep-* flags to run sensitivity analysis.")
    else:
        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
