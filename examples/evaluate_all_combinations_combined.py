"""
Evaluate all combinations of DGPs and Encoders - Combined Visualization.

This script creates a single combined figure showing all encoder-metric 
combinations for each DGP, similar to the provided example image format.

Usage:
    python examples/evaluate_all_combinations_combined.py [--samples N] [--factors D] [--seed S]

    Options:
        --samples N    Number of samples to generate (default: 5000)
        --factors D    Number of latent factors (default: 4)
        --seed S       Random seed for reproducibility (default: 42)
        --output FILE  Output file path (default: results/combined_heatmap.png)

    Examples:
        python examples/evaluate_all_combinations_combined.py
        python examples/evaluate_all_combinations_combined.py --samples 10000
        python examples/evaluate_all_combinations_combined.py --output my_figure.png

Output:
    - Creates a single figure with one heatmap per DGP
    - Each heatmap shows Encoders (rows) × Metrics (columns)
    - Saves figure as PNG file
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import MetricRegistry
from src.dgp import D1Independent, D2Correlated, D3SingleRedundant, D4MultiRedundant
from src.encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
    E4UndercompleteLinear,
    E5OvercompleteLinear,
    E6OvercompleteMulticodes,
)


# Configuration: All DGPs and Encoders
DGP_CLASSES = {
    "D1": D1Independent,
    "D2": D2Correlated,
    "D3": D3SingleRedundant,
    "D4": D4MultiRedundant,
}

ENCODER_CLASSES = {
    "E1": E1ElementwiseLinear,
    "E2": E2ElementwiseNonlinear,
    "E3": E3LinearlyEntangled,
    "E4": E4UndercompleteLinear,
    "E5": E5OvercompleteLinear,
    "E6": E6OvercompleteMulticodes,
}

# Metrics to evaluate - ALL metrics including DCI subscores
METRIC_NAMES = {
    "dci_disentanglement": "DCI-D",
    "dci_completeness": "DCI-C",
    "dci_informativeness": "DCI-I",
    "mcc_pearson": "MCC-P",
    "mcc_spearman": "MCC-S",
    "mcc_rdc": "MCC-RDC",
    "r2": "R²",
}


def evaluate_combination(
    dgp_name: str,
    encoder_name: str,
    n_samples: int,
    n_factors: int,
    seed: int,
    registry: MetricRegistry,
) -> Dict[str, float]:
    """Evaluate one DGP/encoder combination on all metrics."""
    dgp_cls = DGP_CLASSES[dgp_name]
    dgp = dgp_cls(d=n_factors, seed=seed)
    Z = dgp.sample(n_samples)

    encoder_cls = ENCODER_CLASSES[encoder_name]
    encoder = encoder_cls(d=n_factors, seed=seed)
    Z_hat = encoder.encode(Z)

    # Compute ALL metrics using registry.compute_all
    all_results = registry.compute_all(Z, Z_hat)
    
    # Initialize results with NaN for all metrics
    results = {key: np.nan for key in METRIC_NAMES.keys()}
    
    # Extract DCI subscores if available
    if "dci" in all_results:
        dci_result = all_results["dci"]
        results["dci_disentanglement"] = dci_result.subscores["disentanglement"]
        results["dci_completeness"] = dci_result.subscores["completeness"]
        results["dci_informativeness"] = dci_result.subscores["informativeness_test"]
    
    # Extract MCC and R² primary scores if available
    if "mcc_pearson" in all_results:
        results["mcc_pearson"] = all_results["mcc_pearson"].primary_score
    if "mcc_spearman" in all_results:
        results["mcc_spearman"] = all_results["mcc_spearman"].primary_score
    if "mcc_rdc" in all_results:
        results["mcc_rdc"] = all_results["mcc_rdc"].primary_score
    if "r2" in all_results:
        results["r2"] = all_results["r2"].primary_score

    return results


def evaluate_all_combinations(
    n_samples: int = 5000,
    n_factors: int = 4,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate all DGP/encoder combinations.

    Returns:
        Nested dictionary: dgp -> encoder -> metric -> score
    """
    print("=" * 80)
    print("EVALUATING ALL DGP/ENCODER COMBINATIONS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Samples:  {n_samples}")
    print(f"  Factors:  {n_factors}")
    print(f"  Seed:     {seed}")
    print(f"  DGPs:     {list(DGP_CLASSES.keys())}")
    print(f"  Encoders: {list(ENCODER_CLASSES.keys())}")
    print(f"  Metrics:  DCI (3 subscores), MCC (Pearson, Spearman, RDC), R²")
    print("=" * 80)

    registry = MetricRegistry()
    registry.register_defaults()

    # Store results: dgp -> encoder -> metric -> score
    all_results = {dgp: {} for dgp in DGP_CLASSES.keys()}

    total_combinations = len(DGP_CLASSES) * len(ENCODER_CLASSES)
    current = 0

    for dgp_name in DGP_CLASSES.keys():
        for encoder_name in ENCODER_CLASSES.keys():
            current += 1
            print(f"[{current}/{total_combinations}] Evaluating {dgp_name} × {encoder_name}...", end=" ")

            try:
                results = evaluate_combination(
                    dgp_name=dgp_name,
                    encoder_name=encoder_name,
                    n_samples=n_samples,
                    n_factors=n_factors,
                    seed=seed,
                    registry=registry,
                )
                all_results[dgp_name][encoder_name] = results
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                all_results[dgp_name][encoder_name] = {
                    metric: np.nan for metric in METRIC_NAMES.keys()
                }

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return all_results


def create_combined_heatmap(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    dgp_names: List[str],
    encoder_names: List[str],
    metric_names: Dict[str, str],
) -> plt.Figure:
    """
    Create a combined figure with one heatmap per DGP.
    
    Layout: One subplot per DGP, showing Encoders (rows) × Metrics (columns).
    """
    n_dgps = len(dgp_names)
    n_encoders = len(encoder_names)
    n_metrics = len(metric_names)
    
    # Create figure with subplots arranged vertically
    fig, axes = plt.subplots(
        n_dgps, 1,
        figsize=(10, 3 * n_dgps),
        constrained_layout=False
    )
    
    # Adjust layout to make room for the title
    fig.subplots_adjust(top=0.94, bottom=0.05, left=0.15, right=0.92)
    
    # Ensure axes is always iterable
    if n_dgps == 1:
        axes = [axes]
    
    for idx, dgp_name in enumerate(dgp_names):
        ax = axes[idx]
        
        # Prepare data matrix: Encoders (rows) × Metrics (columns)
        data = np.zeros((n_encoders, n_metrics))
        
        for i, encoder in enumerate(encoder_names):
            for j, metric_key in enumerate(metric_names.keys()):
                data[i, j] = all_results[dgp_name][encoder].get(metric_key, np.nan)
        
        # Normalize to 0-100 scale
        data_normalized = data * 100
        
        # Create heatmap
        im = ax.imshow(data_normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_metrics))
        ax.set_yticks(np.arange(n_encoders))
        ax.set_xticklabels(list(metric_names.values()), fontsize=10)
        
        # Get encoder display names
        encoder_labels = []
        for encoder_name in encoder_names:
            encoder_cls = ENCODER_CLASSES[encoder_name]
            encoder_instance = encoder_cls(d=4)
            # Shorten the name for display
            short_name = encoder_instance.name.split(':')[0] if ':' in encoder_instance.name else encoder_instance.name
            encoder_labels.append(f"{encoder_name} ({short_name[:20]})")
        
        ax.set_yticklabels(encoder_labels, fontsize=9)
        
        # Add text annotations
        for i in range(n_encoders):
            for j in range(n_metrics):
                value = data_normalized[i, j]
                if not np.isnan(value):
                    text = ax.text(
                        j, i, f"{value:.0f}",
                        ha="center", va="center",
                        color="black" if 30 < value < 70 else "white",
                        fontsize=11,
                        fontweight="bold",
                    )
        
        # Get DGP display name
        dgp_cls = DGP_CLASSES[dgp_name]
        dgp_instance = dgp_cls(d=4)
        dgp_title = dgp_instance.name
        
        # Add title for this subplot
        ax.set_title(
            f"DGP = {dgp_title}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        
        # Add grid
        ax.set_xticks(np.arange(n_metrics + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_encoders + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        
        # Only show x-axis label on bottom subplot
        if idx == n_dgps - 1:
            ax.set_xlabel("Identifiability Metric", fontsize=11, fontweight="bold")
        
        # Y-axis label for all subplots
        ax.set_ylabel("Encoder", fontsize=11, fontweight="bold")
    
    # Add a single colorbar for all subplots
    fig.colorbar(
        im, ax=axes, 
        label="Score (0-100)", 
        orientation="vertical",
        fraction=0.02,
        pad=0.02
    )
    
    # Overall title
    fig.suptitle(
        "Identifiability Metrics: DGP × Encoder Combinations",
        fontsize=14,
        fontweight="bold"
    )
    
    return fig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate all DGP/Encoder combinations - Combined visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=4,
        help="Number of latent factors (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/combined_heatmap.png",
        help="Output file path (default: results/combined_heatmap.png)",
    )

    args = parser.parse_args()

    # Evaluate all combinations
    all_results = evaluate_all_combinations(
        n_samples=args.samples,
        n_factors=args.factors,
        seed=args.seed,
    )

    # Create combined figure
    print("\nGenerating combined figure...")
    print("=" * 80)
    
    dgp_names = list(DGP_CLASSES.keys())
    encoder_names = list(ENCODER_CLASSES.keys())
    
    fig = create_combined_heatmap(
        all_results=all_results,
        dgp_names=dgp_names,
        encoder_names=encoder_names,
        metric_names=METRIC_NAMES,
    )
    
    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Figure saved to: {output_path.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
