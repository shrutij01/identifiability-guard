"""
Evaluate all combinations of DGPs and Encoders - Combined Visualization.

This script creates a single combined figure showing all encoder-metric 
combinations for each DGP, similar to the provided example image format.
Includes timing/memory profiling information.

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
    - Includes timing/memory profiling table
    - Saves figure as PNG file
"""

import sys
import os
import argparse
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import MetricRegistry
from src.evaluation import (
    DGP_CLASSES,
    ENCODER_CLASSES,
    METRIC_DISPLAY_NAMES,
    extract_metric_scores,
    get_dgp_class,
    get_encoder_class,
    sanitize_array,
)


# Metrics to evaluate - ALL metrics including DCI subscores
METRIC_NAMES = METRIC_DISPLAY_NAMES


def evaluate_combination(
    dgp_name: str,
    encoder_name: str,
    n_samples: int,
    n_factors: int,
    seed: int,
    registry: MetricRegistry,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Evaluate one DGP/encoder combination on all metrics using shared helpers.
    
    Returns:
        Tuple of:
        - results_dict: metric_name -> score
        - metric_timing: metric_name -> (time_seconds, memory_mb)
    """
    dgp_cls = get_dgp_class(dgp_name)
    dgp = dgp_cls(d=n_factors, seed=seed)
    Z = dgp.sample(n_samples)

    encoder_cls = get_encoder_class(encoder_name)
    encoder = encoder_cls(d=n_factors, seed=seed)
    Z_hat = encoder.encode(Z)
    
    # Sanitize arrays to prevent NaN/Inf errors
    Z = sanitize_array(Z)
    Z_hat = sanitize_array(Z_hat)

    # Compute each metric individually to track timing per metric
    results = {}
    metric_timing = {}
    
    # DCI (computes 3 subscores together)
    tracemalloc.start()
    start_time = time.perf_counter()
    try:
        dci_metric = registry.create('dci')
        dci_result = dci_metric.compute(Z, Z_hat)
        results['dci_disentanglement'] = dci_result.subscores.get('disentanglement', np.nan)
        results['dci_completeness'] = dci_result.subscores.get('completeness', np.nan)
        results['dci_informativeness'] = dci_result.subscores.get('informativeness_test', np.nan)
    except Exception:
        results['dci_disentanglement'] = np.nan
        results['dci_completeness'] = np.nan
        results['dci_informativeness'] = np.nan
    dci_time = time.perf_counter() - start_time
    _, dci_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dci_memory = dci_peak / (1024 * 1024)
    # DCI time/memory is shared by all 3 subscores (divide by 3 for per-metric average)
    metric_timing['dci_disentanglement'] = (dci_time / 3, dci_memory / 3)
    metric_timing['dci_completeness'] = (dci_time / 3, dci_memory / 3)
    metric_timing['dci_informativeness'] = (dci_time / 3, dci_memory / 3)
    
    # MCC variants (each computed separately)
    for mcc_variant in ['pearson', 'spearman', 'rdc']:
        metric_key = f'mcc_{mcc_variant}'
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            mcc_metric = registry.create(metric_key)
            mcc_result = mcc_metric.compute(Z, Z_hat)
            results[metric_key] = mcc_result.primary_score
        except Exception:
            results[metric_key] = np.nan
        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metric_timing[metric_key] = (elapsed, peak / (1024 * 1024))
    
    # R²
    tracemalloc.start()
    start_time = time.perf_counter()
    try:
        r2_metric = registry.create('r2')
        r2_result = r2_metric.compute(Z, Z_hat)
        results['r2'] = r2_result.primary_score
    except Exception:
        results['r2'] = np.nan
    elapsed = time.perf_counter() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metric_timing['r2'] = (elapsed, peak / (1024 * 1024))
    
    return results, metric_timing


def evaluate_all_combinations(
    n_samples: int = 5000,
    n_factors: int = 4,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, List[Tuple[float, float]]]]:
    """
    Evaluate all DGP/encoder combinations.

    Returns:
        Tuple of:
        - Nested dictionary: dgp -> encoder -> metric -> score
        - Metric timing dictionary: metric_name -> list of (time, memory) tuples across all combinations
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
    # Store timing per metric: metric_name -> list of (time, memory) tuples
    metric_timing_all = {metric: [] for metric in METRIC_NAMES.keys()}

    total_combinations = len(DGP_CLASSES) * len(ENCODER_CLASSES)
    current = 0

    for dgp_name in DGP_CLASSES.keys():
        for encoder_name in ENCODER_CLASSES.keys():
            current += 1
            print(f"[{current}/{total_combinations}] Evaluating {dgp_name} × {encoder_name}...", end=" ")

            try:
                results, metric_timing = evaluate_combination(
                    dgp_name=dgp_name,
                    encoder_name=encoder_name,
                    n_samples=n_samples,
                    n_factors=n_factors,
                    seed=seed,
                    registry=registry,
                )
                all_results[dgp_name][encoder_name] = results
                # Accumulate timing per metric
                for metric_name, timing in metric_timing.items():
                    metric_timing_all[metric_name].append(timing)
                total_time = sum(t for t, _ in metric_timing.values())
                print(f"✓ ({total_time:.2f}s)")
            except Exception as e:
                print(f"✗ Error: {e}")
                all_results[dgp_name][encoder_name] = {
                    metric: np.nan for metric in METRIC_NAMES.keys()
                }

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return all_results, metric_timing_all


def create_combined_heatmap(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    metric_timing: Dict[str, List[Tuple[float, float]]],
    dgp_names: List[str],
    encoder_names: List[str],
    metric_names: Dict[str, str],
    n_samples: int,
    n_factors: int,
) -> plt.Figure:
    """
    Create a combined figure with one heatmap per DGP and a timing/memory table per metric.
    
    Layout: One subplot per DGP, showing Encoders (rows) × Metrics (columns),
    plus a timing/memory summary table showing mean ± std per metric.
    """
    n_dgps = len(dgp_names)
    n_encoders = len(encoder_names)
    n_metrics = len(metric_names)
    
    # Create figure with GridSpec for flexible layout
    # Add extra space at bottom for timing table
    fig_height = 3.5 * n_dgps + 2.0  # Extra space for table
    fig = plt.figure(figsize=(12, fig_height))
    
    # Create GridSpec: main heatmaps + timing table at bottom
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_dgps + 1, 2, figure=fig, height_ratios=[1]*n_dgps + [0.35],
                  width_ratios=[20, 1], hspace=0.3, wspace=0.05)
    
    # Store axes and images for colorbar
    axes = []
    im = None
    
    for idx, dgp_name in enumerate(dgp_names):
        ax = fig.add_subplot(gs[idx, 0])
        axes.append(ax)
        
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
        ax.set_xticklabels(list(metric_names.values()), fontsize=9, rotation=45, ha='right')
        
        # Get encoder display names
        encoder_labels = []
        for encoder_name in encoder_names:
            encoder_cls = ENCODER_CLASSES[encoder_name]
            encoder_instance = encoder_cls(d=4)
            # Prefer an explicit display_name if provided (e.g., baselines)
            display = getattr(encoder_instance, "display_name", None)
            if display:
                short_name = display
            else:
                short_name = encoder_instance.name.split(':')[0] if ':' in encoder_instance.name else encoder_instance.name
            encoder_labels.append(f"{encoder_name} ({short_name[:22]})")
        
        ax.set_yticklabels(encoder_labels, fontsize=8)
        
        # Add text annotations
        for i in range(n_encoders):
            for j in range(n_metrics):
                value = data_normalized[i, j]
                if not np.isnan(value):
                    text = ax.text(
                        j, i, f"{value:.0f}",
                        ha="center", va="center",
                        color="black" if 30 < value < 70 else "white",
                        fontsize=10,
                        fontweight="bold",
                    )
        
        # Get DGP display name
        dgp_cls = DGP_CLASSES[dgp_name]
        dgp_instance = dgp_cls(d=5)
        dgp_title = dgp_instance.name
        
        # Add title for this subplot
        ax.set_title(
            f"DGP = {dgp_title}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        
        # Add grid
        ax.set_xticks(np.arange(n_metrics + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_encoders + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        
        # Only show x-axis label on bottom heatmap
        if idx == n_dgps - 1:
            ax.set_xlabel("Identifiability Metric", fontsize=10, fontweight="bold")
        
        # Y-axis label for all subplots
        ax.set_ylabel("Encoder", fontsize=10, fontweight="bold")
    
    # Add colorbar on the right side (spanning all heatmaps)
    cbar_ax = fig.add_subplot(gs[:n_dgps, 1])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Score (0-100)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Create timing/memory table at the bottom (per metric)
    table_ax = fig.add_subplot(gs[n_dgps, :])
    table_ax.axis('off')
    
    # Compute mean ± std for time and memory per metric
    metric_keys = list(metric_names.keys())
    metric_display = list(metric_names.values())
    
    time_row = []
    memory_row = []
    
    for metric_key in metric_keys:
        timings = metric_timing.get(metric_key, [])
        if timings:
            times = [t for t, _ in timings]
            memories = [m for _, m in timings]
            time_mean, time_std = np.mean(times), np.std(times)
            mem_mean, mem_std = np.mean(memories), np.std(memories)
            time_row.append(f"{time_mean*1000:.0f}±{time_std*1000:.0f}ms")
            memory_row.append(f"{mem_mean:.1f}±{mem_std:.1f}MB")
        else:
            time_row.append("N/A")
            memory_row.append("N/A")
    
    # Create table data: 2 rows (Time, Memory) × n_metrics columns
    table_data = [time_row, memory_row]
    row_labels = ["Time", "Memory"]
    
    # Create the table
    table = table_ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=metric_display,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    
    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(fontweight='bold', fontsize=8)
            cell.set_facecolor('#E6E6E6')
        elif col == -1:  # Row labels
            cell.set_text_props(fontweight='bold', fontsize=9)
            cell.set_facecolor('#E6E6E6')
    
    table_ax.set_title("Metric Timing & Memory (mean ± std)", fontsize=10, fontweight="bold", pad=5)
    
    # Overall title with samples/factors info
    fig.suptitle(
        f"Identifiability Metrics: DGP × Encoder Combinations\n"
        f"(samples={n_samples}, factors={n_factors})",
        fontsize=13,
        fontweight="bold",
        y=0.98
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
    all_results, metric_timing = evaluate_all_combinations(
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
        metric_timing=metric_timing,
        dgp_names=dgp_names,
        encoder_names=encoder_names,
        metric_names=METRIC_NAMES,
        n_samples=args.samples,
        n_factors=args.factors,
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
