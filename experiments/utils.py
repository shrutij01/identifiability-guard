"""
Shared utilities for all experiment scripts.

Provides common evaluation, plotting, and I/O helpers to avoid code duplication
across the individual experiment files.
"""

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from identifiability_guard.metrics import MetricRegistry
from identifiability_guard.evaluation import (
    sensitivity_analysis_1d,
    compute_sensitivity_statistics,
    DGP_CLASSES,
    ENCODER_CLASSES,
    ALL_METRICS,
    DEFAULT_METRICS,
    METRIC_DISPLAY_NAMES,
    create_dgp_with_params,
    create_encoder_with_params,
    extract_metric_scores,
)
from identifiability_guard.evaluation.multi_seed import run_multi_seed_evaluation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_SAMPLES = 100
DEFAULT_N_FACTORS = 5
DEFAULT_N_SEEDS = 5
DEFAULT_BASE_SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "runs"

# Colour palette – one colour per metric for consistency across all plots.
METRIC_COLORS = {
    "dci_disentanglement": "#1f77b4",
    "dci_completeness": "#ff7f0e",
    "dci_informativeness": "#2ca02c",
    "mcc_pearson": "#d62728",
    "mcc_spearman": "#9467bd",
    "mcc_rdc": "#8c564b",
    "r2": "#e377c2",
    "mig": "#7f7f7f",
    "tmex": "#bcbd22",
    "infom": "#17becf",
    "infoe": "#aec7e8",
    "infoc": "#ffbb78",
}

# Marker cycle
METRIC_MARKERS = {
    "dci_disentanglement": "o",
    "dci_completeness": "s",
    "dci_informativeness": "^",
    "mcc_pearson": "D",
    "mcc_spearman": "v",
    "mcc_rdc": "P",
    "r2": "*",
    "mig": "X",
    "tmex": "h",
    "infom": "p",
    "infoe": "<",
    "infoc": ">",
}

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def get_color(metric_name: str) -> str:
    return METRIC_COLORS.get(metric_name, "#333333")


def get_marker(metric_name: str) -> str:
    return METRIC_MARKERS.get(metric_name, "o")


def display_name(metric_name: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def make_registry() -> MetricRegistry:
    """Return a ready-to-use MetricRegistry with all defaults registered."""
    reg = MetricRegistry()
    reg.register_defaults()
    return reg


def evaluate_dgp_encoder(
    dgp_name: str,
    encoder_name: str,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_factors: int = DEFAULT_N_FACTORS,
    seed: int = DEFAULT_BASE_SEED,
    dgp_kwargs: Optional[Dict[str, Any]] = None,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    metrics_to_compute: Optional[Set[str]] = None,
    registry: Optional[MetricRegistry] = None,
) -> Dict[str, float]:
    """
    Evaluate all requested metrics for a single DGP × Encoder combination.

    Parameters
    ----------
    dgp_name : str
        Key in DGP_CLASSES, e.g. ``'D1'``.
    encoder_name : str
        Key in ENCODER_CLASSES, e.g. ``'E1'``.
    n_samples, n_factors, seed : int
        Standard experiment dimensions.
    dgp_kwargs, encoder_kwargs : dict, optional
        Extra constructor arguments forwarded to ``create_dgp_with_params`` /
        ``create_encoder_with_params``.
    metrics_to_compute : set of str, optional
        Subset of ``ALL_METRICS`` keys.  ``None`` → all.
    registry : MetricRegistry, optional
        Reuse an existing registry instance.

    Returns
    -------
    dict
        ``{metric_name: score}``
    """
    if metrics_to_compute is None:
        metrics_to_compute = set(ALL_METRICS.keys())
    if registry is None:
        registry = make_registry()
    if dgp_kwargs is None:
        dgp_kwargs = {}
    if encoder_kwargs is None:
        encoder_kwargs = {}

    # Build params dict expected by create_dgp_with_params / create_encoder_with_params
    params: Dict[str, Any] = {**dgp_kwargs, **encoder_kwargs}

    dgp = create_dgp_with_params(dgp_name, n_factors, seed, params)
    Z = dgp.sample(n_samples)

    encoder = create_encoder_with_params(encoder_name, n_factors, seed, params)
    Z_hat = encoder.encode(Z)

    all_results = registry.compute_all(Z, Z_hat)
    return extract_metric_scores(all_results, metrics_to_compute)


def evaluate_with_arrays(
    Z: np.ndarray,
    Z_hat: np.ndarray,
    metrics_to_compute: Optional[Set[str]] = None,
    registry: Optional[MetricRegistry] = None,
) -> Dict[str, float]:
    """Compute metrics given pre-built arrays."""
    if metrics_to_compute is None:
        metrics_to_compute = set(ALL_METRICS.keys())
    if registry is None:
        registry = make_registry()
    all_results = registry.compute_all(Z, Z_hat)
    return extract_metric_scores(all_results, metrics_to_compute)


def multi_seed_evaluate(
    eval_one_seed: Callable[[int], Dict[str, float]],
    n_seeds: int = DEFAULT_N_SEEDS,
    base_seed: int = DEFAULT_BASE_SEED,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """Convenience wrapper around ``run_multi_seed_evaluation``."""
    return run_multi_seed_evaluation(
        eval_one_seed, n_seeds=n_seeds, base_seed=base_seed, verbose=False,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def savefig(fig: plt.Figure, name: str, subdir: str = ""):
    """Save a figure to ``RESULTS_DIR / subdir / name``."""
    out = RESULTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def plot_metrics_vs_xaxis(
    x_values: List[Any],
    all_scores: Dict[str, List[float]],
    xlabel: str,
    title: str,
    ylabel: str = "Metric score",
    metrics_to_plot: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    ylim: Optional[Tuple[float, float]] = (-0.05, 1.05),
) -> plt.Figure:
    """
    Line plot with one line per metric sharing the same axes.

    Parameters
    ----------
    x_values : list
        Tick positions on the x-axis.
    all_scores : dict
        ``{metric_name: [score_for_each_x_value]}``.
    """
    setup_plot_style()
    if metrics_to_plot is None:
        metrics_to_plot = list(all_scores.keys())

    fig, ax = plt.subplots(figsize=figsize)
    for m in metrics_to_plot:
        if m not in all_scores:
            continue
        ax.plot(
            x_values,
            all_scores[m],
            marker=get_marker(m),
            color=get_color(m),
            label=display_name(m),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metrics_vs_xaxis_with_ci(
    x_values: List[Any],
    means: Dict[str, List[float]],
    ci_lo: Dict[str, List[float]],
    ci_hi: Dict[str, List[float]],
    xlabel: str,
    title: str,
    ylabel: str = "Metric score",
    metrics_to_plot: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    ylim: Optional[Tuple[float, float]] = (-0.05, 1.05),
) -> plt.Figure:
    """Line plot with confidence bands (one line per metric, shared axes)."""
    setup_plot_style()
    if metrics_to_plot is None:
        metrics_to_plot = list(means.keys())

    fig, ax = plt.subplots(figsize=figsize)
    for m in metrics_to_plot:
        if m not in means:
            continue
        c = get_color(m)
        ax.plot(x_values, means[m], marker=get_marker(m), color=c,
                label=display_name(m))
        ax.fill_between(x_values, ci_lo[m], ci_hi[m], color=c, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fmt: str = ".2f",
) -> plt.Figure:
    """Generic annotated heatmap."""
    setup_plot_style()
    if figsize is None:
        figsize = (max(6, len(col_labels) * 0.9), max(4, len(row_labels) * 0.6))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            txt = f"{val:{fmt}}" if np.isfinite(val) else "—"
            color = "white" if (np.isfinite(val) and abs(val - (vmin or 0)) < 0.3 * ((vmax or 1) - (vmin or 0))) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_grouped_bar(
    group_labels: List[str],
    series: Dict[str, List[float]],
    title: str,
    xlabel: str = "",
    ylabel: str = "Metric score",
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Grouped bar chart.  Each key in *series* is a legend entry."""
    setup_plot_style()
    n_groups = len(group_labels)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=display_name(name),
               color=get_color(name))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=30, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", ncol=3, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
