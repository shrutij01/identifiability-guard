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
    MAIN_METRICS,
    APX_METRICS,
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

DEFAULT_N_SAMPLES = 1000
DEFAULT_N_FACTORS = 5
DEFAULT_N_SEEDS = 5
DEFAULT_BASE_SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "runs"

# Colour palette – one colour per metric for consistency across all plots.
METRIC_COLORS = {
    "dci_disentanglement": "#1f77b4",
    "mcc_pearson": "#d62728",
    "mcc_spearman": "#9467bd",
    "mcc_rdc": "#8c564b",
    "r2": "#e377c2",
    "mig": "#7f7f7f",
    "tmex": "#bcbd22",
    "infom": "#17becf",
}

# Marker cycle
METRIC_MARKERS = {
    "dci_disentanglement": "o",
    "mcc_pearson": "D",
    "mcc_spearman": "v",
    "mcc_rdc": "P",
    "r2": "*",
    "mig": "X",
    "tmex": "h",
    "infom": "p",
}

# Theory overlay styling
THEORY_COLOR = "#888888"
THEORY_LINESTYLE = "--"

# Colours and markers for d-value curves (collapse plots)
D_COLORS = {3: "#1b9e77", 5: "#d95f02", 10: "#7570b3", 20: "#e7298a"}
D_MARKERS = {3: "o", 5: "s", 10: "^", 20: "D"}

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    mpl.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
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
    train_fraction: float = 0.8,
) -> Dict[str, float]:
    """
    Evaluate all requested metrics for a single DGP × Encoder combination.

    Data is split into train/test (controlled by *train_fraction*).  Metrics
    that fit a model (R², InfoE) are trained on the train split and scored on
    held-out test data.  Pure-statistic metrics evaluate on the test split.

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
    train_fraction : float
        Fraction of samples used for training (default 0.8).

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

    # Use a distinct seed for the encoder to prevent seed coupling.
    # With the same seed, null encoders like E9 (Gaussian) produce Z_hat == Z
    # because both RNGs start from the same state and call equivalent functions.
    encoder_seed = seed + 1_000_000
    encoder = create_encoder_with_params(encoder_name, n_factors, encoder_seed, params)
    Z_hat = encoder.encode(Z)

    # Train/test split
    n_train = int(n_samples * train_fraction)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    Z_train, Z_hat_train = Z[train_idx], Z_hat[train_idx]
    Z_test, Z_hat_test = Z[test_idx], Z_hat[test_idx]

    # Map user-facing metric names to registry-level names so we only
    # compute what's actually needed (e.g. dci_disentanglement → dci).
    _METRIC_TO_REGISTRY = {
        'dci_disentanglement': 'dci', 'dci_completeness': 'dci',
        'dci_informativeness': 'dci',
    }
    registry_names = list({
        _METRIC_TO_REGISTRY.get(m, m) for m in metrics_to_compute
    })

    all_results = registry.compute_all_oos(
        Z_train, Z_hat_train, Z_test, Z_hat_test,
        metric_names=registry_names,
    )
    return extract_metric_scores(all_results, metrics_to_compute)


def evaluate_with_arrays(
    Z: np.ndarray,
    Z_hat: np.ndarray,
    metrics_to_compute: Optional[Set[str]] = None,
    registry: Optional[MetricRegistry] = None,
    Z_train: Optional[np.ndarray] = None,
    Z_hat_train: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute metrics given pre-built arrays.

    When *Z_train* and *Z_hat_train* are provided, model-based metrics fit on
    those arrays and evaluate on (Z, Z_hat) as the held-out test set.
    Otherwise all metrics evaluate on (Z, Z_hat) only (original behaviour).
    """
    if metrics_to_compute is None:
        metrics_to_compute = set(ALL_METRICS.keys())
    if registry is None:
        registry = make_registry()

    if Z_train is not None and Z_hat_train is not None:
        all_results = registry.compute_all_oos(
            Z_train, Z_hat_train, Z, Z_hat,
        )
    else:
        all_results = registry.compute_all(Z, Z_hat)
    return extract_metric_scores(all_results, metrics_to_compute)


def multi_seed_evaluate(
    eval_one_seed: Callable[[int], Dict[str, float]],
    n_seeds: int = DEFAULT_N_SEEDS,
    base_seed: int = DEFAULT_BASE_SEED,
    n_jobs: int = 1,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """Convenience wrapper around ``run_multi_seed_evaluation``.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers for the seed loop.  ``1`` = sequential
        (default), ``-1`` = all available cores.
    """
    return run_multi_seed_evaluation(
        eval_one_seed, n_seeds=n_seeds, base_seed=base_seed, verbose=False,
        n_jobs=n_jobs,
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
        ax.fill_between(x_values, ci_lo[m], ci_hi[m], color=c, alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sweep_split(
    x_values: List[Any],
    means: Dict[str, List[float]],
    ci_lo: Dict[str, List[float]],
    ci_hi: Dict[str, List[float]],
    invariant_metrics: List[str],
    varying_metrics: List[str],
    xlabel: str,
    title_left: str = "Invariant metrics",
    title_right: str = "Varying metrics",
    ylabel: str = "Metric score",
    figsize: Tuple[float, float] = (12, 5),
    xscale: Optional[str] = None,
    xscale_kwargs: Optional[Dict[str, Any]] = None,
    ref_lines: Optional[List[Tuple[str, float, str]]] = None,
    theory_lines: Optional[Dict[str, Tuple[List, List, str]]] = None,
    ylim: Optional[Tuple[float, float]] = (-0.05, 1.05),
) -> plt.Figure:
    """1x2 split sweep: left panel shows stable metrics, right panel shows varying ones.

    Parameters
    ----------
    invariant_metrics : list of str
        Metrics expected to stay constant (plotted on left panel).
    varying_metrics : list of str
        Metrics expected to vary (plotted on right panel).
    ref_lines : list of (orientation, value, label), optional
        ``'v'`` for axvline, ``'h'`` for axhline.
    theory_lines : dict mapping metric_name -> (x_vals, y_vals, label), optional
        Dashed theory overlays on both panels.
    """
    setup_plot_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, metric_list, title in [
        (ax_left, invariant_metrics, title_left),
        (ax_right, varying_metrics, title_right),
    ]:
        for m in metric_list:
            if m not in means:
                continue
            c = get_color(m)
            ax.plot(x_values, means[m], marker=get_marker(m), color=c,
                    label=display_name(m), markersize=5)
            ax.fill_between(x_values, ci_lo[m], ci_hi[m], color=c, alpha=0.25)
            # Theory overlay for this metric
            if theory_lines and m in theory_lines:
                tx, ty, tlabel = theory_lines[m]
                ax.plot(tx, ty, color=THEORY_COLOR, linestyle=THEORY_LINESTYLE,
                        lw=1.5, label=tlabel)
        if xscale:
            kw = xscale_kwargs or {}
            ax.set_xscale(xscale, **kw)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)
        # Reference lines
        if ref_lines:
            for orient, val, lab in ref_lines:
                if orient == "v":
                    ax.axvline(val, color="grey", ls="--", lw=0.8, label=lab)
                else:
                    ax.axhline(val, color="grey", ls="--", lw=0.8, label=lab)

    ax_left.set_ylabel(ylabel)
    # Unified legend below both panels
    handles, labels = [], []
    for ax in (ax_left, ax_right):
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def plot_collapse(
    metric_name: str,
    curves: List[Tuple],
    xlabel: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xscale: Optional[str] = None,
    xscale_kwargs: Optional[Dict[str, Any]] = None,
    theory_line: Optional[Tuple] = None,
    ylim: Optional[Tuple[float, float]] = (-0.05, 1.05),
) -> plt.Axes:
    """Plot multiple overlaid curves for a single metric on one axes.

    Parameters
    ----------
    curves : list of (x_vals, means, ci_lo, ci_hi, label, color, marker)
    theory_line : (x_vals, y_vals, label), optional
    """
    if ax is None:
        _, ax = plt.subplots()
    for x_vals, m_vals, lo, hi, label, color, marker in curves:
        ax.plot(x_vals, m_vals, marker=marker, color=color, label=label,
                markersize=6)
        ax.fill_between(x_vals, lo, hi, color=color, alpha=0.12)
    if theory_line:
        tx, ty, tlabel = theory_line
        ax.plot(tx, ty, color=THEORY_COLOR, linestyle=THEORY_LINESTYLE,
                lw=1.5, label=tlabel)
    if xscale:
        kw = xscale_kwargs or {}
        ax.set_xscale(xscale, **kw)
    ax.set_xlabel(xlabel)
    if title is None:
        title = display_name(metric_name)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def plot_collapse_grid(
    metrics: List[str],
    curves_per_metric: Dict[str, List[Tuple]],
    xlabel: str,
    suptitle: str,
    ncols: int = 4,
    xscale: Optional[str] = None,
    xscale_kwargs: Optional[Dict[str, Any]] = None,
    theory_lines: Optional[Dict[str, Tuple]] = None,
    ylim: Optional[Tuple[float, float]] = (-0.05, 1.05),
    figsize_per_cell: Tuple[float, float] = (5, 4),
) -> plt.Figure:
    """Grid of collapse plots — one panel per metric, multiple curves per panel.

    Parameters
    ----------
    curves_per_metric : dict mapping metric_name -> list of curve tuples
        Each curve: (x_vals, means, ci_lo, ci_hi, label, color, marker)
    theory_lines : dict mapping metric_name -> (x_vals, y_vals, label), optional
    """
    setup_plot_style()
    n_metrics = len(metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_cell[0] * ncols,
                                      figsize_per_cell[1] * nrows),
                             sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, met in enumerate(metrics):
        ax = axes_flat[idx]
        theory = theory_lines.get(met) if theory_lines else None
        curves = curves_per_metric.get(met, [])
        plot_collapse(met, curves, xlabel, ax=ax, xscale=xscale,
                      xscale_kwargs=xscale_kwargs, theory_line=theory,
                      ylim=ylim)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Metric score")
    fig.suptitle(suptitle, y=1.02, fontsize=13)
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
    center: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fmt: str = ".2f",
) -> plt.Figure:
    """Generic annotated heatmap.

    Parameters
    ----------
    center : float, optional
        When provided, use a diverging colormap centered on this value.
        ``vmin`` and ``vmax`` are set symmetrically around *center* if not
        already specified.  The default *cmap* switches to ``'RdYlGn'``
        (green = center = correct, red = far from center = broken).
    """
    setup_plot_style()
    if center is not None:
        finite = data[np.isfinite(data)]
        if len(finite) > 0:
            max_dev = max(abs(finite.max() - center), abs(finite.min() - center))
        else:
            max_dev = 1.0
        if vmin is None:
            vmin = center - max_dev
        if vmax is None:
            vmax = center + max_dev
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
