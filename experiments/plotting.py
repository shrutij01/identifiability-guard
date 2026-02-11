"""
Pretty, paper-style plotting helpers for experiments.

Design goals:
- Consistent font sizes and mathtext styling.
- Thick lines and readable markers.
- Modular helpers for bar, line, scatter, and heatmap plots.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

DEFAULT_STYLE = {
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "lines.linewidth": 2.5,
    "lines.markersize": 7,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
}


def set_pretty_style(overrides: Optional[Dict[str, float]] = None) -> None:
    """Apply a consistent, paper-style matplotlib theme."""
    mpl.rcParams.update(DEFAULT_STYLE)
    if overrides:
        mpl.rcParams.update(overrides)


def _mathify(text: str) -> str:
    if text is None:
        return text
    if "$" in text:
        return text
    return f"${text}$"


def apply_math_ticks(ax: plt.Axes, axis: str = "both") -> None:
    """Format numeric tick labels using mathtext."""
    fmt = FuncFormatter(lambda x, pos: rf"${x:g}$")
    if axis in ("both", "x"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("both", "y"):
        ax.yaxis.set_major_formatter(fmt)


def apply_axis_labels(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    math: bool = False,
) -> None:
    """Set axis labels with optional mathtext wrapping."""
    if title is not None:
        ax.set_title(_mathify(title) if math else title)
    if xlabel is not None:
        ax.set_xlabel(_mathify(xlabel) if math else xlabel)
    if ylabel is not None:
        ax.set_ylabel(_mathify(ylabel) if math else ylabel)


def _maybe_mathify_labels(labels: Sequence[str], math: bool) -> List[str]:
    if not math:
        return list(labels)
    return [_mathify(lbl) for lbl in labels]


# ---------------------------------------------------------------------------
# Bar plots
# ---------------------------------------------------------------------------

def plot_grouped_bars(
    ax: plt.Axes,
    x_labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    errors: Optional[Dict[str, Sequence[float]]] = None,
    colors: Optional[Dict[str, str]] = None,
    hatches: Optional[Dict[str, str]] = None,
    width: float = 0.18,
    edgecolor: str = "#3a3a3a",
    linewidth: float = 1.2,
    math_labels: bool = False,
    math_ticks: bool = True,
    legend: bool = True,
) -> plt.Axes:
    """Grouped bar chart with optional hatching and error bars."""
    labels = _maybe_mathify_labels(x_labels, math_labels)
    n_groups = len(x_labels)
    keys = list(series.keys())
    x = np.arange(n_groups)
    offset = (len(keys) - 1) * width / 2

    for i, key in enumerate(keys):
        vals = np.array(series[key])
        err = None if errors is None else np.array(errors.get(key, []))
        color = None if colors is None else colors.get(key)
        hatch = None if hatches is None else hatches.get(key)

        ax.bar(
            x - offset + i * width,
            vals,
            width=width,
            label=_mathify(key) if math_labels else key,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
            yerr=err,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if math_ticks:
        apply_math_ticks(ax, axis="y")
    if legend:
        ax.legend(frameon=True)
    return ax


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------

def plot_lines(
    ax: plt.Axes,
    x: Sequence[float],
    series: Dict[str, Sequence[float]],
    errors: Optional[Dict[str, Sequence[float]]] = None,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    math_labels: bool = False,
    math_ticks: bool = True,
    legend: bool = True,
) -> plt.Axes:
    """Multi-series line plot with optional error bands."""
    for key, y in series.items():
        color = None if colors is None else colors.get(key)
        marker = None if markers is None else markers.get(key, "o")
        y = np.array(y)
        ax.plot(
            x,
            y,
            label=_mathify(key) if math_labels else key,
            color=color,
            marker=marker,
        )
        if errors is not None and key in errors:
            err = np.array(errors[key])
            ax.fill_between(
                x,
                y - err,
                y + err,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    if math_ticks:
        apply_math_ticks(ax, axis="both")
    if legend:
        ax.legend(frameon=True)
    return ax


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def plot_scatter(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: str = "o",
    size: int = 40,
    alpha: float = 0.85,
    math_label: bool = False,
    math_ticks: bool = True,
) -> plt.Axes:
    """Scatter plot with consistent styling."""
    ax.scatter(
        x,
        y,
        label=_mathify(label) if (label and math_label) else label,
        color=color,
        marker=marker,
        s=size,
        alpha=alpha,
        edgecolors="none",
    )
    if math_ticks:
        apply_math_ticks(ax, axis="both")
    if label:
        ax.legend(frameon=True)
    return ax


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
    fmt: str = ".2f",
    math_labels: bool = False,
    math_ticks: bool = False,
    cbar: bool = True,
) -> plt.Axes:
    """Heatmap with optional annotations."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(_maybe_mathify_labels(x_labels, math_labels))
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(_maybe_mathify_labels(y_labels, math_labels))

    if math_ticks:
        apply_math_ticks(ax, axis="both")

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    format(data[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if data[i, j] > np.nanmean(data) else "black",
                )

    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax

