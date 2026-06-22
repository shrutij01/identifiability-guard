#!/usr/bin/env python3
"""
Generate paper-ready figures for experiments 03, 06a, and 15.

Colour scheme is consistent with the teaser figure (fig1):
  MCC-P  = #c0392b   R²     = #27ae60
  DCI-D  = #2980b9   MCC-S  = #e74c3c

Usage
-----
    python experiments/plot_paper_figures.py                # all three
    python experiments/plot_paper_figures.py --only exp03   # one experiment
    python experiments/plot_paper_figures.py --list         # show available
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results_io import load_results

# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper_figures"

# ============================================================================
# Global matplotlib style  (matches teaser fig1)
# ============================================================================

plt.rcParams.update({
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "font.size":          8,
    "axes.labelsize":     10,
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "lines.linewidth":    1.5,
    "lines.markersize":   3.2,
    "axes.linewidth":     0.45,
    "xtick.major.width":  0.35,
    "ytick.major.width":  0.35,
    "xtick.major.size":   2,
    "ytick.major.size":   2,
    "figure.dpi":         300,
})

# ============================================================================
# Metric style — consistent with teaser fig1
# ============================================================================

MS = {
    "dci_disentanglement": {"label": "DCI-D",   "color": "#2980b9", "marker": "D",  "ls": "-",  "z": 4},
    "mcc_pearson":         {"label": "MCC-P",   "color": "#c0392b", "marker": "o",  "ls": "-",  "z": 5},
    "mcc_spearman":        {"label": "MCC-S",   "color": "#e74c3c", "marker": "p",  "ls": "--", "z": 4},
    "r2":                  {"label": r"R$^2$",  "color": "#27ae60", "marker": "^",  "ls": "-",  "z": 4},
    "infom":               {"label": "InfoM",   "color": "#8e44ad", "marker": "s",  "ls": "-.", "z": 3},
    "mcc_rdc":             {"label": "MCC-RDC", "color": "#e67e22", "marker": "h",  "ls": "-.", "z": 3},
    "mig":                 {"label": "MIG",     "color": "#7f8c8d", "marker": "v",  "ls": "-.", "z": 3},
    "tmex":                {"label": "T-MEX",   "color": "#f1c40f", "marker": "X",  "ls": ":",  "z": 3},
}

MET_MAIN = ["dci_disentanglement", "mcc_pearson", "mcc_spearman", "r2"]
MET_APX  = sorted(MS.keys())

# ============================================================================
# Helpers
# ============================================================================


def _style_ax(ax, ylim=(-0.04, 1.06)):
    """Clean axes: remove top/right spines, add faint 0/1 lines."""
    ax.set_ylim(*ylim)
    for y in [0, 1]:
        ax.axhline(y, color="0.87", ls=":", lw=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _metric_legend(fig, metrics, anchor=(0.53, -0.02), ncol=None):
    """Shared metric legend at bottom of figure."""
    if ncol is None:
        ncol = min(len(metrics), 8)
    handles = []
    for met in metrics:
        s = MS[met]
        h, = plt.plot([], [], color=s["color"], marker=s["marker"],
                       linestyle=s["ls"], lw=1.3, ms=4, label=s["label"])
        handles.append(h)
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", ncol=ncol,
        frameon=True, fancybox=False, edgecolor="0.85",
        bbox_to_anchor=anchor, fontsize=8.5,
        columnspacing=1.2, handletextpad=0.35,
    )


def _plot_lines(ax, x, means, ci_lo, ci_hi, metrics):
    """Plot metric lines with CI shading on a single axis."""
    for met in metrics:
        if met not in means:
            continue
        s = MS[met]
        y_m = np.asarray(means[met], dtype=float)
        y_l = np.asarray(ci_lo[met], dtype=float)
        y_h = np.asarray(ci_hi[met], dtype=float)
        valid = np.isfinite(y_m)
        xv = np.asarray(x)[valid]
        ax.plot(xv, y_m[valid], color=s["color"], marker=s["marker"],
                linestyle=s["ls"], zorder=s["z"], markersize=3.2)
        ax.fill_between(xv, y_l[valid], y_h[valid],
                        color=s["color"], alpha=0.15, zorder=1)


def _heatmap_annotate(ax, data, fontsize=6):
    """Annotate heatmap cells with numeric values."""
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            val = data[ii, jj]
            if np.isfinite(val):
                txt = f"{val:.2f}"
                color = "white" if val > 0.5 else "black"
            else:
                txt = "\u2014"
                color = "grey"
            ax.text(jj, ii, txt, ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")


def _save(fig, stem):
    """Save as PDF to paper_figures/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"  Saved: {path}")
    plt.close(fig)


def _try_load(exp_key):
    """Load results; return (data, config) or (None, None)."""
    try:
        return load_results(exp_key)
    except FileNotFoundError:
        print(f"  [skip] No results for {exp_key}")
        return None, None


def _tolist(v):
    """Ensure value is a plain Python list."""
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


# ============================================================================
# exp03 — Sign asymmetry across dimensionalities
#
#   3×2 grid:  rows = d ∈ {2, 5, 10},  cols = encoder ∈ {E1, E3}
#   x-axis = ρ,  y-axis = metric score with 95 % CI
# ============================================================================

EXP03_ENCODERS = ["E1", "E3"]
EXP03_D_VALUES = [2, 10]  # d=2 (full rho range) and d=10 (PSD-constrained)


def _load_exp03():
    all_d = {}
    for d in EXP03_D_VALUES:
        exp_key = "exp03" if d == 2 else f"exp03_d{d}"
        data, config = _try_load(exp_key)
        if data is None:
            continue
        rho_vals = config["correlation_values"]
        d_results = {}
        for enc in config["encoders"]:
            dd = data[enc]
            d_results[enc] = (
                rho_vals,
                {k: _tolist(v) for k, v in dd["means"].items()},
                {k: _tolist(v) for k, v in dd["ci_lo"].items()},
                {k: _tolist(v) for k, v in dd["ci_hi"].items()},
            )
        all_d[d] = d_results
    return all_d


def plot_exp03():
    """Sign asymmetry: 1×4 row — (d=2,E1), (d=2,E3), (d=10,E1), (d=10,E3)."""
    print("\n=== exp03: sign asymmetry across dimensionalities ===")
    all_d = _load_exp03()
    if not all_d:
        return
    d_vals = sorted(all_d.keys())

    # Build flat panel list: [(d, enc), ...]
    panels = [(d, enc) for d in d_vals for enc in EXP03_ENCODERS]
    n_panels = len(panels)

    for tag, metrics in [("main", MET_MAIN), ("apx", MET_APX)]:
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(3.0 * n_panels, 2.2),
            sharey=True, squeeze=False,
            gridspec_kw={"wspace": 0.12,
                         "left": 0.07, "right": 0.99,
                         "top": 0.88, "bottom": 0.28},
        )
        for col, (d, enc) in enumerate(panels):
            ax = axes[0, col]
            enc_results = all_d.get(d, {})
            min_rho = -1.0 / (d - 1) if d > 1 else -1.0
            if enc not in enc_results:
                ax.set_visible(False)
                continue
            rho_vals, means, ci_lo, ci_hi = enc_results[enc]
            _plot_lines(ax, rho_vals, means, ci_lo, ci_hi, metrics)
            ax.axvline(0, color="0.65", ls="--", lw=0.5, zorder=0)
            if min_rho > -0.99:
                ax.axvline(min_rho, color="#c0392b", ls=":", lw=0.4,
                           zorder=0, alpha=0.6)
            _style_ax(ax)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel(r"Correlation ($\boldsymbol{\rho}$)")
            ax.set_title(f"$\\mathbf{{d={d}}}$, {enc}", fontsize=9)
        axes[0, 0].set_ylabel(r"$\boldsymbol{\mathcal{M}}(\cdot)$", fontweight="bold")

        _metric_legend(fig, metrics)
        _save(fig, f"exp03_sign_asymmetry_{tag}")


# ============================================================================
# exp06a — Dropped variables
#
#   1×4 panels:  one per DGP (D1–D4)
#   x-axis = m/d (retained fraction),  y-axis = metric score with 95 % CI
# ============================================================================


def _load_exp06a():
    data, config = _try_load("exp06")
    if data is None:
        return None, None, None
    dgps = config["dgps"]
    m_values = config["m_values_6a"]
    d_total = config["d_total"]
    results = {}
    for dgp in dgps:
        dd = data["6a"][dgp]
        results[dgp] = (
            {k: _tolist(v) for k, v in dd["means"].items()},
            {k: _tolist(v) for k, v in dd["ci_lo"].items()},
            {k: _tolist(v) for k, v in dd["ci_hi"].items()},
        )
    return results, m_values, d_total


EXP06_DGPS_MAIN = ["D1", "D3", "D4"]
EXP06_MET_MAIN = ["dci_disentanglement", "mcc_pearson", "r2"]


def plot_exp06a():
    """Dropped variables: 1×3 panels (D1, D3, D4), m/d on x-axis."""
    print("\n=== exp06a: dropped variables ===")
    results, m_values, d_total = _load_exp06a()
    if results is None:
        return
    x_ratio = [m / d_total for m in m_values]

    for tag, metrics in [("main", EXP06_MET_MAIN), ("apx", MET_APX)]:
        dgps = EXP06_DGPS_MAIN if tag == "main" else list(results.keys())
        n = len(dgps)
        fig, axes = plt.subplots(
            1, n, figsize=(2.6 * n, 2.2),
            sharey=True, squeeze=False,
            gridspec_kw={"wspace": 0.10, "left": 0.08, "right": 0.99,
                         "top": 0.88, "bottom": 0.28},
        )
        for col, dgp in enumerate(dgps):
            ax = axes[0, col]
            means, ci_lo, ci_hi = results[dgp]
            _plot_lines(ax, x_ratio, means, ci_lo, ci_hi, metrics)
            ax.axvline(1.0, color="0.55", ls="--", lw=1.2, zorder=0)
            ax.text(0.95, 0.12, r"$m{=}d$", fontsize=8, color="0.40",
                    ha="right", transform=ax.transAxes)
            _style_ax(ax)
            ax.set_xlim(min(x_ratio) - 0.03, max(x_ratio) + 0.03)
            ax.set_xlabel(r"$\mathbf{m\,/\,d}$")
            ax.set_title(f"$\\mathbf{{{dgp}}}$", fontsize=9)
        axes[0, 0].set_ylabel(r"$\boldsymbol{\mathcal{M}}(\cdot)$", fontweight="bold")

        _metric_legend(fig, metrics)
        _save(fig, f"exp06a_dropped_variables_{tag}")


# ============================================================================
# exp09 — Ratio sweep (overcomplete representations)
#
#   1×2 panels: DCI-D and MCC-P
#   x-axis = m/d ratio,  one line per encoder (E1–E8) with 95 % CI
# ============================================================================

# Encoder visual style — distinct colours + markers
ES = {
    "E1": {"label": "E1", "color": "#3498db", "marker": "o"},
    "E3": {"label": "E3", "color": "#2c3e50", "marker": "D"},
    "E5": {"label": "E5", "color": "#e67e22", "marker": "s"},
    "E6": {"label": "E6", "color": "#27ae60", "marker": "^"},
    "E7": {"label": "E7", "color": "#c0392b", "marker": "v"},
    "E8": {"label": "E8", "color": "#8e44ad", "marker": "p"},
}

EXP09_METRICS = ["dci_disentanglement", "mcc_pearson"]


def _load_exp09_ratio():
    data, config = _try_load("exp09_ratio")
    if data is None:
        return None
    return data["sweep"]


def plot_exp09():
    """Ratio sweep: 1×2 panels (DCI-D, MCC-P), one line per encoder."""
    print("\n=== exp09: ratio sweep ===")
    sweep = _load_exp09_ratio()
    if sweep is None:
        return

    n_met = len(EXP09_METRICS)
    fig, axes = plt.subplots(
        1, n_met, figsize=(2.8 * n_met, 2.2),
        sharey=True, squeeze=False,
        gridspec_kw={"wspace": 0.12, "left": 0.10, "right": 0.99,
                     "top": 0.88, "bottom": 0.28},
    )

    max_ratio = max(
        r for enc_data in sweep.values() for r in enc_data["ratios"]
    )

    for col, met in enumerate(EXP09_METRICS):
        ax = axes[0, col]
        for enc in sweep:
            s = ES.get(enc, {"label": enc, "color": "#888", "marker": "o"})
            enc_data = sweep[enc]
            ratios = enc_data["ratios"]
            means = [enc_data["means"][met][i] for i in range(len(ratios))]
            lo = [enc_data["ci_lo"][met][i] for i in range(len(ratios))]
            hi = [enc_data["ci_hi"][met][i] for i in range(len(ratios))]
            ax.plot(ratios, means, marker=s["marker"], color=s["color"],
                    label=s["label"], markersize=3.5, lw=1.3)
            ax.fill_between(ratios, lo, hi, color=s["color"], alpha=0.12)
        ax.axvline(1.0, color="0.55", ls="--", lw=1.2, zorder=0)
        ax.text(1.05, 0.08, r"$m{=}d$", fontsize=7, color="0.40",
                transform=ax.get_xaxis_transform(), va="bottom")
        _style_ax(ax)
        ax.set_xlim(0.7, max_ratio + 0.3)
        ax.set_xlabel(r"$\mathbf{m\,/\,d}$")
        ax.set_title(f"$\\mathbf{{{MS[met]['label']}}}$", fontsize=9)

    axes[0, 0].set_ylabel(r"$\boldsymbol{\mathcal{M}}(\cdot)$", fontweight="bold")

    # Encoder legend
    handles = []
    for enc in sweep:
        s = ES.get(enc, {"label": enc, "color": "#888", "marker": "o"})
        h, = plt.plot([], [], color=s["color"], marker=s["marker"],
                       ls="-", lw=1.1, ms=3, label=s["label"])
        handles.append(h)
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", ncol=min(len(handles), 6),
        frameon=True, fancybox=False, edgecolor="0.85",
        bbox_to_anchor=(0.53, -0.02), fontsize=7.5,
        columnspacing=1.0, handletextpad=0.3,
    )
    _save(fig, "exp09_ratio_sweep_main")


# ============================================================================
# exp15 — Phase diagram
#
#   Heatmap grid:  one panel per metric
#   rows = m/d,  cols = m/n,  colour = null-encoder score (should be 0)
#   RdYlGn_r: green ≈ 0 (trustworthy), red > 0 (inflated)
# ============================================================================


def _load_exp15(enc_key):
    data, config = _try_load(enc_key)
    if data is None:
        return None, None
    grids = data.get("grids", data)
    return grids, config


def _plot_phase_heatmap(grids, metrics, md_ratios, mn_ratios, encoder, stem):
    n_met = len(metrics)
    fig, axes = plt.subplots(
        1, n_met,
        figsize=(3.2 * n_met + 0.6, 2.4),
        gridspec_kw={"wspace": 0.25,
                     "left": 0.08, "right": 0.88,
                     "top": 0.88, "bottom": 0.14},
    )
    if n_met == 1:
        axes = [axes]

    row_labels = [f"{md:.0f}" if md >= 1 else f"{md:.1f}" for md in md_ratios]
    col_labels = [f"{mn:.2f}" for mn in mn_ratios]

    im = None
    for idx, met in enumerate(metrics):
        ax = axes[idx]
        grid = np.asarray(grids.get(met, np.full((len(md_ratios), len(mn_ratios)), np.nan)))
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontweight="bold")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontweight="bold")
        ax.set_xlabel(r"$\mathbf{m\,/\,n}$", fontsize=11)
        ax.set_ylabel(r"$\mathbf{m\,/\,d}$", fontsize=11)
        ax.set_title(f"$\\mathbf{{{MS[met]['label']}}}$", fontsize=10)

    # Single shared colorbar on the right
    if im is not None:
        cax = fig.add_axes([0.91, 0.14, 0.02, 0.74])
        fig.colorbar(im, cax=cax)

    _save(fig, stem)


EXP15_MET_MAIN = ["mcc_pearson", "dci_disentanglement"]


def plot_exp15():
    """Phase diagram for E10 (main text only)."""
    print("\n=== exp15: phase diagram (E10) ===")
    grids, config = _load_exp15("exp15_e10")
    if grids is None:
        return
    md_ratios = config.get("md_ratios", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    mn_ratios = config.get("mn_ratios", [0.01, 0.05, 0.10, 0.50, 1.00, 5.00])
    _plot_phase_heatmap(grids, EXP15_MET_MAIN, md_ratios, mn_ratios,
                        "E10", "exp15_phase_diagram_e10_main")


# ============================================================================
# Registry & CLI
# ============================================================================

GENERATORS = {
    "exp03": plot_exp03,
    "exp06a": plot_exp06a,
    "exp09": plot_exp09,
    "exp15": plot_exp15,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-ready figures (exp03, exp06a, exp09, exp15).")
    parser.add_argument("--only", type=str, default=None,
                        help="Generate only this experiment.")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments.")
    args = parser.parse_args()

    if args.list:
        for k in sorted(GENERATORS):
            print(f"  {k}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        gen = GENERATORS.get(args.only)
        if gen is None:
            print(f"Unknown: {args.only}.  Available: {', '.join(sorted(GENERATORS))}")
            return
        gen()
    else:
        for gen in GENERATORS.values():
            gen()

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
