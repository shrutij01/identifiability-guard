"""
Experiment 9 – Can metrics distinguish correctly between equivalence classes
in overcomplete representations?

Comparisons:
  • D × E2 vs D × E7   (elementwise nonlinear vs overcomplete entangled)
  • D × E7 vs D × E5–E8 (overcomplete entangled vs other overcomplete encoders)

For each DGP (D1–D4) we compare how metrics rank the different overcomplete
encoders.  A grouped bar chart shows encoders on the X axis and metric scores
as grouped bars, one colour per metric.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    savefig,
    setup_plot_style,
    make_registry,
    display_name,
    get_color,
    DEFAULT_N_SAMPLES, DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    ALL_METRICS,
)

# ── Configuration ──────────────────────────────────────────────────────────
DGPS = ["D1", "D2", "D3", "D4"]
# E2 = exact elementwise nonlinear (square, m=d)
# E5-E8 = overcomplete variants
ENCODERS = ["E2", "E5", "E6", "E7", "E8"]
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())

ENCODER_COLORS = {
    "E2": "#1f77b4",
    "E5": "#ff7f0e",
    "E6": "#2ca02c",
    "E7": "#d62728",
    "E8": "#9467bd",
}


# ── Experiment logic ───────────────────────────────────────────────────────

def evaluate_all_combinations():
    """Compute mean metric scores for every DGP × encoder pair."""
    registry = make_registry()
    scores = {}
    for dgp in DGPS:
        scores[dgp] = {}
        for enc in ENCODERS:
            print(f"    {dgp} × {enc}")

            def eval_one_seed(seed, _dgp=dgp, _enc=enc):
                return evaluate_dgp_encoder(
                    _dgp, _enc,
                    n_samples=N_SAMPLES, n_factors=N_FACTORS, seed=seed,
                    registry=registry,
                )

            _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                             base_seed=BASE_SEED)
            scores[dgp][enc] = {
                met: agg.get(met, {}).get("mean", np.nan) for met in METRICS
            }
    return scores


def plot_overcomplete_comparison(scores):
    """Multi-panel figure: one panel per DGP, encoders as groups, metrics as bars."""
    setup_plot_style()
    n_dgps = len(DGPS)
    fig, axes = plt.subplots(1, n_dgps, figsize=(6 * n_dgps, 5), sharey=True)
    if n_dgps == 1:
        axes = [axes]

    n_enc = len(ENCODERS)
    n_metrics = len(METRICS)
    width = 0.8 / n_metrics
    x = np.arange(n_enc)

    for ax, dgp in zip(axes, DGPS):
        for j, met in enumerate(METRICS):
            vals = [scores[dgp][enc].get(met, np.nan) for enc in ENCODERS]
            offset = (j - n_metrics / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=display_name(met),
                   color=get_color(met))
        ax.set_xticks(x)
        ax.set_xticklabels(ENCODERS, fontsize=9)
        ax.set_title(dgp)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Overcomplete representations: E2 vs E5–E8\n"
        f"(d={N_FACTORS}, n={N_SAMPLES})", y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_e2_vs_e7(scores):
    """Focused comparison: E2 (exact nonlinear) vs E7 (overcomplete entangled)."""
    setup_plot_style()
    fig, axes = plt.subplots(1, len(DGPS), figsize=(5 * len(DGPS), 4.5), sharey=True)
    if len(DGPS) == 1:
        axes = [axes]

    pair = ["E2", "E7"]
    n_pair = len(pair)
    width = 0.35
    x = np.arange(len(METRICS))

    for ax, dgp in zip(axes, DGPS):
        for i, enc in enumerate(pair):
            vals = [scores[dgp][enc].get(m, np.nan) for m in METRICS]
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=enc,
                   color=ENCODER_COLORS[enc])
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in METRICS],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(dgp)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("E2 (exact nonlinear) vs E7 (overcomplete entangled)", y=1.02)
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 9 – Overcomplete representations (E2 vs E5–E8)")
    print("=" * 70)
    scores = evaluate_all_combinations()

    fig1 = plot_overcomplete_comparison(scores)
    savefig(fig1, "exp09_overcomplete_all.pdf", subdir="exp09")
    savefig(
        plot_overcomplete_comparison(scores),
        "exp09_overcomplete_all.png", subdir="exp09",
    )

    fig2 = plot_e2_vs_e7(scores)
    savefig(fig2, "exp09_e2_vs_e7.pdf", subdir="exp09")
    savefig(
        plot_e2_vs_e7(scores),
        "exp09_e2_vs_e7.png", subdir="exp09",
    )
    print("Done.")


if __name__ == "__main__":
    main()
