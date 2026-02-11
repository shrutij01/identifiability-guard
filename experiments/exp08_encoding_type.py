"""
Experiment 8 – What is the effect of the type of encoding of the factor in the
codes?

Compare D × E1 (elementwise linear baseline) against D × E5 (overcomplete
linear), D × E6 (overcomplete multicodes), and D × E8 (overcomplete disjoint
nonlinear subsets).

For each DGP (D1–D4) we produce a grouped bar chart where bars correspond to
encoders and each group of bars is one metric.
"""

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
ENCODERS = ["E1", "E5", "E6", "E8"]
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())

ENCODER_COLORS = {
    "E1": "#1f77b4",
    "E5": "#ff7f0e",
    "E6": "#2ca02c",
    "E8": "#d62728",
}


# ── Experiment logic ───────────────────────────────────────────────────────

def evaluate_all_combinations():
    """
    Returns
    -------
    scores : dict[dgp][encoder][metric] → float (mean over seeds)
    """
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


def plot_encoding_type_effect(scores):
    """One figure per DGP: grouped bar chart comparing encoders for each metric."""
    setup_plot_style()
    figs = {}
    for dgp in DGPS:
        n_metrics = len(METRICS)
        n_enc = len(ENCODERS)
        width = 0.8 / n_enc
        x = np.arange(n_metrics)

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, enc in enumerate(ENCODERS):
            vals = [scores[dgp][enc].get(m, np.nan) for m in METRICS]
            offset = (i - n_enc / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=enc,
                   color=ENCODER_COLORS.get(enc, "#888"))

        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in METRICS],
                           rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Metric score")
        ax.set_title(f"Effect of encoding type – {dgp}, d={N_FACTORS}")
        ax.legend(title="Encoder")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        figs[dgp] = fig
    return figs


def plot_encoding_type_combined(scores):
    """All DGPs in a single multi-panel figure."""
    setup_plot_style()
    n_dgps = len(DGPS)
    fig, axes = plt.subplots(1, n_dgps, figsize=(6 * n_dgps, 5), sharey=True)
    if n_dgps == 1:
        axes = [axes]

    n_metrics = len(METRICS)
    n_enc = len(ENCODERS)
    width = 0.8 / n_enc
    x = np.arange(n_metrics)

    for ax, dgp in zip(axes, DGPS):
        for i, enc in enumerate(ENCODERS):
            vals = [scores[dgp][enc].get(m, np.nan) for m in METRICS]
            offset = (i - n_enc / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=enc,
                   color=ENCODER_COLORS.get(enc, "#888"))
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in METRICS],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(dgp)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ENCODERS),
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Effect of encoding type: E1 vs E5, E6, E8\n"
        f"(d={N_FACTORS}, n={N_SAMPLES})", y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 8 – Encoding type effect (E1 vs E5, E6, E8)")
    print("=" * 70)
    scores = evaluate_all_combinations()

    figs = plot_encoding_type_effect(scores)
    for dgp, fig in figs.items():
        savefig(fig, f"exp08_encoding_type_{dgp}.pdf", subdir="exp08")

    fig_combined = plot_encoding_type_combined(scores)
    savefig(fig_combined, "exp08_encoding_type_combined.pdf", subdir="exp08")
    savefig(
        plot_encoding_type_combined(scores),
        "exp08_encoding_type_combined.png", subdir="exp08",
    )
    print("Done.")


if __name__ == "__main__":
    main()
