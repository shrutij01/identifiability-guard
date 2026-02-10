"""
Experiment 1 – Do all metrics recognise identifiability up to permutation and
rescaling correctly across DGP types?

Plot: DGPs (D1–D4) on the X axis, metric score on the Y axis, with one line
per metric.  Ideal outcome: flat curves (metric scores should be constant
across DGP types when the encoder is exactly identifiable up to permutation
and rescaling).

We use encoder E1 (elementwise linear ≡ permutation + rescaling) which is the
canonical "perfectly identifiable" encoder for all DGP types.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    plot_metrics_vs_xaxis_with_ci,
    savefig,
    setup_plot_style,
    DEFAULT_N_SAMPLES,
    DEFAULT_N_FACTORS,
    DEFAULT_N_SEEDS,
    DEFAULT_BASE_SEED,
    ALL_METRICS,
    METRIC_DISPLAY_NAMES,
    display_name,
    get_color,
    get_marker,
    make_registry,
)

# ── Configuration ──────────────────────────────────────────────────────────
DGP_NAMES = ["D1", "D2", "D3", "D4"]
ENCODER = "E1"  # exact elementwise linear → perfectly identifiable
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())


# ── Experiment logic ───────────────────────────────────────────────────────

def run_invariance_across_dgps(
    dgp_names=DGP_NAMES,
    encoder=ENCODER,
    n_samples=N_SAMPLES,
    n_factors=N_FACTORS,
    n_seeds=N_SEEDS,
    base_seed=BASE_SEED,
):
    """
    For each DGP type, evaluate metrics over multiple seeds with E1.

    Returns
    -------
    dgp_names : list[str]
    means, ci_lo, ci_hi : dict[str, list[float]]
        Keyed by metric name, values are lists aligned with *dgp_names*.
    """
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for dgp in dgp_names:
        print(f"  Evaluating {dgp} × {encoder} …")

        def eval_one_seed(seed, _dgp=dgp):
            return evaluate_dgp_encoder(
                _dgp, encoder,
                n_samples=n_samples, n_factors=n_factors, seed=seed,
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=n_seeds,
                                         base_seed=base_seed)

        for m in METRICS:
            stats = agg.get(m, {})
            means[m].append(stats.get("mean", np.nan))
            ci_lo[m].append(stats.get("ci_lower", np.nan))
            ci_hi[m].append(stats.get("ci_upper", np.nan))

    return dgp_names, means, ci_lo, ci_hi


def plot_invariance_across_dgps(dgp_names, means, ci_lo, ci_hi):
    """Create the main figure for Experiment 1."""
    fig = plot_metrics_vs_xaxis_with_ci(
        x_values=list(range(len(dgp_names))),
        means=means, ci_lo=ci_lo, ci_hi=ci_hi,
        xlabel="DGP type", ylabel="Metric score",
        title="Metric invariance to DGP type under exact element-wise identifiability (E1)",
        metrics_to_plot=METRICS,
    )
    # Replace numeric ticks with DGP names
    ax = fig.axes[0]
    ax.set_xticks(range(len(dgp_names)))
    ax.set_xticklabels(dgp_names)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 1 – Permutation / rescaling invariance across DGP types")
    print("=" * 70)
    dgp_names, means, ci_lo, ci_hi = run_invariance_across_dgps()
    fig = plot_invariance_across_dgps(dgp_names, means, ci_lo, ci_hi)
    savefig(fig, "exp01_invariance_across_dgps.pdf", subdir="exp01")
    savefig(
        plot_invariance_across_dgps(dgp_names, means, ci_lo, ci_hi),
        "exp01_invariance_across_dgps.png", subdir="exp01",
    )
    print("Done.")


if __name__ == "__main__":
    main()
