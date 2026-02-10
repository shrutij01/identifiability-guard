"""
Experiment 2 – How sensitive are metrics to the difference between elementwise
rescaling v/s invertible non-linear transformations?

Plot: vary the strength of elementwise non-linearity (and type of non-linearity)
on the X axis, metric values on the Y axis.  The spectrum goes from linear
(nonlinearity_strength=0, equivalent to E1) to fully non-linear
(nonlinearity_strength=1, full E2).

For each non-linearity type we show a separate panel/subplot.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    plot_metrics_vs_xaxis_with_ci,
    savefig,
    setup_plot_style,
    make_registry,
    get_color, get_marker, display_name,
    DEFAULT_N_SAMPLES, DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    ALL_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_encoder_nonlinearity
from src.encoders.e2_elementwise_nonlinear import E2ElementwiseNonlinear
from src.encoders.base import BaseEncoder

# ── Configuration ──────────────────────────────────────────────────────────
DGP = "D1"  # independent – isolates encoder effect
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())

# Strengths from linear to fully non-linear
NONLINEARITY_STRENGTHS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Use the default invertible nonlinear functions from the encoder base class.
# Build a name → function mapping for cycling through them.
_DEFAULT_FNS = BaseEncoder._get_default_nonlinear_invertible_functions()
NONLINEARITY_TYPES = {
    "tanh":        _DEFAULT_FNS[0],
    "sinh":        _DEFAULT_FNS[1],
    # "signed_sqrt": _DEFAULT_FNS[2],
    # "cube":        _DEFAULT_FNS[3],
    # "fifth_power": _DEFAULT_FNS[4],
    # "exp":         _DEFAULT_FNS[5],
    # "signed_log":  _DEFAULT_FNS[6],
}


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_nonlinearity_strength(
    nonlinear_fn,
    fn_name: str,
    dgp=DGP,
    strengths=NONLINEARITY_STRENGTHS,
    n_samples=N_SAMPLES,
    n_factors=N_FACTORS,
    n_seeds=N_SEEDS,
    base_seed=BASE_SEED,
):
    """
    Sweep nonlinearity_strength from 0 → 1 for a single non-linearity type.

    Returns
    -------
    strengths, means, ci_lo, ci_hi (same layout as exp01)
    """
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for alpha in strengths:
        print(f"    {fn_name}  α = {alpha:.2f}")

        def eval_one_seed(seed, _alpha=alpha):
            return evaluate_dgp_encoder(
                dgp, "E2",
                n_samples=n_samples, n_factors=n_factors, seed=seed,
                encoder_kwargs={
                    "nonlinearity_strength": _alpha,
                },
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=n_seeds,
                                         base_seed=base_seed)
        for m in METRICS:
            stats = agg.get(m, {})
            means[m].append(stats.get("mean", np.nan))
            ci_lo[m].append(stats.get("ci_lower", np.nan))
            ci_hi[m].append(stats.get("ci_upper", np.nan))

    return strengths, means, ci_lo, ci_hi


def run_all_nonlinearity_types():
    """Run sweep for each non-linearity type.  Returns dict of results."""
    results = {}
    for fn_name, fn in NONLINEARITY_TYPES.items():
        print(f"  Non-linearity type: {fn_name}")
        strengths, means, ci_lo, ci_hi = sweep_nonlinearity_strength(fn, fn_name)
        results[fn_name] = (strengths, means, ci_lo, ci_hi)
    return results


def plot_nonlinearity_sweep(results):
    """One subplot per non-linearity type, all metrics overlaid."""
    setup_plot_style()
    n_types = len(results)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5), sharey=True)
    if n_types == 1:
        axes = [axes]

    for ax, (fn_name, (strengths, means, ci_lo, ci_hi)) in zip(axes, results.items()):
        for m in METRICS:
            c = get_color(m)
            ax.plot(strengths, means[m], marker=get_marker(m), color=c,
                    label=display_name(m), markersize=4)
            ax.fill_between(strengths, ci_lo[m], ci_hi[m], color=c, alpha=0.12)
        ax.set_xlabel("Nonlinearity strength α")
        ax.set_title(fn_name)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score")
    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Metric sensitivity to elementwise non-linearity strength\n"
        f"({DGP} + E2, n={N_SAMPLES}, d={N_FACTORS})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# Also produce a single-panel plot averaging over all nonlinearity types
def plot_nonlinearity_sweep_combined(results):
    """Average across nonlinearity types → single panel."""
    setup_plot_style()
    strengths = NONLINEARITY_STRENGTHS
    combined_means = {m: np.zeros(len(strengths)) for m in METRICS}
    combined_ci_lo = {m: np.zeros(len(strengths)) for m in METRICS}
    combined_ci_hi = {m: np.zeros(len(strengths)) for m in METRICS}
    n = len(results)
    for fn_name, (_, means, ci_lo, ci_hi) in results.items():
        for m in METRICS:
            combined_means[m] += np.array(means[m])
            combined_ci_lo[m] += np.array(ci_lo[m])
            combined_ci_hi[m] += np.array(ci_hi[m])
    for m in METRICS:
        combined_means[m] /= n
        combined_ci_lo[m] /= n
        combined_ci_hi[m] /= n

    fig = plot_metrics_vs_xaxis_with_ci(
        strengths, combined_means, combined_ci_lo, combined_ci_hi,
        xlabel="Nonlinearity strength α",
        title="Metric sensitivity to non-linearity (averaged over function types)\n"
              f"({DGP} + E2)",
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 2 – Elementwise rescaling vs non-linear transformations")
    print("=" * 70)
    results = run_all_nonlinearity_types()
    fig1 = plot_nonlinearity_sweep(results)
    savefig(fig1, "exp02_nonlinearity_per_type.pdf", subdir="exp02")
    savefig(
        plot_nonlinearity_sweep(results),
        "exp02_nonlinearity_per_type.png", subdir="exp02",
    )
    fig2 = plot_nonlinearity_sweep_combined(results)
    savefig(fig2, "exp02_nonlinearity_combined.pdf", subdir="exp02")
    savefig(
        plot_nonlinearity_sweep_combined(results),
        "exp02_nonlinearity_combined.png", subdir="exp02",
    )

    # Sensitivity sweep: encoder nonlinearity for D1 × E2 (tanh_modified)
    print("\n  Running sensitivity sweep: encoder nonlinearity for D1 × E2 …")
    from pathlib import Path
    sweep_encoder_nonlinearity(
        dgp="D1",
        nonlinearity_values=NONLINEARITY_STRENGTHS,
        n_samples=N_SAMPLES,
        n_factors=N_FACTORS,
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED,
        output_dir=Path(RESULTS_DIR / "exp02"),
        metrics_to_compute=set(ALL_METRICS.keys()),
        nonlinearity_type="tanh_modified",
    )

    print("Done.")


if __name__ == "__main__":
    main()
