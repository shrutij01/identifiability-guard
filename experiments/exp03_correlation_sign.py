"""
Experiment 3 – Does the sign of correlation affect metric values despite
identical dependence strength?

Plot: metric scores varying correlation from −0.99 to +0.99, for several
entanglement strengths.  We use D2 (correlated DGP) × E1 (elementwise linear).
If metrics are well-behaved they should be symmetric around ρ=0 (same score for
+ρ and −ρ), because the sign of the correlation should not affect
identifiability.

We produce:
  • A single figure with correlation on the X axis and one line per metric
    (n_seeds averaged).
  • A multi-panel figure where each panel is a different encoder (E1 vs E2)
    to see if entanglement interacts with sign asymmetry.
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
from sensitivity import sweep_correlation

# ── Configuration ──────────────────────────────────────────────────────────
CORRELATION_VALUES = [
    -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99,
]
ENCODERS = ["E1", "E2"]  # E1 = elementwise linear, E2 = elementwise nonlinear
DGP = "D2"
N_SAMPLES = DEFAULT_N_SAMPLES
# Use d=2 so that the full ρ range (-1, 1) is feasible for a uniform
# correlation matrix (PSD requires ρ > -1/(d-1); for d=2 this is ρ > -1).
N_FACTORS = 2
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_correlation_sign(
    encoder: str,
    correlation_values=CORRELATION_VALUES,
    n_samples=N_SAMPLES,
    n_factors=N_FACTORS,
    n_seeds=N_SEEDS,
    base_seed=BASE_SEED,
):
    """Sweep ρ from −0.99 to +0.99 for a given encoder."""
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for rho in correlation_values:
        # Check if this ρ is feasible for a uniform correlation matrix of size d.
        # A d×d matrix with all off-diagonal entries ρ is PSD iff ρ > -1/(d-1).
        min_rho = -1.0 / (n_factors - 1) if n_factors > 1 else -1.0
        if rho <= min_rho:
            print(f"    ρ = {rho:+.2f} — skipped (not PSD for d={n_factors})")
            for m in METRICS:
                means[m].append(np.nan)
                ci_lo[m].append(np.nan)
                ci_hi[m].append(np.nan)
            continue
        print(f"    ρ = {rho:+.2f}")

        def eval_one_seed(seed, _rho=rho):
            return evaluate_dgp_encoder(
                DGP, encoder,
                n_samples=n_samples, n_factors=n_factors, seed=seed,
                dgp_kwargs={"correlation": _rho},
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=n_seeds,
                                         base_seed=base_seed)
        for m in METRICS:
            stats = agg.get(m, {})
            means[m].append(stats.get("mean", np.nan))
            ci_lo[m].append(stats.get("ci_lower", np.nan))
            ci_hi[m].append(stats.get("ci_upper", np.nan))

    return correlation_values, means, ci_lo, ci_hi


def plot_sign_effect_single(encoder, rho_vals, means, ci_lo, ci_hi):
    """Single plot for one encoder."""
    fig = plot_metrics_vs_xaxis_with_ci(
        rho_vals, means, ci_lo, ci_hi,
        xlabel="Correlation ρ",
        title=f"Metric sensitivity to correlation sign ({DGP} + {encoder})",
        ylim=(-0.05, 1.05),
    )
    ax = fig.axes[0]
    # Vertical line at ρ=0
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    # Ensure x-axis covers the full ρ range
    ax.set_xlim(rho_vals[0] - 0.03, rho_vals[-1] + 0.03)
    return fig


def plot_sign_effect_multi_encoder(all_results):
    """One subplot per encoder, all metrics overlaid."""
    setup_plot_style()
    n_enc = len(all_results)
    fig, axes = plt.subplots(1, n_enc, figsize=(7 * n_enc, 5), sharey=True)
    if n_enc == 1:
        axes = [axes]

    for ax, (enc, (rho_vals, means, ci_lo, ci_hi)) in zip(axes, all_results.items()):
        for m in METRICS:
            c = get_color(m)
            ax.plot(rho_vals, means[m], marker=get_marker(m), color=c,
                    label=display_name(m), markersize=4)
            ax.fill_between(rho_vals, ci_lo[m], ci_hi[m], color=c, alpha=0.12)
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Correlation ρ")
        ax.set_title(f"Encoder {enc}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(rho_vals[0] - 0.03, rho_vals[-1] + 0.03)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Does the sign of correlation affect metric values?\n"
        f"({DGP}, d={N_FACTORS}, n={N_SAMPLES})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 3 – Sign of correlation effect on metrics")
    print("=" * 70)

    all_results = {}
    for enc in ENCODERS:
        print(f"  Encoder: {enc}")
        rho_vals, means, ci_lo, ci_hi = sweep_correlation_sign(enc)
        all_results[enc] = (rho_vals, means, ci_lo, ci_hi)

        fig = plot_sign_effect_single(enc, rho_vals, means, ci_lo, ci_hi)
        savefig(fig, f"exp03_sign_correlation_{enc}.pdf", subdir="exp03")
        savefig(
            plot_sign_effect_single(enc, rho_vals, means, ci_lo, ci_hi),
            f"exp03_sign_correlation_{enc}.png", subdir="exp03",
        )

    fig_multi = plot_sign_effect_multi_encoder(all_results)
    savefig(fig_multi, "exp03_sign_correlation_all_encoders.pdf", subdir="exp03")
    savefig(
        plot_sign_effect_multi_encoder(all_results),
        "exp03_sign_correlation_all_encoders.png", subdir="exp03",
    )

    # Sensitivity sweeps: correlation for E1 and E2
    from pathlib import Path
    out_dir = Path(RESULTS_DIR / "exp03")
    for enc in ENCODERS:
        print(f"\n  Running sensitivity sweep: correlation for D2 × {enc} …")
        sweep_correlation(
            encoder=enc,
            correlation_values=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
            n_samples=N_SAMPLES,
            n_factors=N_FACTORS,
            n_seeds=N_SEEDS,
            base_seed=BASE_SEED,
            output_dir=out_dir,
            metrics_to_compute=set(ALL_METRICS.keys()),
        )

    print("Done.")


if __name__ == "__main__":
    main()
