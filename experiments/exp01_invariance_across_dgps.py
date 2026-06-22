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

import matplotlib.pyplot as plt
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
    MAIN_METRICS,
    APX_METRICS,
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
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Multi-d configuration: verify invariance at higher d
D_VALUES = [5, 10, 20]


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


def plot_invariance_across_dgps(dgp_names, means, ci_lo, ci_hi, metrics=None):
    """Create the main figure for Experiment 1."""
    if metrics is None:
        metrics = METRICS
    fig = plot_metrics_vs_xaxis_with_ci(
        x_values=list(range(len(dgp_names))),
        means=means, ci_lo=ci_lo, ci_hi=ci_hi,
        xlabel="DGP type", ylabel="Metric score",
        title="Metric invariance to DGP type under exact element-wise identifiability (E1)",
        metrics_to_plot=metrics,
    )
    # Replace numeric ticks with DGP names
    ax = fig.axes[0]
    ax.set_xticks(range(len(dgp_names)))
    ax.set_xticklabels(dgp_names)
    return fig


def plot_invariance_multi_d(all_d_results, metrics=None):
    """Multi-panel figure: one panel per d value."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    d_vals = sorted(all_d_results.keys())
    n_d = len(d_vals)
    fig, axes = plt.subplots(1, n_d, figsize=(6 * n_d, 5), sharey=True)
    if n_d == 1:
        axes = [axes]

    for ax, d in zip(axes, d_vals):
        dgp_names, means, ci_lo, ci_hi = all_d_results[d]
        for m in metrics:
            c = get_color(m)
            ax.plot(range(len(dgp_names)), means[m], marker=get_marker(m), color=c,
                    label=display_name(m), markersize=4)
            ax.fill_between(range(len(dgp_names)), ci_lo[m], ci_hi[m], color=c, alpha=0.25)
        ax.set_xticks(range(len(dgp_names)))
        ax.set_xticklabels(dgp_names)
        ax.set_xlabel("DGP type")
        ax.set_title(f"d={d}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Metric invariance to DGP type under E1\n"
        f"d ∈ {{{', '.join(str(d) for d in d_vals)}}}, n={N_SAMPLES}",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 1 – Permutation / rescaling invariance across DGP types")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp01")
        dgp_names = config["dgp_names"]
        means = data["means"]
        ci_lo = data["ci_lo"]
        ci_hi = data["ci_hi"]
    else:
        dgp_names, means, ci_lo, ci_hi = run_invariance_across_dgps()
        save_results("exp01", {
            "means": means, "ci_lo": ci_lo, "ci_hi": ci_hi,
        }, config={
            "dgp_names": DGP_NAMES,
            "encoder": ENCODER,
            "n_samples": N_SAMPLES,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            fig = plot_invariance_across_dgps(dgp_names, means, ci_lo, ci_hi, metrics=mets)
            savefig(fig, f"exp01_invariance_across_dgps_{tag}.{ext}", subdir="exp01")

    # ── Multi-d sweeps ────────────────────────────────────────────────────
    all_d_results = {N_FACTORS: (dgp_names if not plot_only else config["dgp_names"],
                                  means if plot_only else means,
                                  ci_lo if plot_only else ci_lo,
                                  ci_hi if plot_only else ci_hi)}

    for d in D_VALUES:
        if d == N_FACTORS:
            continue
        exp_key = f"exp01_d{d}"

        if plot_only:
            try:
                d_data, d_config = load_results(exp_key)
                d_dgps = d_config["dgp_names"]
                all_d_results[d] = (d_dgps, d_data["means"], d_data["ci_lo"], d_data["ci_hi"])
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping d={d} plots")
                continue
        else:
            print(f"\n  ── d={d} ──")
            d_dgps, d_means, d_ci_lo, d_ci_hi = run_invariance_across_dgps(n_factors=d)
            save_results(exp_key, {
                "means": d_means, "ci_lo": d_ci_lo, "ci_hi": d_ci_hi,
            }, config={
                "dgp_names": DGP_NAMES,
                "encoder": ENCODER,
                "n_samples": N_SAMPLES,
                "n_factors": d,
                "n_seeds": N_SEEDS,
            })
            all_d_results[d] = (d_dgps, d_means, d_ci_lo, d_ci_hi)

        # Per-d plot
        tags_d = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags_d:
            for ext in ("pdf", "png"):
                d_dgps_plot, d_means_plot, d_ci_lo_plot, d_ci_hi_plot = all_d_results[d]
                fig = plot_invariance_across_dgps(d_dgps_plot, d_means_plot, d_ci_lo_plot, d_ci_hi_plot, metrics=mets)
                savefig(fig, f"exp01_invariance_across_dgps_d{d}_{tag}.{ext}", subdir="exp01")

    # Combined multi-d figure
    if len(all_d_results) > 1:
        tags_d = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags_d:
            for ext in ("pdf", "png"):
                fig = plot_invariance_multi_d(all_d_results, metrics=mets)
                savefig(fig, f"exp01_invariance_multi_d_{tag}.{ext}", subdir="exp01")

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Load saved results and regenerate plots")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: main plots only, skip sensitivity")
    args = parser.parse_args()
    main(plot_only=args.plot_only, quick=args.quick)
