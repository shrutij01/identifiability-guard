"""
Experiment 2 – How sensitive are metrics to the difference between elementwise
rescaling v/s invertible non-linear transformations?

Plot: vary the strength of elementwise non-linearity (and type of non-linearity)
on the X axis, metric values on the Y axis.  The spectrum goes from linear
(nonlinearity_strength=0, equivalent to E1) to fully non-linear
(nonlinearity_strength=1, full E2).

For each non-linearity type we show a separate panel/subplot.
"""

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
    MAIN_METRICS,
    APX_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_encoder_nonlinearity
from identifiability_guard.encoders.e2_elementwise_nonlinear import E2ElementwiseNonlinear
from identifiability_guard.encoders.base import BaseEncoder

# ── Configuration ──────────────────────────────────────────────────────────
DGP = "D1"  # independent – isolates encoder effect
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Multi-d: confirm α sensitivity scales with d
D_VALUES = [5, 10]

# Strengths from linear to fully non-linear
NONLINEARITY_STRENGTHS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

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


def run_all_nonlinearity_types(n_factors=N_FACTORS):
    """Run sweep for each non-linearity type.  Returns dict of results."""
    results = {}
    for fn_name, fn in NONLINEARITY_TYPES.items():
        print(f"  Non-linearity type: {fn_name}")
        strengths, means, ci_lo, ci_hi = sweep_nonlinearity_strength(fn, fn_name, n_factors=n_factors)
        results[fn_name] = (strengths, means, ci_lo, ci_hi)
    return results


def plot_nonlinearity_sweep(results, metrics=None):
    """One subplot per non-linearity type, all metrics overlaid."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    n_types = len(results)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5), sharey=True)
    if n_types == 1:
        axes = [axes]

    for ax, (fn_name, (strengths, means, ci_lo, ci_hi)) in zip(axes, results.items()):
        for m in metrics:
            c = get_color(m)
            ax.plot(strengths, means[m], marker=get_marker(m), color=c,
                    label=display_name(m), markersize=4)
            ax.fill_between(strengths, ci_lo[m], ci_hi[m], color=c, alpha=0.25)
        ax.set_xlabel(r"Non-linearity ($\alpha$)")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, strengths[-1] * 1.1)
        ax.set_title(fn_name)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score")
    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Metric sensitivity to elementwise non-linearity\n"
        f"({DGP} + E2, n={N_SAMPLES}, d={N_FACTORS})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_nonlinearity_sweep_single_panel(results, metrics=None, fn_name="tanh"):
    """Main-text variant: single panel showing one nonlinearity type only."""
    if metrics is None:
        metrics = METRICS_MAIN
    if fn_name not in results:
        fn_name = list(results.keys())[0]
    strengths, means, ci_lo, ci_hi = results[fn_name]
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in metrics:
        c = get_color(m)
        ax.plot(strengths, means[m], marker=get_marker(m), color=c,
                label=display_name(m), markersize=5)
        ax.fill_between(strengths, ci_lo[m], ci_hi[m], color=c, alpha=0.25)
    ax.set_xlabel(r"Non-linearity ($\alpha$)")
    ax.set_ylabel("Metric score")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(0, strengths[-1] * 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Metric sensitivity to elementwise non-linearity ({fn_name})\n"
                 f"({DGP} + E2, d={N_FACTORS})")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# Also produce a single-panel plot averaging over all nonlinearity types
def plot_nonlinearity_sweep_combined(results, metrics=None):
    """Average across nonlinearity types → single panel."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    strengths = NONLINEARITY_STRENGTHS
    combined_means = {m: np.zeros(len(strengths)) for m in metrics}
    combined_ci_lo = {m: np.zeros(len(strengths)) for m in metrics}
    combined_ci_hi = {m: np.zeros(len(strengths)) for m in metrics}
    n = len(results)
    for fn_name, (_, means, ci_lo, ci_hi) in results.items():
        for m in metrics:
            combined_means[m] += np.array(means[m])
            combined_ci_lo[m] += np.array(ci_lo[m])
            combined_ci_hi[m] += np.array(ci_hi[m])
    for m in metrics:
        combined_means[m] /= n
        combined_ci_lo[m] /= n
        combined_ci_hi[m] /= n

    fig = plot_metrics_vs_xaxis_with_ci(
        strengths, combined_means, combined_ci_lo, combined_ci_hi,
        xlabel=r"Non-linearity ($\alpha$)",
        title="Metric sensitivity to non-linearity (averaged over function types)\n"
              f"({DGP} + E2)",
    )
    ax = fig.axes[0]
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(0, strengths[-1] * 1.1)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 2 – Elementwise rescaling vs non-linear transformations")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp02")
        strengths = config["strengths"]
        fn_names = config["nonlinearity_types"]
        results = {}
        for fn_name in fn_names:
            d = data[fn_name]
            results[fn_name] = (strengths, d["means"], d["ci_lo"], d["ci_hi"])
    else:
        results = run_all_nonlinearity_types()
        save_data = {}
        for fn_name, (strengths, means, ci_lo, ci_hi) in results.items():
            save_data[fn_name] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}
        save_results("exp02", save_data, config={
            "strengths": NONLINEARITY_STRENGTHS,
            "nonlinearity_types": list(NONLINEARITY_TYPES.keys()),
            "dgp": DGP,
            "n_samples": N_SAMPLES,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            fig = plot_nonlinearity_sweep(results, metrics=mets)
            savefig(fig, f"exp02_nonlinearity_per_type_{tag}.{ext}", subdir="exp02")
            fig = plot_nonlinearity_sweep_combined(results, metrics=mets)
            savefig(fig, f"exp02_nonlinearity_combined_{tag}.{ext}", subdir="exp02")

    # Main-text single-panel figure (tanh only, 4 core metrics)
    for ext in ("pdf", "png"):
        fig = plot_nonlinearity_sweep_single_panel(results, metrics=METRICS_MAIN)
        savefig(fig, f"exp02_nonlinearity_single_panel_main.{ext}", subdir="exp02")

    if not plot_only and not quick:
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
            metrics_to_compute=APX_METRICS,
            nonlinearity_type="tanh_modified",
        )

    # ── Multi-d sweep (d=10) ──────────────────────────────────────────────
    for d in D_VALUES:
        if d == N_FACTORS:
            continue
        exp_key = f"exp02_d{d}"

        if plot_only:
            try:
                d_data, d_config = load_results(exp_key)
                d_strengths = d_config["strengths"]
                d_fn_names = d_config["nonlinearity_types"]
                d_results = {}
                for fn_name in d_fn_names:
                    dd = d_data[fn_name]
                    d_results[fn_name] = (d_strengths, dd["means"], dd["ci_lo"], dd["ci_hi"])
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping d={d}")
                continue
        else:
            print(f"\n  ── d={d} ──")
            d_results = run_all_nonlinearity_types(n_factors=d)
            d_save = {}
            for fn_name, (strengths, means, ci_lo, ci_hi) in d_results.items():
                d_save[fn_name] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}
            save_results(exp_key, d_save, config={
                "strengths": NONLINEARITY_STRENGTHS,
                "nonlinearity_types": list(NONLINEARITY_TYPES.keys()),
                "dgp": DGP,
                "n_samples": N_SAMPLES,
                "n_factors": d,
                "n_seeds": N_SEEDS,
            })

        tags_d = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags_d:
            for ext in ("pdf", "png"):
                fig = plot_nonlinearity_sweep(d_results, metrics=mets)
                savefig(fig, f"exp02_nonlinearity_per_type_d{d}_{tag}.{ext}", subdir="exp02")
                fig = plot_nonlinearity_sweep_combined(d_results, metrics=mets)
                savefig(fig, f"exp02_nonlinearity_combined_d{d}_{tag}.{ext}", subdir="exp02")

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
