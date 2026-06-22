"""
Experiment 11 – Are metrics inflated when they should not?

Tests with E9 (random Gaussian) and E10 (random Uniform), which ignore the
input entirely.  All identifiability metrics should converge to 0 (or their
theoretical minimum) for these null encoders.

In particular we check:
  • Whether MCC-RDC ever converges to 0 even with many samples.
  • Whether any metric remains inflated above a reasonable threshold.

Plot:
  • Line plot: n_samples on X axis, metric score on Y axis, for E9 and E10.
  • Bar chart: metric scores at large n for E9 and E10 side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    plot_metrics_vs_xaxis_with_ci,
    savefig,
    setup_plot_style,
    make_registry,
    get_color, get_marker, display_name,
    DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    MAIN_METRICS, APX_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_samples

# ── Configuration ──────────────────────────────────────────────────────────
DGPS = ["D1", "D2"]
NULL_ENCODERS = ["E9", "E10"]
# High-m arms: stress test with overcomplete null encoders
# E9 normally outputs m=d, but we override m to test high m/n
HIGH_M_VALUES = [50, 200]
SAMPLE_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

ENCODER_LABELS = {
    "E9": "E9 (Random Gaussian)",
    "E10": "E10 (Random Uniform)",
}


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_samples_null_encoder(dgp_name, encoder_name):
    """Sweep n_samples with a null encoder."""
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for n in SAMPLE_VALUES:
        print(f"      n = {n}")

        def eval_one_seed(seed, _n=n):
            return evaluate_dgp_encoder(
                dgp_name, encoder_name,
                n_samples=_n, n_factors=N_FACTORS, seed=seed,
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED)
        for m in METRICS:
            stats = agg.get(m, {})
            means[m].append(stats.get("mean", np.nan))
            ci_lo[m].append(stats.get("ci_lower", np.nan))
            ci_hi[m].append(stats.get("ci_upper", np.nan))

    return means, ci_lo, ci_hi


def sweep_samples_high_m(dgp_name, encoder_name, m_override):
    """Sweep n_samples with a null encoder at overridden output dim m."""
    registry = make_registry()
    means = {m_: [] for m_ in METRICS}
    ci_lo = {m_: [] for m_ in METRICS}
    ci_hi = {m_: [] for m_ in METRICS}

    for n in SAMPLE_VALUES:
        print(f"      n = {n}, m = {m_override}")

        def eval_one_seed(seed, _n=n):
            return evaluate_dgp_encoder(
                dgp_name, encoder_name,
                n_samples=_n, n_factors=N_FACTORS, seed=seed,
                encoder_kwargs={"m": m_override},
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED)
        for m_ in METRICS:
            stats = agg.get(m_, {})
            means[m_].append(stats.get("mean", np.nan))
            ci_lo[m_].append(stats.get("ci_lower", np.nan))
            ci_hi[m_].append(stats.get("ci_upper", np.nan))

    return means, ci_lo, ci_hi


def run_all_null_experiments():
    results = {}
    for dgp in DGPS:
        for enc in NULL_ENCODERS:
            key = f"{dgp}×{enc}"
            print(f"    {key}")
            means, ci_lo, ci_hi = sweep_samples_null_encoder(dgp, enc)
            results[key] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

    # High-m arms
    for m_val in HIGH_M_VALUES:
        for dgp in DGPS:
            key = f"{dgp}×E9(m={m_val})"
            print(f"    {key}")
            means, ci_lo, ci_hi = sweep_samples_high_m(dgp, "E9", m_val)
            results[key] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

    return results


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_null_convergence_grid(results, metrics=None):
    """Grid of subplots: one per D × null-encoder combo."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    keys = list(results.keys())
    n = len(keys)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, key in enumerate(keys):
        ax = axes_flat[idx]
        r = results[key]
        for m in metrics:
            c = get_color(m)
            ax.plot(SAMPLE_VALUES, r["means"][m], marker=get_marker(m),
                    color=c, label=display_name(m), markersize=4)
            ax.fill_between(SAMPLE_VALUES, r["ci_lo"][m], r["ci_hi"][m],
                            color=c, alpha=0.25)
        ax.axhline(0, color="black", ls=":", lw=0.8)
        ax.set_xlabel("n_samples")
        ax.set_title(key)
        ax.set_xscale("log")
        ax.set_xticks(SAMPLE_VALUES)
        ax.set_xticklabels([str(v) for v in SAMPLE_VALUES], fontsize=7)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    for idx in range(len(keys), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Metric score")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Null-encoder convergence – do metrics reach 0?", y=1.01)
    fig.tight_layout()
    return fig


def plot_bar_at_large_n(results, metrics=None):
    """Bar chart at the largest n comparing E9 and E10."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    large_idx = -1  # last entry in SAMPLE_VALUES
    large_n = SAMPLE_VALUES[large_idx]

    fig, axes = plt.subplots(1, len(DGPS), figsize=(7 * len(DGPS), 5), sharey=True)
    if len(DGPS) == 1:
        axes = [axes]

    width = 0.35
    x = np.arange(len(metrics))
    enc_colors = {"E9": "#1f77b4", "E10": "#ff7f0e"}

    for ax, dgp in zip(axes, DGPS):
        for i, enc in enumerate(NULL_ENCODERS):
            key = f"{dgp}×{enc}"
            vals = [results[key]["means"][m][large_idx] for m in metrics]
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=ENCODER_LABELS[enc],
                   color=enc_colors.get(enc, None), edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", ls=":", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in metrics],
                           rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{dgp} at n={large_n}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Metric inflation with null encoders (should be ≈ 0)", y=1.02)
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 11 – Metric inflation with null encoders (E9, E10)")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp11")
        results = data
        global SAMPLE_VALUES
        SAMPLE_VALUES = config["sample_values"]
    else:
        results = run_all_null_experiments()
        save_results("exp11", results, config={
            "sample_values": SAMPLE_VALUES,
            "dgps": DGPS,
            "null_encoders": NULL_ENCODERS,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            savefig(
                plot_null_convergence_grid(results, metrics=mets),
                f"exp11_null_convergence_grid_{tag}.{ext}", subdir="exp11",
            )
            savefig(
                plot_bar_at_large_n(results, metrics=mets),
                f"exp11_inflation_bar_{tag}.{ext}", subdir="exp11",
            )

    if not plot_only and not quick:
        # Sensitivity sweep: sample size for D1 × E10
        print("\n  Running sensitivity sweep: samples for D1 × E10 …")
        from pathlib import Path
        sweep_samples(
            dgp="D1",
            encoder="E10",
            sample_values=[50, 100, 200, 500, 1000, 2000],
            n_factors=N_FACTORS,
            n_seeds=N_SEEDS,
            base_seed=BASE_SEED,
            output_dir=Path(RESULTS_DIR / "exp11"),
            metrics_to_compute=APX_METRICS,
        )

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
