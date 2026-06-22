"""
Experiment 12 -- Re-plot Experiment 6 with structural ratio m/d on the x-axis.

Experiment 6a sweeps m (retained factors) with fixed d = 10.
Experiment 6b sweeps d (total factors) with fixed m = 3.
Both produce curves that can be re-plotted against the ratio m/d.  If the
curves collapse onto each other in the overlapping range, this demonstrates
that the structural ratio m/d -- not m or d individually -- governs metric
behaviour under information loss.

This requires no new experimental design; it re-runs the same sweeps as
Experiment 6 and changes only the x-axis.

Output
------
  * Per-DGP single-panel plots (exp06a data) with m/d on x-axis.
  * Overlay figures for D1 and D2: exp06a and exp06b curves on the same axes,
    both plotted against m/d.  Visual overlap = evidence of collapse.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    plot_metrics_vs_xaxis_with_ci,
    plot_collapse,
    savefig,
    setup_plot_style,
    make_registry,
    get_color, get_marker, display_name,
    DEFAULT_N_SAMPLES, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    ALL_METRICS,
    RESULTS_DIR,
)

# -- Configuration -----------------------------------------------------------
DGPS_6A = ["D1", "D2", "D3", "D4"]
DGPS_6B = ["D1", "D2"]           # exp06b only ran these two
D_TOTAL = 10                      # total factors in exp06a
M_FIXED = 3                       # fixed m in exp06b
N_SAMPLES = DEFAULT_N_SAMPLES
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
M_MAX_RATIO = 3.0                 # sweep m/d up to this value
M_MIN_RATIO = 0.3                 # sweep m/d down to this value
N_JOBS = -1                       # parallel seeds (-1 = all cores)
METRICS = sorted(ALL_METRICS.keys())


# -- Sweep functions (mirroring exp06) ---------------------------------------

def sweep_m_fixed_d(dgp_name: str, d_total: int = D_TOTAL):
    """Exp06a: sweep m from M_MAX_RATIO*d down to M_MIN_RATIO*d, d fixed."""
    registry = make_registry()
    m_lo = max(1, int(round(M_MIN_RATIO * d_total)))
    m_values = list(range(int(M_MAX_RATIO * d_total), m_lo - 1, -1))
    means = {met: [] for met in METRICS}
    ci_lo = {met: [] for met in METRICS}
    ci_hi = {met: [] for met in METRICS}

    for m in m_values:
        if m == d_total:
            encoder, enc_kw = "E1", {}
        elif m < d_total:
            encoder, enc_kw = "E4", {"m": m}
        else:
            encoder, enc_kw = "E5", {"m": m}
        print(f"    {dgp_name}  m={m}, d={d_total} (encoder={encoder})")

        def eval_one_seed(seed, _m=m, _enc=encoder, _enc_kw=enc_kw):
            return evaluate_dgp_encoder(
                dgp_name, _enc,
                n_samples=N_SAMPLES, n_factors=d_total, seed=seed,
                encoder_kwargs=_enc_kw, registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED, n_jobs=N_JOBS)
        for met in METRICS:
            stats = agg.get(met, {})
            means[met].append(stats.get("mean", np.nan))
            ci_lo[met].append(stats.get("ci_lower", np.nan))
            ci_hi[met].append(stats.get("ci_upper", np.nan))

    ratio_values = [m / d_total for m in m_values]
    return ratio_values, means, ci_lo, ci_hi


def sweep_d_fixed_m(dgp_name: str, m_fixed: int = M_FIXED):
    """Exp06b: sweep d from m/M_MAX_RATIO up to m/M_MIN_RATIO, m fixed."""
    registry = make_registry()
    d_lo = max(1, int(round(m_fixed / M_MAX_RATIO)))
    d_hi = int(round(m_fixed / M_MIN_RATIO))
    d_values = list(range(d_lo, d_hi + 1))
    means = {met: [] for met in METRICS}
    ci_lo = {met: [] for met in METRICS}
    ci_hi = {met: [] for met in METRICS}

    for d in d_values:
        if d == m_fixed:
            encoder, enc_kw = "E1", {}
        elif d > m_fixed:
            encoder, enc_kw = "E4", {"m": m_fixed}
        else:
            encoder, enc_kw = "E5", {"m": m_fixed}
        print(f"    {dgp_name}  d={d}, m={m_fixed} (encoder={encoder})")

        def eval_one_seed(seed, _d=d, _enc=encoder, _enc_kw=enc_kw):
            return evaluate_dgp_encoder(
                dgp_name, _enc,
                n_samples=N_SAMPLES, n_factors=_d, seed=seed,
                encoder_kwargs=_enc_kw, registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED, n_jobs=N_JOBS)
        for met in METRICS:
            stats = agg.get(met, {})
            means[met].append(stats.get("mean", np.nan))
            ci_lo[met].append(stats.get("ci_lower", np.nan))
            ci_hi[met].append(stats.get("ci_upper", np.nan))

    ratio_values = [m_fixed / d for d in d_values]
    return ratio_values, means, ci_lo, ci_hi


# -- Plotting ----------------------------------------------------------------

def plot_single_dgp_ratio(dgp_name, ratio_values, means, ci_lo, ci_hi,
                          source_label="6a"):
    """Single-panel: metric score vs m/d for one DGP."""
    fig = plot_metrics_vs_xaxis_with_ci(
        ratio_values, means, ci_lo, ci_hi,
        xlabel=r"$m\,/\,d$",
        title=f"Metric score vs structural ratio m/d ({dgp_name}, {source_label})",
    )
    return fig


def plot_overlay(dgp_name, data_6a, data_6b):
    """
    Overlay exp06a and exp06b curves on the same axes, plotted against m/d.

    Parameters
    ----------
    data_6a, data_6b : tuple
        (ratio_values, means, ci_lo, ci_hi) from the respective sweeps.
    """
    setup_plot_style()
    ratio_a, means_a, ci_lo_a, ci_hi_a = data_6a
    ratio_b, means_b, ci_lo_b, ci_hi_b = data_6b

    n_metrics = len(METRICS)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, met in enumerate(METRICS):
        ax = axes_flat[idx]
        c = get_color(met)

        # 6a: vary m, d fixed  (solid line)
        ax.plot(ratio_a, means_a[met], marker=get_marker(met), color=c,
                label=f"vary m (d={D_TOTAL})", linestyle="-", markersize=5)
        ax.fill_between(ratio_a, ci_lo_a[met], ci_hi_a[met],
                        color=c, alpha=0.25)

        # 6b: vary d, m fixed  (dashed line)
        ax.plot(ratio_b, means_b[met], marker=get_marker(met), color=c,
                label=f"vary d (m={M_FIXED})", linestyle="--", markersize=5)
        ax.fill_between(ratio_b, ci_lo_b[met], ci_hi_b[met],
                        color=c, alpha=0.25)

        ax.set_xlabel(r"$m\,/\,d$")
        ax.set_title(display_name(met), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(M_MIN_RATIO - 0.05, M_MAX_RATIO + 0.1)
        ax.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Metric score")
    fig.suptitle(
        f"Structural ratio collapse test: exp06a vs exp06b ({dgp_name})\n"
        r"If curves overlap $\Rightarrow$ m/d is the governing quantity",
        y=1.02, fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_overlay_main(dgp_name, data_6a, data_6b, focus_metrics=None):
    """Main-text overlay: 1×4 panels for focus metrics using plot_collapse style.

    Each panel shows two curves (sweep-m solid, sweep-d dashed) for one metric.
    """
    if focus_metrics is None:
        focus_metrics = ["r2", "dci_disentanglement", "mcc_pearson", "tmex"]
    focus_metrics = [m for m in focus_metrics if m in METRICS]
    setup_plot_style()

    ratio_a, means_a, ci_lo_a, ci_hi_a = data_6a
    ratio_b, means_b, ci_lo_b, ci_hi_b = data_6b

    n = len(focus_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, met in enumerate(focus_metrics):
        ax = axes[idx]
        c = get_color(met)

        # Curve A: vary m, d fixed (solid)
        ax.plot(ratio_a, means_a[met], marker=get_marker(met), color=c,
                label=f"vary m (d={D_TOTAL})", linestyle="-", markersize=5)
        ax.fill_between(ratio_a, ci_lo_a[met], ci_hi_a[met],
                        color=c, alpha=0.20)

        # Curve B: vary d, m fixed (dashed)
        ax.plot(ratio_b, means_b[met], marker=get_marker(met), color=c,
                label=f"vary d (m={M_FIXED})", linestyle="--", markersize=5)
        ax.fill_between(ratio_b, ci_lo_b[met], ci_hi_b[met],
                        color=c, alpha=0.12)

        ax.set_xlabel(r"$m\,/\,d$")
        ax.set_title(display_name(met), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(M_MIN_RATIO - 0.05, M_MAX_RATIO + 0.1)
        ax.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("Metric score")
    fig.suptitle(
        f"Structural ratio collapse: sweep-m vs sweep-d ({dgp_name})\n"
        r"Overlap $\Rightarrow$ m/d governs metric behaviour",
        y=1.02, fontsize=13,
    )
    fig.tight_layout()
    return fig


# -- Main --------------------------------------------------------------------

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 12 -- Re-plot exp06 with m/d on x-axis")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp12")
        d_total = config["d_total"]
        m_fixed = config["m_fixed"]

        # Reconstruct 6a
        results_6a = {}
        for dgp in config["dgps_6a"]:
            d = data["6a"][dgp]
            m_lo = max(1, int(round(M_MIN_RATIO * d_total)))
            m_values = list(range(int(M_MAX_RATIO * d_total), m_lo - 1, -1))
            ratio_values = [m / d_total for m in m_values]
            results_6a[dgp] = (ratio_values, d["means"], d["ci_lo"], d["ci_hi"])

        # Reconstruct 6b
        results_6b = {}
        for dgp in config["dgps_6b"]:
            d = data["6b"][dgp]
            d_lo = max(1, int(round(m_fixed / M_MAX_RATIO)))
            d_hi = int(round(m_fixed / M_MIN_RATIO))
            d_values = list(range(d_lo, d_hi + 1))
            ratio_values = [m_fixed / dv for dv in d_values]
            results_6b[dgp] = (ratio_values, d["means"], d["ci_lo"], d["ci_hi"])
    else:
        # --- exp06a re-plot: all DGPs with m/d x-axis ---
        results_6a = {}
        save_6a = {}
        for dgp in DGPS_6A:
            print(f"\n  [6a] DGP: {dgp}")
            ratio_vals, means, ci_lo, ci_hi = sweep_m_fixed_d(dgp)
            results_6a[dgp] = (ratio_vals, means, ci_lo, ci_hi)
            save_6a[dgp] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

        # --- exp06b: D1 and D2 with m/d x-axis ---
        results_6b = {}
        save_6b = {}
        for dgp in DGPS_6B:
            print(f"\n  [6b] DGP: {dgp}")
            ratio_vals, means, ci_lo, ci_hi = sweep_d_fixed_m(dgp)
            results_6b[dgp] = (ratio_vals, means, ci_lo, ci_hi)
            save_6b[dgp] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

        save_results("exp12", {"6a": save_6a, "6b": save_6b}, config={
            "dgps_6a": DGPS_6A,
            "dgps_6b": DGPS_6B,
            "d_total": D_TOTAL,
            "m_fixed": M_FIXED,
            "n_samples": N_SAMPLES,
            "n_seeds": N_SEEDS,
        })

    # --- Plot 6a single-DGP ratio plots ---
    for dgp, (ratio_vals, means, ci_lo, ci_hi) in results_6a.items():
        for ext in ("pdf", "png"):
            fig = plot_single_dgp_ratio(dgp, ratio_vals, means, ci_lo, ci_hi,
                                        source_label="vary m, d=10")
            savefig(fig, f"exp12_ratio_6a_{dgp}.{ext}", subdir="exp12")

    # --- Plot 6b single-DGP ratio plots ---
    for dgp, (ratio_vals, means, ci_lo, ci_hi) in results_6b.items():
        for ext in ("pdf", "png"):
            fig = plot_single_dgp_ratio(dgp, ratio_vals, means, ci_lo, ci_hi,
                                        source_label="vary d, m=3")
            savefig(fig, f"exp12_ratio_6b_{dgp}.{ext}", subdir="exp12")

    # --- Overlay: 6a vs 6b ---
    dgps_6b = config["dgps_6b"] if plot_only else DGPS_6B
    for dgp in dgps_6b:
        for ext in ("pdf", "png"):
            fig = plot_overlay(dgp, results_6a[dgp], results_6b[dgp])
            savefig(fig, f"exp12_overlay_{dgp}.{ext}", subdir="exp12")

    # --- Main-text overlay: 1×4 panels for focus metrics ---
    for dgp in dgps_6b:
        for ext in ("pdf", "png"):
            fig = plot_overlay_main(dgp, results_6a[dgp], results_6b[dgp])
            savefig(fig, f"exp12_overlay_main_{dgp}.{ext}", subdir="exp12")

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Load saved results and regenerate plots")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: main plots only, skip sensitivity")
    args = parser.parse_args()
    main(plot_only=args.plot_only, quick=args.quick)
