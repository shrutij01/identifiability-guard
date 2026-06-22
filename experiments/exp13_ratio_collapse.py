"""
Experiment 13 -- Ratio collapse test: does d/n govern metric behaviour?

Uses D1 (independent factors) + E1 (elementwise linear, m = d) -- the
cleanest possible setting with no confounds from correlation or nonlinearity.

We run the same metric at multiple (d, n) pairs that share the same d/n
ratio and check whether the resulting scores overlap.

Design
------
d in {3, 5, 10}
d/n in {0.02, 0.05, 0.10, 0.20}

  d/n    d=3    d=5    d=10
  0.02   150    250    500
  0.05    60    100    200
  0.10    30     50    100
  0.20    15     25     50

10 seeds per cell (finer overlap detection).

Expected results
----------------
Metrics split into three groups (from DL theory):

1. *Ratio-governed (m/n)*: R2, DCI, InfoE -- internal regression with m
   features on n samples.  Since m = d here, d/n = m/n and these should
   show approximate curve collapse.

2. *Independently-governed (m and n separately)*: MCC, MIG, InfoM, InfoC --
   pairwise estimation depends on n; matching/aggregation depends on m or
   m/d.  The ratio m/n is NOT the natural quantity; these should NOT
   collapse.  (MCC may be flat and near 1.0 for D1+E1 since it is trivially
   identifiable, making collapse uninformative for that metric.)

3. *d/n-governed*: T-MEX -- PCM test conditions on d-1 factors, so d/n is
   the relevant ratio.  Should collapse.

The *partial* collapse is itself the result: metrics fall into structurally
different ratio-dependence classes, and no single ratio universally governs
metric behaviour.

Output
------
  * One panel per metric with three curves (d=3, 5, 10) plotted against d/n.
  * Visual overlap = evidence that d/n governs; separation = evidence that
    d or n matters independently.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    plot_collapse_grid as plot_collapse_grid_util,
    savefig,
    setup_plot_style,
    make_registry,
    get_color, get_marker, display_name,
    ALL_METRICS,
    D_COLORS as UTIL_D_COLORS,
    D_MARKERS as UTIL_D_MARKERS,
    RESULTS_DIR,
)

# -- Configuration -----------------------------------------------------------
DGP = "D1"
ENCODER = "E1"                       # m = d
D_VALUES = [3, 5, 10, 20]
RATIO_VALUES = [0.02, 0.05, 0.10, 0.20]
N_SEEDS = 10                          # more seeds for fine overlap detection
BASE_SEED = 42
N_JOBS = -1                           # parallel seeds (-1 = all cores)
METRICS = sorted(ALL_METRICS.keys())

# Derived: n = d / ratio (rounded to int)
GRID = {
    d: [int(round(d / r)) for r in RATIO_VALUES]
    for d in D_VALUES
}
# e.g. GRID = {3: [150,60,30,15], 5: [250,100,50,25], 10: [500,200,100,50]}

# Curve colours -- one per d value
D_COLORS = {3: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c", 20: "#d62728"}
D_MARKERS = {3: "o", 5: "s", 10: "D", 20: "^"}


# -- Sweep -------------------------------------------------------------------

def sweep_collapse():
    """
    For each d, sweep n values (chosen so d/n hits RATIO_VALUES).

    Returns
    -------
    results : dict[int, dict]
        Keyed by d.  Each value is a dict with keys 'means', 'ci_lo',
        'ci_hi', each mapping metric_name -> list of floats (one per ratio).
    """
    results = {}
    for d in D_VALUES:
        print(f"\n  d = {d}")
        registry = make_registry()
        means = {met: [] for met in METRICS}
        ci_lo = {met: [] for met in METRICS}
        ci_hi = {met: [] for met in METRICS}

        for ratio, n in zip(RATIO_VALUES, GRID[d]):
            print(f"    d/n = {ratio:.2f}  (d={d}, n={n})")

            def eval_one_seed(seed, _d=d, _n=n):
                return evaluate_dgp_encoder(
                    DGP, ENCODER,
                    n_samples=_n, n_factors=_d, seed=seed,
                    registry=registry,
                )

            _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                             base_seed=BASE_SEED,
                                             n_jobs=N_JOBS)
            for met in METRICS:
                stats = agg.get(met, {})
                means[met].append(stats.get("mean", np.nan))
                ci_lo[met].append(stats.get("ci_lower", np.nan))
                ci_hi[met].append(stats.get("ci_upper", np.nan))

        results[d] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

    return results


# -- Plotting ----------------------------------------------------------------

def plot_collapse_grid(results):
    """One panel per metric.  Three curves per panel (one per d)."""
    setup_plot_style()
    n_metrics = len(METRICS)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, met in enumerate(METRICS):
        ax = axes_flat[idx]
        for d in D_VALUES:
            r = results[d]
            c = D_COLORS[d]
            mk = D_MARKERS[d]
            ax.plot(RATIO_VALUES, r["means"][met], marker=mk, color=c,
                    label=f"d={d}", markersize=6)
            ax.fill_between(RATIO_VALUES, r["ci_lo"][met], r["ci_hi"][met],
                            color=c, alpha=0.12)

        ax.set_xlabel(r"$d\,/\,n$")
        ax.set_title(display_name(met), fontsize=11)
        ax.set_xscale("log")
        ax.set_xticks(RATIO_VALUES)
        ax.set_xticklabels([f"{r:.2f}" for r in RATIO_VALUES], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Metric score")
    fig.suptitle(
        r"Ratio collapse test: metric score vs $d/n$ "
        f"({DGP} + {ENCODER}, m = d)\n"
        r"Overlap $\Rightarrow$ $d/n$ governs; "
        r"separation $\Rightarrow$ $d$ or $n$ matters independently",
        y=1.03, fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_collapse_grid_main(results, focus_metrics=None):
    """Main-text 1×4 panels using the shared plot_collapse_grid utility.

    Shows R², DCI-D, MCC-P, T-MEX with d-curves overlaid for visual collapse test.
    """
    if focus_metrics is None:
        focus_metrics = ["r2", "dci_disentanglement", "mcc_pearson", "tmex"]
    focus_metrics = [m for m in focus_metrics if m in METRICS]

    # Build curves_per_metric dict
    curves_per_metric = {}
    for met in focus_metrics:
        curves = []
        for d in D_VALUES:
            r = results[d]
            c = D_COLORS[d]
            mk = D_MARKERS[d]
            curves.append((
                RATIO_VALUES, r["means"][met], r["ci_lo"][met], r["ci_hi"][met],
                f"d={d}", c, mk,
            ))
        curves_per_metric[met] = curves

    fig = plot_collapse_grid_util(
        focus_metrics, curves_per_metric,
        xlabel=r"$d\,/\,n$",
        suptitle=(
            r"Ratio collapse: metric score vs $d/n$ "
            f"({DGP} + {ENCODER}, m = d)\n"
            r"Overlap $\Rightarrow$ $d/n$ governs"
        ),
        ncols=len(focus_metrics),
        xscale="log",
    )
    return fig


def plot_collapse_by_group(results):
    """
    Three-panel summary grouping metrics by expected behaviour.

    Panel 1: ratio-governed (R2, DCI, InfoE) -- expect collapse.
    Panel 2: independently-governed (MCC-P, MCC-S, MIG, InfoM, InfoC).
    Panel 3: d/n-governed (T-MEX) -- expect collapse.
    """
    setup_plot_style()
    groups = {
        r"Ratio-governed ($m/n$): expect collapse": [
            "r2", "dci_disentanglement", "infom",
        ],
        r"Independent ($m$ and $n$ separately)": [
            "mcc_pearson", "mcc_spearman", "mig", "infom",
        ],
        r"$d/n$-governed: expect collapse": [
            "tmex",
        ],
    }
    # Filter to metrics actually present
    groups = {
        k: [m for m in v if m in METRICS]
        for k, v in groups.items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (group_title, group_metrics) in zip(axes, groups.items()):
        for met in group_metrics:
            for d in D_VALUES:
                r = results[d]
                c = D_COLORS[d]
                mk = D_MARKERS[d]
                ax.plot(RATIO_VALUES, r["means"][met], marker=mk, color=c,
                        linestyle="-" if met == group_metrics[0] else "--",
                        markersize=5, alpha=0.8,
                        label=f"d={d}, {display_name(met)}")
                ax.fill_between(RATIO_VALUES, r["ci_lo"][met],
                                r["ci_hi"][met], color=c, alpha=0.06)

        ax.set_xlabel(r"$d\,/\,n$")
        ax.set_title(group_title, fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(RATIO_VALUES)
        ax.set_xticklabels([f"{r:.2f}" for r in RATIO_VALUES], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, ncol=1, loc="lower left")

    axes[0].set_ylabel("Metric score")
    fig.suptitle(
        "Metrics split into three ratio-dependence classes",
        y=1.02, fontsize=14,
    )
    fig.tight_layout()
    return fig


# -- Main --------------------------------------------------------------------

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 13 -- Ratio collapse test (d/n)")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp13")
        # Keys are strings after load; reconstruct with int keys for D_VALUES iteration
        results = {}
        for d in config["d_values"]:
            results[d] = data[str(d)]
    else:
        print(f"DGP: {DGP}, Encoder: {ENCODER} (m = d)")
        print(f"d values: {D_VALUES}")
        print(f"d/n ratios: {RATIO_VALUES}")
        print(f"Grid (d -> n values): {GRID}")
        print(f"Seeds per cell: {N_SEEDS}")

        results = sweep_collapse()
        # Convert int keys to str for save
        save_data = {str(d): v for d, v in results.items()}
        save_results("exp13", save_data, config={
            "d_values": D_VALUES,
            "ratio_values": RATIO_VALUES,
            "dgp": DGP,
            "encoder": ENCODER,
            "n_seeds": N_SEEDS,
        })

    for ext in ("pdf", "png"):
        fig1 = plot_collapse_grid(results)
        savefig(fig1, f"exp13_ratio_collapse_grid.{ext}", subdir="exp13")

        fig2 = plot_collapse_by_group(results)
        savefig(fig2, f"exp13_ratio_collapse_groups.{ext}", subdir="exp13")

        # Main-text 1×4 panel (focus metrics)
        fig3 = plot_collapse_grid_main(results)
        savefig(fig3, f"exp13_ratio_collapse_main.{ext}", subdir="exp13")

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
