"""
Experiment 14 -- Disentangle m/n from d/n.

Experiment 13 uses m = d (E1), so d/n = m/n and the two ratios cannot be
separated.  This experiment breaks that symmetry by using E4 (undercomplete
linear, m < d) with fixed d = 10.

Design (revised)
------
Since m/n = (m/d) × (d/n), only two of the three ratios are independent.
We need exactly two sweeps:

**Sweep A -- vary m/d, hold d/n constant at 0.01.**
Fix d=10, n=1000.  Vary m ∈ {1, 3, 5, 7, 9}.
d/n is perfectly constant.  m/n = m/d × 0.01 stays in safe zone.

**Sweep B -- vary d/n (and m/n), hold m/d constant at 0.50.**
Fix m=5, d=10.  Vary n ∈ {2000, 1000, 500, 200, 100, 50}.
m/d is perfectly constant.  d/n co-varies with m/n (unavoidable when
d and m are fixed).

Expected results
----------------
  * Sweep A: If a metric is flat, m/n governs it (m/d doesn't matter
    independently).  If it varies, m/d has structural effect.
  * Sweep B: convergence study -- metrics should improve as n grows
    (d/n shrinks).  The rate reveals which metrics are most sample-hungry.

Output
------
  * Sweep A figure: one panel per metric, x = m/d.
  * Sweep B figure: one panel per metric, x = d/n (log scale).
  * Combined summary figure.
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
    RESULTS_DIR,
)

# -- Configuration -----------------------------------------------------------
DGP = "D1"
ENCODER = "E4"                        # undercomplete linear, m < d
D_FIXED = 10
N_SEEDS = 10
BASE_SEED = 42
N_JOBS = -1                           # parallel seeds (-1 = all cores)
METRICS = sorted(ALL_METRICS.keys())

# Sweep A: vary m/d, d/n = 0.01 constant
N_FIXED_A = 1000                      # n for sweep A (d/n = 10/1000 = 0.01)
M_VALUES_A = [1, 3, 5, 7, 9]
SWEEP_A = [(m, N_FIXED_A) for m in M_VALUES_A]

# Sweep B: vary d/n, m/d = 0.50 constant
M_FIXED_B = 5                         # m for sweep B (m/d = 5/10 = 0.50)
N_VALUES_B = [2000, 1000, 500, 200, 100, 50]
SWEEP_B = [(M_FIXED_B, n) for n in N_VALUES_B]


# -- Sweep functions ---------------------------------------------------------

def _run_sweep(pairs, label=""):
    """
    Run evaluations for a list of (m, n) pairs with d = D_FIXED.

    Returns
    -------
    means, ci_lo, ci_hi : dict[str, list[float]]
        One entry per metric, lists aligned with *pairs*.
    """
    registry = make_registry()
    means = {met: [] for met in METRICS}
    ci_lo = {met: [] for met in METRICS}
    ci_hi = {met: [] for met in METRICS}

    for m, n in pairs:
        print(f"    {label}  m={m}, n={n}, d={D_FIXED}  "
              f"(m/d={m/D_FIXED:.2f}, m/n={m/n:.3f}, d/n={D_FIXED/n:.3f})")

        def eval_one_seed(seed, _m=m, _n=n):
            return evaluate_dgp_encoder(
                DGP, ENCODER,
                n_samples=_n, n_factors=D_FIXED, seed=seed,
                encoder_kwargs={"m": _m},
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED, n_jobs=N_JOBS)
        for met in METRICS:
            stats = agg.get(met, {})
            means[met].append(stats.get("mean", np.nan))
            ci_lo[met].append(stats.get("ci_lower", np.nan))
            ci_hi[met].append(stats.get("ci_upper", np.nan))

    return means, ci_lo, ci_hi


def run_sweep_a():
    """Sweep A: vary m/d at constant d/n = 0.01."""
    print(f"\n  Sweep A: d/n = {D_FIXED/N_FIXED_A:.3f}, n = {N_FIXED_A}")
    means, ci_lo, ci_hi = _run_sweep(SWEEP_A, label="[A]")
    return {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}


def run_sweep_b():
    """Sweep B: vary d/n at constant m/d = 0.50."""
    print(f"\n  Sweep B: m/d = {M_FIXED_B/D_FIXED:.2f}, m = {M_FIXED_B}")
    means, ci_lo, ci_hi = _run_sweep(SWEEP_B, label="[B]")
    return {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}


# -- Plotting ----------------------------------------------------------------

def plot_sweep_a(results_a):
    """
    Sweep A: one panel per metric, x = m/d.
    """
    setup_plot_style()
    md_values = [m / D_FIXED for m in M_VALUES_A]

    n_metrics = len(METRICS)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, met in enumerate(METRICS):
        ax = axes_flat[idx]
        c = get_color(met)
        ax.plot(md_values, results_a["means"][met], marker=get_marker(met),
                color=c, markersize=6)
        ax.fill_between(md_values, results_a["ci_lo"][met],
                        results_a["ci_hi"][met], color=c, alpha=0.15)

        ax.set_xlabel(r"$m\,/\,d$")
        ax.set_title(display_name(met), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Metric score")
    fig.suptitle(
        r"Sweep A: vary $m/d$, constant $d/n$"
        f" = {D_FIXED/N_FIXED_A:.3f} "
        f"({DGP} + {ENCODER}, d={D_FIXED}, n={N_FIXED_A})\n"
        r"Flat $\Rightarrow$ only $d/n$ matters; "
        r"varying $\Rightarrow$ $m/d$ has structural effect",
        y=1.03, fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_sweep_b(results_b):
    """
    Sweep B: one panel per metric, x = d/n (log scale).
    """
    setup_plot_style()
    dn_values = [D_FIXED / n for n in N_VALUES_B]

    n_metrics = len(METRICS)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, met in enumerate(METRICS):
        ax = axes_flat[idx]
        c = get_color(met)
        ax.plot(dn_values, results_b["means"][met], marker=get_marker(met),
                color=c, markersize=6)
        ax.fill_between(dn_values, results_b["ci_lo"][met],
                        results_b["ci_hi"][met], color=c, alpha=0.15)

        ax.set_xlabel(r"$d\,/\,n$")
        ax.set_xscale("log")
        ax.set_title(display_name(met), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Metric score")
    fig.suptitle(
        r"Sweep B: vary $d/n$, constant $m/d$"
        f" = {M_FIXED_B/D_FIXED:.2f} "
        f"({DGP} + {ENCODER}, d={D_FIXED}, m={M_FIXED_B})\n"
        r"Convergence study: metrics should improve as $n$ grows",
        y=1.03, fontsize=13,
    )
    fig.tight_layout()
    return fig


FOCUS_METRICS = ["r2", "dci_disentanglement", "mcc_pearson", "tmex"]


def plot_combined_summary(results_a, results_b, focus_metrics=None):
    """
    Two-row summary: top row = Sweep A, bottom row = Sweep B.
    Show only the four most informative metrics (R2, DCI, MCC-P, T-MEX).
    """
    setup_plot_style()
    if focus_metrics is None:
        focus_metrics = FOCUS_METRICS
    focus_metrics = [m for m in focus_metrics if m in METRICS]
    n_focus = len(focus_metrics)

    md_values = [m / D_FIXED for m in M_VALUES_A]
    dn_values = [D_FIXED / n for n in N_VALUES_B]

    fig, axes = plt.subplots(2, n_focus, figsize=(5 * n_focus, 8),
                             sharey=True)

    # Top row: Sweep A
    for j, met in enumerate(focus_metrics):
        ax = axes[0, j]
        c = get_color(met)
        ax.plot(md_values, results_a["means"][met], marker=get_marker(met),
                color=c, markersize=6)
        ax.fill_between(md_values, results_a["ci_lo"][met],
                        results_a["ci_hi"][met], color=c, alpha=0.15)
        ax.set_xlabel(r"$m\,/\,d$")
        ax.set_title(display_name(met), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if j == 0:
            ax.set_ylabel(f"Sweep A\n(const d/n={D_FIXED/N_FIXED_A:.3f})")

    # Bottom row: Sweep B
    for j, met in enumerate(focus_metrics):
        ax = axes[1, j]
        c = get_color(met)
        ax.plot(dn_values, results_b["means"][met], marker=get_marker(met),
                color=c, markersize=6)
        ax.fill_between(dn_values, results_b["ci_lo"][met],
                        results_b["ci_hi"][met], color=c, alpha=0.15)
        ax.set_xlabel(r"$d\,/\,n$")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if j == 0:
            ax.set_ylabel(f"Sweep B\n(const m/d={M_FIXED_B/D_FIXED:.2f})")

    fig.suptitle(
        f"Disentangling m/n from d/n ({DGP} + {ENCODER}, d={D_FIXED})",
        y=1.02, fontsize=14,
    )
    fig.tight_layout()
    return fig


# -- Main --------------------------------------------------------------------

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 14 -- Disentangle m/n from d/n")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp14")
        results_a = data["sweep_a"]
        results_b = data["sweep_b"]
    else:
        print(f"DGP: {DGP}, Encoder: {ENCODER}, d = {D_FIXED}")
        print(f"Sweep A (const d/n={D_FIXED/N_FIXED_A:.3f}): m = {M_VALUES_A}, n = {N_FIXED_A}")
        print(f"Sweep B (const m/d={M_FIXED_B/D_FIXED:.2f}): m = {M_FIXED_B}, n = {N_VALUES_B}")
        print(f"Seeds per cell: {N_SEEDS}")

        # --- Sweep A ---
        results_a = run_sweep_a()
        # --- Sweep B ---
        results_b = run_sweep_b()

        save_results("exp14", {
            "sweep_a": results_a,
            "sweep_b": results_b,
        }, config={
            "m_values_a": M_VALUES_A,
            "n_fixed_a": N_FIXED_A,
            "m_fixed_b": M_FIXED_B,
            "n_values_b": N_VALUES_B,
            "d_fixed": D_FIXED,
            "dgp": DGP,
            "encoder": ENCODER,
            "n_seeds": N_SEEDS,
        })

    for ext in ("pdf", "png"):
        fig_a = plot_sweep_a(results_a)
        savefig(fig_a, f"exp14_sweep_a.{ext}", subdir="exp14")

        fig_b = plot_sweep_b(results_b)
        savefig(fig_b, f"exp14_sweep_b.{ext}", subdir="exp14")

        fig_c = plot_combined_summary(results_a, results_b)
        savefig(fig_c, f"exp14_combined_summary.{ext}", subdir="exp14")

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
