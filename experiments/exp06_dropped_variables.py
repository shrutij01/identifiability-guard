"""
Experiment 6 – Do metrics report perfect identifiability for different numbers
of dropped variables if the ones learned are perfectly identified?

Encoder E4 (undercomplete linear) selects m < d factors.  We sweep m from 1 to
d−1.  For the *retained* factors the encoder is elementwise-linear ≡ perfectly
identifiable.

Plot: m (number of retained factors) on the X axis, metric score on the Y axis.
A well-behaved metric should stay at its maximum as long as the retained factors
are correctly identified.

Extra concern (from notes): does increasing the total number of variables d
(even if they add no information) inflate the metric?  We also sweep d for
fixed m to check.
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
    DEFAULT_N_SAMPLES, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    ALL_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_factors

# ── Configuration ──────────────────────────────────────────────────────────
DGPS = ["D1", "D2", "D3", "D4"]
D_TOTAL = 10          # total ground-truth factors
N_SAMPLES = DEFAULT_N_SAMPLES
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())


# ── Experiment 6a: sweep m (number of retained factors) ────────────────────

def sweep_dropped_variables(dgp_name: str, d_total=D_TOTAL):
    """
    For a given DGP, sweep m from d_total-1 down to 1.

    Returns
    -------
    m_values : list[int]
    means, ci_lo, ci_hi : dict[str, list[float]]
    """
    registry = make_registry()
    m_values = list(range(d_total, 0, -1))  # d, d-1, d-2, …, 1
    means = {met: [] for met in METRICS}
    ci_lo = {met: [] for met in METRICS}
    ci_hi = {met: [] for met in METRICS}

    for m in m_values:
        # When m == d, E4 raises (requires m < d).  Use E1 (elementwise
        # linear, retains all factors) as the natural d=m baseline.
        encoder = "E1" if m == d_total else "E4"
        enc_kw = {} if m == d_total else {"m": m}
        print(f"    {dgp_name}  m = {m}  (encoder={encoder})")

        def eval_one_seed(seed, _m=m, _enc=encoder, _enc_kw=enc_kw):
            return evaluate_dgp_encoder(
                dgp_name, _enc,
                n_samples=N_SAMPLES, n_factors=d_total, seed=seed,
                encoder_kwargs=_enc_kw,
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED)
        for met in METRICS:
            stats = agg.get(met, {})
            means[met].append(stats.get("mean", np.nan))
            ci_lo[met].append(stats.get("ci_lower", np.nan))
            ci_hi[met].append(stats.get("ci_upper", np.nan))

    return m_values, means, ci_lo, ci_hi


def plot_dropped_variables_single(dgp_name, m_values, means, ci_lo, ci_hi):
    """Single-panel figure for one DGP."""
    fig = plot_metrics_vs_xaxis_with_ci(
        m_values, means, ci_lo, ci_hi,
        xlabel="Number of retained factors m",
        title=f"Effect of dropping variables ({dgp_name} + E4, d={D_TOTAL})",
    )
    ax = fig.axes[0]
    ax.axvline(D_TOTAL, color="grey", ls="--", lw=1.0, label="d = m")
    # Extend xlim so the d=m vertical line is clearly visible
    ax.set_xlim(min(m_values) - 0.5, max(m_values) + 0.5)
    ax.legend(loc="best", ncol=2, fontsize=8)
    return fig


def plot_dropped_variables_all_dgps(all_results):
    """Multi-panel: one panel per DGP."""
    setup_plot_style()
    n_dgps = len(all_results)
    fig, axes = plt.subplots(1, n_dgps, figsize=(6 * n_dgps, 5), sharey=True)
    if n_dgps == 1:
        axes = [axes]

    for ax, (dgp_name, (m_vals, means, ci_lo, ci_hi)) in zip(axes, all_results.items()):
        for met in METRICS:
            c = get_color(met)
            ax.plot(m_vals, means[met], marker=get_marker(met), color=c,
                    label=display_name(met), markersize=4)
            ax.fill_between(m_vals, ci_lo[met], ci_hi[met], color=c, alpha=0.12)
        # Mark d = m point
        ax.axvline(D_TOTAL, color="grey", ls="--", lw=1.0, label="d = m")
        ax.set_xlabel("Retained factors m")
        ax.set_title(f"{dgp_name}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        # Extend xlim so the d=m vertical line is clearly visible
        ax.set_xlim(min(m_vals) - 0.5, max(m_vals) + 0.5)
        ax.invert_xaxis()  # high m on left → more factors retained

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(
        f"Do metrics report perfect identifiability as factors are dropped? (d={D_TOTAL})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Experiment 6b: sweep d (total factors) with fixed m ────────────────────

def sweep_total_factors(dgp_name: str, m_fixed: int = 3):
    """
    Fix m and increase d.  If metrics inflate with d even though the extra
    factors are not encoded, that is a problem.
    """
    registry = make_registry()
    d_values = list(range(m_fixed, m_fixed + 9))  # m, m+1, … m+8
    means = {met: [] for met in METRICS}
    ci_lo = {met: [] for met in METRICS}
    ci_hi = {met: [] for met in METRICS}

    for d in d_values:
        # When d == m, E4 raises (requires m < d).  Use E1 as the d=m
        # baseline (all factors retained, elementwise linear).
        encoder = "E1" if d == m_fixed else "E4"
        enc_kw = {} if d == m_fixed else {"m": m_fixed}
        print(f"    {dgp_name}  d = {d}, m = {m_fixed}  (encoder={encoder})")

        def eval_one_seed(seed, _d=d, _enc=encoder, _enc_kw=enc_kw):
            return evaluate_dgp_encoder(
                dgp_name, _enc,
                n_samples=N_SAMPLES, n_factors=_d, seed=seed,
                encoder_kwargs=_enc_kw,
                registry=registry,
            )

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED)
        for met in METRICS:
            stats = agg.get(met, {})
            means[met].append(stats.get("mean", np.nan))
            ci_lo[met].append(stats.get("ci_lower", np.nan))
            ci_hi[met].append(stats.get("ci_upper", np.nan))

    return d_values, means, ci_lo, ci_hi


def plot_inflated_dimensions(dgp_name, d_values, means, ci_lo, ci_hi, m_fixed):
    fig = plot_metrics_vs_xaxis_with_ci(
        d_values, means, ci_lo, ci_hi,
        xlabel="Total ground-truth factors d",
        title=f"Metric inflation with extra dimensions\n"
              f"({dgp_name} + E4, m={m_fixed} fixed)",
    )
    ax = fig.axes[0]
    ax.axvline(m_fixed, color="grey", ls="--", lw=1.0, label="d = m")
    # Extend xlim so the d=m vertical line is clearly visible
    ax.set_xlim(min(d_values) - 0.5, max(d_values) + 0.5)
    ax.legend(loc="best", ncol=2, fontsize=8)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 6 – Dropped variables & dimension inflation")
    print("=" * 70)

    # 6a: sweep m
    all_results_m = {}
    for dgp in DGPS:
        print(f"  DGP: {dgp}")
        m_vals, means, ci_lo, ci_hi = sweep_dropped_variables(dgp)
        all_results_m[dgp] = (m_vals, means, ci_lo, ci_hi)
        fig = plot_dropped_variables_single(dgp, m_vals, means, ci_lo, ci_hi)
        savefig(fig, f"exp06a_dropped_{dgp}.pdf", subdir="exp06")
        savefig(
            plot_dropped_variables_single(dgp, m_vals, means, ci_lo, ci_hi),
            f"exp06a_dropped_{dgp}.png", subdir="exp06",
        )

    fig_all = plot_dropped_variables_all_dgps(all_results_m)
    savefig(fig_all, "exp06a_dropped_all_dgps.pdf", subdir="exp06")
    savefig(
        plot_dropped_variables_all_dgps(all_results_m),
        "exp06a_dropped_all_dgps.png", subdir="exp06",
    )

    # 6b: sweep d with fixed m
    m_fixed = 3
    for dgp in ["D1", "D2"]:
        print(f"  DGP: {dgp}, fixed m={m_fixed}")
        d_vals, means, ci_lo, ci_hi = sweep_total_factors(dgp, m_fixed=m_fixed)
        fig = plot_inflated_dimensions(dgp, d_vals, means, ci_lo, ci_hi, m_fixed)
        savefig(fig, f"exp06b_inflate_{dgp}_m{m_fixed}.pdf", subdir="exp06")
        savefig(
            plot_inflated_dimensions(dgp, d_vals, means, ci_lo, ci_hi, m_fixed),
            f"exp06b_inflate_{dgp}_m{m_fixed}.png", subdir="exp06",
        )

    # Sensitivity sweeps: factors for D2 × E4 and D2 × E5
    from pathlib import Path
    out_dir = Path(RESULTS_DIR / "exp06")
    for enc in ["E4", "E5"]:
        print(f"\n  Running sensitivity sweep: factors for D2 × {enc} …")
        sweep_factors(
            dgp="D2",
            encoder=enc,
            factor_values=[3, 5, 7, 10],
            n_samples=N_SAMPLES,
            n_seeds=N_SEEDS,
            base_seed=BASE_SEED,
            output_dir=out_dir,
            metrics_to_compute=set(ALL_METRICS.keys()),
            n_factors_ground_truth=D_TOTAL,
        )

    print("Done.")


if __name__ == "__main__":
    main()
