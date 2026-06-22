"""
Experiment 3 – Does the sign of correlation affect metric values despite
identical dependence strength?

Plot: metric scores varying ρ from −0.99 to +0.99. We use D2 (correlated DGP)
with E1 (elementwise linear, reference) and E3 (linearly entangled, evaluated).
If metrics are well-behaved they should be symmetric around ρ=0 (same score for
+ρ and −ρ), because the sign of the correlation should not affect
identifiability.

We produce:
  • A single figure with ρ on the X axis and one line per metric
    (n_seeds averaged).
  • A multi-panel figure where each panel is a different encoder (E1 vs E3)
    to see if entanglement (κ) interacts with sign asymmetry.
  • A multi-d figure (d=2, 5, 10) to verify sign asymmetry beyond trivial d=2.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    plot_metrics_vs_xaxis_with_ci,
    plot_sweep_split,
    savefig,
    setup_plot_style,
    make_registry,
    get_color, get_marker, display_name,
    DEFAULT_N_SAMPLES, DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    MAIN_METRICS,
    APX_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_correlation

# ── Configuration ──────────────────────────────────────────────────────────
CORRELATION_VALUES = [
    -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99,
]
ENCODERS = ["E1", "E3"]  # E1 = elementwise linear (reference), E3 = linearly entangled (evaluated)
DGP = "D2"
# Override DEFAULT_N_SAMPLES: d=2 base case needs n=100 for full ρ range.
N_SAMPLES = 100
# Use d=2 so that the full ρ range (-1, 1) is feasible for a uniform
# correlation matrix (PSD requires ρ > -1/(d-1); for d=2 this is ρ > -1).
N_FACTORS = 2
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Multi-d configuration: verify sign asymmetry beyond trivial d=2.
D_VALUES = [2, 5, 10]
N_SAMPLES_BY_D = {2: 100, 5: 1000, 10: 1000}
MAIN_D = 5  # d used for main-text figures


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


def plot_sign_effect_single(encoder, rho_vals, means, ci_lo, ci_hi, metrics=None):
    """Single plot for one encoder."""
    fig = plot_metrics_vs_xaxis_with_ci(
        rho_vals, means, ci_lo, ci_hi,
        xlabel=r"Correlation ($\rho$)",
        title=f"Metric sensitivity to correlation sign ({DGP} + {encoder})",
        ylim=(-0.05, 1.05),
        metrics_to_plot=metrics,
    )
    ax = fig.axes[0]
    # Vertical line at ρ=0
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    # Ensure x-axis covers the full ρ range
    ax.set_xlim(rho_vals[0] - 0.03, rho_vals[-1] + 0.03)
    return fig


def plot_sign_effect_split(rho_vals, means, ci_lo, ci_hi,
                           invariant=None, varying=None, d=MAIN_D):
    """Main-text 1x2 split: invariant metrics (left) vs affected metrics (right).

    Uses E1 data only with d=5.  Left panel shows metrics unaffected by sign
    (e.g. MCC-S, DCI-D stay flat ≈1).  Right panel shows metrics that exhibit
    sign asymmetry (e.g. MCC-P, R²).

    NaN entries (infeasible ρ values due to PSD constraint) are filtered out
    so the x-axis is trimmed to the actual data range.
    """
    if invariant is None:
        invariant = ["mcc_spearman", "dci_disentanglement"]
    if varying is None:
        varying = ["mcc_pearson", "r2"]
    # Filter to metrics actually present
    invariant = [m for m in invariant if m in means]
    varying = [m for m in varying if m in means]

    # Filter out NaN entries (infeasible ρ values) to trim x-axis
    all_mets = invariant + varying
    # A point is valid if at least one metric has a non-NaN value
    valid = [i for i, _ in enumerate(rho_vals)
             if any(np.isfinite(means[m][i]) for m in all_mets if m in means)]
    if valid:
        rho_vals = [rho_vals[i] for i in valid]
        means = {m: [means[m][i] for i in valid] for m in means}
        ci_lo = {m: [ci_lo[m][i] for i in valid] for m in ci_lo}
        ci_hi = {m: [ci_hi[m][i] for i in valid] for m in ci_hi}

    fig = plot_sweep_split(
        rho_vals, means, ci_lo, ci_hi,
        invariant_metrics=invariant,
        varying_metrics=varying,
        xlabel=r"Correlation ($\rho$)",
        title_left="Sign-invariant metrics",
        title_right="Sign-sensitive metrics",
        ref_lines=[("v", 0, r"$\rho=0$")],
    )
    fig.suptitle(
        f"Correlation sign effect ({DGP} + E1, d={d})",
        y=1.02,
    )
    return fig


def plot_sign_effect_multi_encoder(all_results, metrics=None, d=N_FACTORS):
    """One subplot per encoder, all metrics overlaid."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    n_enc = len(all_results)
    fig, axes = plt.subplots(1, n_enc, figsize=(7 * n_enc, 5), sharey=True)
    if n_enc == 1:
        axes = [axes]

    for ax, (enc, (rho_vals, means, ci_lo, ci_hi)) in zip(axes, all_results.items()):
        for m in metrics:
            c = get_color(m)
            ax.plot(rho_vals, means[m], marker=get_marker(m), color=c,
                    label=display_name(m), markersize=4)
            ax.fill_between(rho_vals, ci_lo[m], ci_hi[m], color=c, alpha=0.25)
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel(r"Correlation ($\rho$)")
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
        f"({DGP}, d={d}, n={N_SAMPLES_BY_D.get(d, N_SAMPLES)})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_sign_effect_multi_d(all_d_results, metrics=None):
    """Multi-row figure: one row per d value, columns per encoder.

    Parameters
    ----------
    all_d_results : dict[int, dict[str, tuple]]
        ``{d: {encoder: (rho_vals, means, ci_lo, ci_hi)}}``
    """
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    d_vals = sorted(all_d_results.keys())
    n_rows = len(d_vals)
    n_cols = len(ENCODERS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4.5 * n_rows),
                             sharey=True, squeeze=False)

    for row, d in enumerate(d_vals):
        enc_results = all_d_results[d]
        for col, enc in enumerate(ENCODERS):
            ax = axes[row, col]
            if enc not in enc_results:
                ax.set_visible(False)
                continue
            rho_vals, means, ci_lo, ci_hi = enc_results[enc]
            for m in metrics:
                c = get_color(m)
                ax.plot(rho_vals, means[m], marker=get_marker(m), color=c,
                        label=display_name(m), markersize=4)
                ax.fill_between(rho_vals, ci_lo[m], ci_hi[m], color=c, alpha=0.25)
            ax.axvline(0, color="grey", ls="--", lw=0.8)
            ax.set_xlabel(r"Correlation ($\rho$)")
            ax.set_title(f"d={d}, {enc}")
            ax.grid(True, alpha=0.3)
            # PSD limit for negative rho
            min_rho = -1.0 / (d - 1) if d > 1 else -1.0
            ax.set_xlim(min(rho_vals[0], min_rho) - 0.03,
                        rho_vals[-1] + 0.03)
            ax.set_ylim(-0.05, 1.05)

    # y-label on left column only
    for row in range(n_rows):
        axes[row, 0].set_ylabel("Metric score")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Sign asymmetry across dimensionalities\n"
        f"({DGP}, d ∈ {{{', '.join(str(d) for d in d_vals)}}})",
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 3 – Sign of correlation effect on metrics")
    print("=" * 70)

    # ── d=2 sweep (original, backward-compatible) ─────────────────────────
    if plot_only:
        data, config = load_results("exp03")
        rho_vals = config["correlation_values"]
        all_results = {}
        for enc in config["encoders"]:
            d = data[enc]
            all_results[enc] = (rho_vals, d["means"], d["ci_lo"], d["ci_hi"])
    else:
        all_results = {}
        save_data = {}
        for enc in ENCODERS:
            print(f"  Encoder: {enc}")
            rho_vals, means, ci_lo, ci_hi = sweep_correlation_sign(enc)
            all_results[enc] = (rho_vals, means, ci_lo, ci_hi)
            save_data[enc] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}
        save_results("exp03", save_data, config={
            "correlation_values": CORRELATION_VALUES,
            "encoders": ENCODERS,
            "dgp": DGP,
            "n_samples": N_SAMPLES,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    # d=2 plots (backward compat)
    for enc in (config["encoders"] if plot_only else ENCODERS):
        rho_vals_enc, means, ci_lo, ci_hi = all_results[enc]
        tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags:
            for ext in ("pdf", "png"):
                fig = plot_sign_effect_single(enc, rho_vals_enc, means, ci_lo, ci_hi, metrics=mets)
                savefig(fig, f"exp03_sign_correlation_{enc}_{tag}.{ext}", subdir="exp03")

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            fig = plot_sign_effect_multi_encoder(all_results, metrics=mets, d=N_FACTORS)
            savefig(fig, f"exp03_sign_correlation_all_encoders_{tag}.{ext}", subdir="exp03")

    # ── Multi-d sweeps (d=5, d=10) ────────────────────────────────────────
    all_d_results = {N_FACTORS: all_results}  # d=2 already computed

    for d in D_VALUES:
        if d == N_FACTORS:
            continue  # already done above
        n_samp = N_SAMPLES_BY_D.get(d, N_SAMPLES)
        exp_key = f"exp03_d{d}"

        if plot_only:
            try:
                d_data, d_config = load_results(exp_key)
                d_rho_vals = d_config["correlation_values"]
                d_results = {}
                for enc in d_config["encoders"]:
                    dd = d_data[enc]
                    d_results[enc] = (d_rho_vals, dd["means"], dd["ci_lo"], dd["ci_hi"])
                all_d_results[d] = d_results
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping d={d} plots")
                continue
        else:
            print(f"\n  ── d={d} (n={n_samp}) ──")
            d_results = {}
            d_save = {}
            for enc in ENCODERS:
                print(f"  Encoder: {enc}, d={d}")
                rho_vals, means, ci_lo, ci_hi = sweep_correlation_sign(
                    enc, n_samples=n_samp, n_factors=d,
                )
                d_results[enc] = (rho_vals, means, ci_lo, ci_hi)
                d_save[enc] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi}
            save_results(exp_key, d_save, config={
                "correlation_values": CORRELATION_VALUES,
                "encoders": ENCODERS,
                "dgp": DGP,
                "n_samples": n_samp,
                "n_factors": d,
                "n_seeds": N_SEEDS,
            })
            all_d_results[d] = d_results

        # Per-d multi-encoder plot
        tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags:
            for ext in ("pdf", "png"):
                fig = plot_sign_effect_multi_encoder(d_results, metrics=mets, d=d)
                savefig(fig, f"exp03_sign_correlation_all_encoders_d{d}_{tag}.{ext}",
                        subdir="exp03")

    # ── Main-text split figure (E1, d=MAIN_D) ────────────────────────────
    if MAIN_D in all_d_results and "E1" in all_d_results[MAIN_D]:
        rho_vals_e1, means_e1, ci_lo_e1, ci_hi_e1 = all_d_results[MAIN_D]["E1"]
        for ext in ("pdf", "png"):
            fig = plot_sign_effect_split(rho_vals_e1, means_e1, ci_lo_e1, ci_hi_e1,
                                         d=MAIN_D)
            savefig(fig, f"exp03_sign_split_main.{ext}", subdir="exp03")

    # ── Combined multi-d figure ───────────────────────────────────────────
    if len(all_d_results) > 1:
        tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags:
            for ext in ("pdf", "png"):
                fig = plot_sign_effect_multi_d(all_d_results, metrics=mets)
                savefig(fig, f"exp03_sign_correlation_multi_d_{tag}.{ext}", subdir="exp03")

    if not plot_only and not quick:
        # Sensitivity sweeps: correlation for E1 and E3
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
