"""
Experiment 5 – Is factor predictability conflated with disentanglement?

Contrast *functional dependency* between latent factors (D3: z2 = f(z1))
with *statistical correlation* (D2: correlated z1, z2) and compare metric
values.

Approach:
  • D3 with varying non-linearity α of the redundant function f (α=0 linear,
    α=1 fully non-linear).
  • D2 with varying correlation ρ in [0, 1].
  • Both use E1 (elementwise linear – perfectly identifiable).

Plot:
  Two panels comparing D2 (correlation ρ) vs D3 (non-linearity α).
  Y axis = metric score.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_with_arrays,
    multi_seed_evaluate,
    savefig,
    setup_plot_style,
    make_registry,
    get_color, get_marker, display_name,
    DEFAULT_N_SAMPLES, DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    MAIN_METRICS, APX_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_nonlinearity, sweep_correlation
from identifiability_guard.dgp import D2Correlated, D3SingleRedundant
from identifiability_guard.evaluation.helpers import ENCODER_CLASSES
from identifiability_guard.encoders.base import BaseEncoder

# ── Configuration ──────────────────────────────────────────────────────────
ENCODER = "E1"  # encoder applied to both D2 and D3 sweeps
COUPLING_VALUES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Multi-d: confirm coupling effects scale with d
D_VALUES = [5, 10]

# Use the same default nonlinear functions as the encoder base class
_DEFAULT_FNS = BaseEncoder._get_default_nonlinear_invertible_functions()

# Nonlinear functions of increasing non-linearity — sourced from encoder defaults
_NL_FNS = [
    ("linear",      lambda x: x),
    ("cube",        _DEFAULT_FNS[3]),     # x**3
    ("tanh modified",        _DEFAULT_FNS[0]),     # tanh(2x³ - 0.1)
    # ("signed_sqrt", _DEFAULT_FNS[2]),     # sign(x)|x|^0.5
    # ("exp",         _DEFAULT_FNS[5]),     # exp(x)
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _interpolate_fn(alpha: float):
    """Return a redundant function that interpolates between identity (α=0)
    and a fully non-linear function (α=1)."""
    hard_fn = _DEFAULT_FNS[0]  # tanh(2x³ - 0.1) — from encoder defaults
    def fn(x, _a=alpha, _h=hard_fn):
        return (1 - _a) * x + _a * _h(x)
    return fn


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_d2_correlation(coupling_values=COUPLING_VALUES, n_factors=None, n_samples=None):
    """Sweep D2 correlation from 0 → 0.99."""
    if n_factors is None:
        n_factors = N_FACTORS
    if n_samples is None:
        n_samples = N_SAMPLES
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for rho in coupling_values:
        print(f"    D2 ρ = {rho:.2f}")

        def eval_one_seed(seed, _rho=rho):
            dgp = D2Correlated(d=n_factors, correlation=_rho, seed=seed)
            Z = dgp.sample(n_samples)
            enc = ENCODER_CLASSES[ENCODER](d=n_factors, seed=seed)
            Z_hat = enc.encode(Z)
            return evaluate_with_arrays(Z, Z_hat, registry=registry)

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED)
        for m in METRICS:
            stats = agg.get(m, {})
            means[m].append(stats.get("mean", np.nan))
            ci_lo[m].append(stats.get("ci_lower", np.nan))
            ci_hi[m].append(stats.get("ci_upper", np.nan))

    return means, ci_lo, ci_hi


def sweep_d3_nonlinearity(coupling_values=COUPLING_VALUES, n_factors=None, n_samples=None):
    """Sweep D3 non-linearity α from 0 (linear) to 1 (fully non-linear)."""
    if n_factors is None:
        n_factors = N_FACTORS
    if n_samples is None:
        n_samples = N_SAMPLES
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for s in coupling_values:
        print(f"    D3 α = {s:.2f}")
        fn = _interpolate_fn(s)

        def eval_one_seed(seed, _fn=fn):
            dgp = D3SingleRedundant(d=n_factors, r=1, redundant_fns=_fn, seed=seed)
            Z = dgp.sample(n_samples)
            enc = ENCODER_CLASSES[ENCODER](d=n_factors, seed=seed)
            Z_hat = enc.encode(Z)
            return evaluate_with_arrays(Z, Z_hat, registry=registry)

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED)
        for m in METRICS:
            stats = agg.get(m, {})
            means[m].append(stats.get("mean", np.nan))
            ci_lo[m].append(stats.get("ci_lower", np.nan))
            ci_hi[m].append(stats.get("ci_upper", np.nan))

    return means, ci_lo, ci_hi


def plot_predictability_vs_disentanglement(
    coupling_values, d2_results, d3_results, metrics=None,
):
    """Two-panel figure: D2-correlation vs D3-functional dependency."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    d2_means, d2_lo, d2_hi = d2_results
    d3_means, d3_lo, d3_hi = d3_results

    for m in metrics:
        c = get_color(m)
        mk = get_marker(m)
        ax1.plot(coupling_values, d2_means[m], marker=mk, color=c,
                 label=display_name(m), markersize=4)
        ax1.fill_between(coupling_values, d2_lo[m], d2_hi[m],
                         color=c, alpha=0.25)

        ax2.plot(coupling_values, d3_means[m], marker=mk, color=c,
                 label=display_name(m), markersize=4)
        ax2.fill_between(coupling_values, d3_lo[m], d3_hi[m],
                         color=c, alpha=0.25)

    ax1.set_xlabel(r"Correlation ($\rho$)")
    ax1.set_ylabel("Metric score")
    ax1.set_title(r"D2 – Correlation strength ($\rho$)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_xlabel(r"Non-linearity ($\alpha$)")
    ax2.set_title(r"D3 – Redundancy non-linearity ($\alpha$)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Factor predictability vs disentanglement\n"
        r"(coupling strength: $\rho$ for D2, $\alpha$ for D3)"
        f"  ({ENCODER}, d={N_FACTORS}, n={N_SAMPLES})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Also: D3 with different specific non-linear functions ──────────────────

def sweep_d3_specific_functions():
    """Evaluate D3 with each named non-linear function."""
    registry = make_registry()
    fn_names = [name for name, _ in _NL_FNS]
    scores = {m: [] for m in METRICS}

    for fn_name, fn in _NL_FNS:
        print(f"    D3 function: {fn_name}")

        def eval_one_seed(seed, _fn=fn):
            dgp = D3SingleRedundant(d=N_FACTORS, r=1, redundant_fns=_fn, seed=seed)
            Z = dgp.sample(N_SAMPLES)
            enc = ENCODER_CLASSES[ENCODER](d=N_FACTORS, seed=seed)
            Z_hat = enc.encode(Z)
            return evaluate_with_arrays(Z, Z_hat, registry=registry)

        _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                         base_seed=BASE_SEED)
        for m in METRICS:
            scores[m].append(agg.get(m, {}).get("mean", np.nan))

    return fn_names, scores


def plot_d3_specific_functions(fn_names, scores, metrics=None):
    """Bar chart comparing metrics for each D3 redundant function."""
    if metrics is None:
        metrics = METRICS
    scores = {m: v for m, v in scores.items() if m in metrics}
    from utils import plot_grouped_bar
    fig = plot_grouped_bar(
        fn_names, scores,
        title="Metric scores for different D3 redundant functions\n"
              f"({ENCODER}, d={N_FACTORS})",
        xlabel="Redundant function type",
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 5 – Factor predictability conflated with disentanglement?")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp05")
        coupling_values = config["coupling_values"]
        d2_means, d2_lo, d2_hi = data["d2"]["means"], data["d2"]["ci_lo"], data["d2"]["ci_hi"]
        d3_means, d3_lo, d3_hi = data["d3"]["means"], data["d3"]["ci_lo"], data["d3"]["ci_hi"]
        d2_results = (d2_means, d2_lo, d2_hi)
        d3_results = (d3_means, d3_lo, d3_hi)
        fn_names = config["fn_names"]
        fn_scores = data["d3_fns"]
    else:
        coupling_values = COUPLING_VALUES
        print("  Sweeping D2 correlation …")
        d2_results = sweep_d2_correlation()
        print("  Sweeping D3 non-linearity α …")
        d3_results = sweep_d3_nonlinearity()

        print("  Evaluating D3 with specific non-linear functions …")
        fn_names, fn_scores = sweep_d3_specific_functions()

        d2_means, d2_lo, d2_hi = d2_results
        d3_means, d3_lo, d3_hi = d3_results
        save_results("exp05", {
            "d2": {"means": d2_means, "ci_lo": d2_lo, "ci_hi": d2_hi},
            "d3": {"means": d3_means, "ci_lo": d3_lo, "ci_hi": d3_hi},
            "d3_fns": fn_scores,
        }, config={
            "coupling_values": COUPLING_VALUES,
            "fn_names": fn_names,
            "encoder": ENCODER,
            "n_samples": N_SAMPLES,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            fig = plot_predictability_vs_disentanglement(
                coupling_values, d2_results, d3_results, metrics=mets,
            )
            savefig(fig, f"exp05_predictability_vs_disentanglement_{tag}.{ext}", subdir="exp05")

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            fig = plot_d3_specific_functions(fn_names, fn_scores, metrics=mets)
            savefig(fig, f"exp05_d3_specific_functions_{tag}.{ext}", subdir="exp05")

    if not plot_only and not quick:
        # Sensitivity sweeps
        from pathlib import Path
        out_dir = Path(RESULTS_DIR / "exp05")

        print("\n  Running sensitivity sweep: encoder nonlinearity for D3 × E1 …")
        sweep_nonlinearity(
            dgp="D3",
            nonlinearity_values=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            n_samples=N_SAMPLES,
            n_factors=N_FACTORS,
            n_seeds=N_SEEDS,
            base_seed=BASE_SEED,
            output_dir=out_dir,
            metrics_to_compute=APX_METRICS,
        )

        print(f"\n  Running sensitivity sweep: correlation for D2 × {ENCODER} …")
        sweep_correlation(
            encoder=ENCODER,
            correlation_values=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
            n_samples=N_SAMPLES,
            n_factors=N_FACTORS,
            n_seeds=N_SEEDS,
            base_seed=BASE_SEED,
            output_dir=out_dir,
            metrics_to_compute=APX_METRICS,
        )

    # ── Multi-d sweep (d=10) ──────────────────────────────────────────────
    for d in D_VALUES:
        if d == N_FACTORS:
            continue
        exp_key = f"exp05_d{d}"

        if plot_only:
            try:
                d_data, d_config = load_results(exp_key)
                d_coupling = d_config["coupling_values"]
                d_d2 = (d_data["d2"]["means"], d_data["d2"]["ci_lo"], d_data["d2"]["ci_hi"])
                d_d3 = (d_data["d3"]["means"], d_data["d3"]["ci_lo"], d_data["d3"]["ci_hi"])
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping d={d}")
                continue
        else:
            print(f"\n  ── d={d} ──")
            print(f"  Sweeping D2 correlation (d={d}) …")
            d_d2 = sweep_d2_correlation(n_factors=d)
            print(f"  Sweeping D3 non-linearity (d={d}) …")
            d_d3 = sweep_d3_nonlinearity(n_factors=d)

            d2_means, d2_lo, d2_hi = d_d2
            d3_means, d3_lo, d3_hi = d_d3
            save_results(exp_key, {
                "d2": {"means": d2_means, "ci_lo": d2_lo, "ci_hi": d2_hi},
                "d3": {"means": d3_means, "ci_lo": d3_lo, "ci_hi": d3_hi},
            }, config={
                "coupling_values": COUPLING_VALUES,
                "encoder": ENCODER,
                "n_samples": N_SAMPLES,
                "n_factors": d,
                "n_seeds": N_SEEDS,
            })

        tags_d = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags_d:
            for ext in ("pdf", "png"):
                fig = plot_predictability_vs_disentanglement(
                    coupling_values if not plot_only else d_coupling,
                    d_d2, d_d3, metrics=mets,
                )
                savefig(fig, f"exp05_predictability_vs_disentanglement_d{d}_{tag}.{ext}", subdir="exp05")

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
