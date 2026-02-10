"""
Experiment 5 – Is factor predictability conflated with disentanglement?

Contrast *functional dependency* between latent factors (D3: z2 = f(z1))
with *statistical correlation* (D2: correlated z1, z2) and compare metric
values.

Approach:
  • D3 with varying "difficulty" of the redundant function f (easy = linear,
    hard = highly non-linear).  We interpolate nonlinearity_strength of the
    redundant function.
  • D2 with varying correlation strength (ρ in [0, 1]).
  • Both use E1 (elementwise linear – perfectly identifiable).

Plot:
  Shared X axis = "difficulty / coupling strength" from 0 to 1.
    – For D2: coupling = |ρ|.
    – For D3: coupling = nonlinearity_strength of the redundant function
      (0 = z2 is linear copy of z1, 1 = z2 is a hard non-linear function).
  Y axis = metric score.
  Two panels (or overlaid lines) to compare D2 vs D3.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    ALL_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_nonlinearity, sweep_correlation
from src.dgp import D2Correlated, D3SingleRedundant
from src.evaluation.helpers import ENCODER_CLASSES
from src.encoders.base import BaseEncoder

# ── Configuration ──────────────────────────────────────────────────────────
ENCODER = "E1"  # encoder applied to both D2 and D3 sweeps
COUPLING_VALUES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())

# Use the same default nonlinear functions as the encoder base class
_DEFAULT_FNS = BaseEncoder._get_default_nonlinear_invertible_functions()

# Nonlinear functions of increasing difficulty — sourced from encoder defaults
_NL_FNS = [
    ("linear",      lambda x: x),
    ("cube",        _DEFAULT_FNS[3]),     # x**3
    ("tanh modified",        _DEFAULT_FNS[0]),     # tanh(2x³ - 0.1)
    # ("signed_sqrt", _DEFAULT_FNS[2]),     # sign(x)|x|^0.5
    # ("exp",         _DEFAULT_FNS[5]),     # exp(x)
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _interpolate_fn(strength: float):
    """Return a redundant function that interpolates between identity (strength=0)
    and a hard non-linear function (strength=1)."""
    hard_fn = _DEFAULT_FNS[0]  # tanh(2x³ - 0.1) — from encoder defaults
    def fn(x, _s=strength, _h=hard_fn):
        return (1 - _s) * x + _s * _h(x)
    return fn


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_d2_correlation(coupling_values=COUPLING_VALUES):
    """Sweep D2 correlation from 0 → 0.99."""
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for rho in coupling_values:
        print(f"    D2 ρ = {rho:.2f}")

        def eval_one_seed(seed, _rho=rho):
            dgp = D2Correlated(d=N_FACTORS, correlation=_rho, seed=seed)
            Z = dgp.sample(N_SAMPLES)
            enc = ENCODER_CLASSES[ENCODER](d=N_FACTORS, seed=seed)
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


def sweep_d3_nonlinearity(coupling_values=COUPLING_VALUES):
    """Sweep D3 redundant-function difficulty from linear → hard non-linear."""
    registry = make_registry()
    means = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}

    for s in coupling_values:
        print(f"    D3 nonlinearity strength = {s:.2f}")
        fn = _interpolate_fn(s)

        def eval_one_seed(seed, _fn=fn):
            dgp = D3SingleRedundant(d=N_FACTORS, r=1, redundant_fns=_fn, seed=seed)
            Z = dgp.sample(N_SAMPLES)
            enc = ENCODER_CLASSES[ENCODER](d=N_FACTORS, seed=seed)
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
    coupling_values, d2_results, d3_results,
):
    """Two-panel figure: D2-correlation vs D3-functional dependency."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    d2_means, d2_lo, d2_hi = d2_results
    d3_means, d3_lo, d3_hi = d3_results

    for m in METRICS:
        c = get_color(m)
        mk = get_marker(m)
        ax1.plot(coupling_values, d2_means[m], marker=mk, color=c,
                 label=display_name(m), markersize=4)
        ax1.fill_between(coupling_values, d2_lo[m], d2_hi[m],
                         color=c, alpha=0.12)

        ax2.plot(coupling_values, d3_means[m], marker=mk, color=c,
                 label=display_name(m), markersize=4)
        ax2.fill_between(coupling_values, d3_lo[m], d3_hi[m],
                         color=c, alpha=0.12)

    ax1.set_xlabel("Correlation strength |ρ|")
    ax1.set_ylabel("Metric score")
    ax1.set_title("D2 – Statistical correlation")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_xlabel("Redundant function difficulty")
    ax2.set_title("D3 – Functional dependency")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Factor predictability vs disentanglement\n"
        f"({ENCODER}, d={N_FACTORS}, n={N_SAMPLES})",
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


def plot_d3_specific_functions(fn_names, scores):
    """Bar chart comparing metrics for each D3 redundant function."""
    from utils import plot_grouped_bar
    fig = plot_grouped_bar(
        fn_names, scores,
        title="Metric scores for different D3 redundant functions\n"
              f"({ENCODER}, d={N_FACTORS})",
        xlabel="Redundant function type",
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 5 – Factor predictability conflated with disentanglement?")
    print("=" * 70)

    print("  Sweeping D2 correlation …")
    d2_results = sweep_d2_correlation()
    print("  Sweeping D3 nonlinearity difficulty …")
    d3_results = sweep_d3_nonlinearity()

    fig1 = plot_predictability_vs_disentanglement(
        COUPLING_VALUES, d2_results, d3_results,
    )
    savefig(fig1, "exp05_predictability_vs_disentanglement.pdf", subdir="exp05")
    savefig(
        plot_predictability_vs_disentanglement(COUPLING_VALUES, d2_results, d3_results),
        "exp05_predictability_vs_disentanglement.png", subdir="exp05",
    )

    print("  Evaluating D3 with specific non-linear functions …")
    fn_names, fn_scores = sweep_d3_specific_functions()
    fig2 = plot_d3_specific_functions(fn_names, fn_scores)
    savefig(fig2, "exp05_d3_specific_functions.pdf", subdir="exp05")
    savefig(
        plot_d3_specific_functions(fn_names, fn_scores),
        "exp05_d3_specific_functions.png", subdir="exp05",
    )

    # Sensitivity sweeps
    from pathlib import Path
    out_dir = Path(RESULTS_DIR / "exp05")

    # Sweep nonlinearity (encoder E2) with D3
    print("\n  Running sensitivity sweep: encoder nonlinearity for D3 × E1 …")
    sweep_nonlinearity(
        dgp="D3",
        nonlinearity_values=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        n_samples=N_SAMPLES,
        n_factors=N_FACTORS,
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED,
        output_dir=out_dir,
        metrics_to_compute=set(ALL_METRICS.keys()),
    )

    # Sweep correlation for D2 × ENCODER
    print(f"\n  Running sensitivity sweep: correlation for D2 × {ENCODER} …")
    sweep_correlation(
        encoder=ENCODER,
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
