"""
Experiment 10 – Are metrics sensitive to the number of samples?

For every D × E combination, plot metric score vs n_samples with error bounds
(over multiple seeds).  The goal is to reveal:
  • How many samples each metric needs to stabilise.
  • Whether MI-based methods or DCI are particularly sample-hungry.
  • Whether any metric produces NaN / Inf at small sample sizes.

Output:
  • A grid of small plots (one per D × E combination): each plot has
    n_samples on the X axis and one line per metric.
  • A summary heatmap of "variance at n=200" across D × E × metric.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    savefig,
    setup_plot_style,
    make_registry,
    get_color, get_marker, display_name,
    DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    ALL_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_samples

# ── Configuration ──────────────────────────────────────────────────────────
DGPS = ["D1", "D2", "D3", "D4"]
ENCODERS = ["E1", "E2", "E3"]
SAMPLE_VALUES = [50, 100, 200, 500, 1000]
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_samples_for_combo(dgp_name, encoder_name):
    """
    Sweep n_samples for one D × E combination.

    Returns
    -------
    means, stds : dict[str, list[float]]
    nan_counts  : dict[str, list[int]]   – how many seeds produced NaN
    """
    registry = make_registry()
    means = {m: [] for m in METRICS}
    stds = {m: [] for m in METRICS}
    ci_lo = {m: [] for m in METRICS}
    ci_hi = {m: [] for m in METRICS}
    nan_counts = {m: [] for m in METRICS}

    for n in SAMPLE_VALUES:
        print(f"      n = {n}")

        def eval_one_seed(seed, _n=n):
            return evaluate_dgp_encoder(
                dgp_name, encoder_name,
                n_samples=_n, n_factors=N_FACTORS, seed=seed,
                registry=registry,
            )

        raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                        base_seed=BASE_SEED)
        for m in METRICS:
            stats = agg.get(m, {})
            means[m].append(stats.get("mean", np.nan))
            stds[m].append(stats.get("std", np.nan))
            ci_lo[m].append(stats.get("ci_lower", np.nan))
            ci_hi[m].append(stats.get("ci_upper", np.nan))
            vals = raw.get(m, [])
            nan_counts[m].append(int(sum(np.isnan(v) for v in vals)))

    return means, stds, ci_lo, ci_hi, nan_counts


def run_all_combos():
    results = {}
    for dgp in DGPS:
        for enc in ENCODERS:
            key = f"{dgp}×{enc}"
            print(f"    {key}")
            means, stds, ci_lo, ci_hi, nan_counts = sweep_samples_for_combo(dgp, enc)
            results[key] = {
                "means": means, "stds": stds,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "nan_counts": nan_counts,
            }
    return results


def plot_sample_sensitivity_grid(results):
    """Grid of subplots: one per D×E combo."""
    setup_plot_style()
    keys = list(results.keys())
    n = len(keys)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, key in enumerate(keys):
        ax = axes_flat[idx]
        r = results[key]
        for m in METRICS:
            c = get_color(m)
            ax.plot(SAMPLE_VALUES, r["means"][m], marker=get_marker(m),
                    color=c, label=display_name(m), markersize=4)
            ax.fill_between(SAMPLE_VALUES, r["ci_lo"][m], r["ci_hi"][m],
                            color=c, alpha=0.1)
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
    fig.suptitle(
        "Metric sensitivity to sample size\n(with 95 % CI over seeds)",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_variance_heatmap(results):
    """Heatmap of metric std at the smallest sample size across D×E combos."""
    setup_plot_style()
    keys = list(results.keys())
    small_idx = 0  # index into SAMPLE_VALUES for the smallest n
    data = np.full((len(keys), len(METRICS)), np.nan)
    for i, key in enumerate(keys):
        for j, m in enumerate(METRICS):
            data[i, j] = results[key]["stds"][m][small_idx]

    fig, ax = plt.subplots(figsize=(max(8, len(METRICS) * 0.9), max(4, len(keys) * 0.6)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([display_name(m) for m in METRICS], rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=9)
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            v = data[ii, jj]
            txt = f"{v:.3f}" if np.isfinite(v) else "NaN"
            ax.text(jj, ii, txt, ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Metric std at n={SAMPLE_VALUES[small_idx]} (over {N_SEEDS} seeds)")
    fig.tight_layout()
    return fig


def plot_nan_count_heatmap(results):
    """Heatmap showing NaN counts at small n across combos."""
    setup_plot_style()
    keys = list(results.keys())
    small_idx = 0
    data = np.zeros((len(keys), len(METRICS)))
    for i, key in enumerate(keys):
        for j, m in enumerate(METRICS):
            data[i, j] = results[key]["nan_counts"][m][small_idx]

    fig, ax = plt.subplots(figsize=(max(8, len(METRICS) * 0.9), max(4, len(keys) * 0.6)))
    im = ax.imshow(data, cmap="Reds", aspect="auto", vmin=0, vmax=N_SEEDS)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([display_name(m) for m in METRICS], rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=9)
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            v = int(data[ii, jj])
            ax.text(jj, ii, str(v), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"NaN count at n={SAMPLE_VALUES[small_idx]} (out of {N_SEEDS} seeds)")
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 10 – Sample sensitivity")
    print("=" * 70)
    results = run_all_combos()

    fig1 = plot_sample_sensitivity_grid(results)
    savefig(fig1, "exp10_sample_sensitivity_grid.pdf", subdir="exp10")
    savefig(
        plot_sample_sensitivity_grid(results),
        "exp10_sample_sensitivity_grid.png", subdir="exp10",
    )

    fig2 = plot_variance_heatmap(results)
    savefig(fig2, "exp10_variance_heatmap.pdf", subdir="exp10")
    savefig(
        plot_variance_heatmap(results),
        "exp10_variance_heatmap.png", subdir="exp10",
    )

    fig3 = plot_nan_count_heatmap(results)
    savefig(fig3, "exp10_nan_count_heatmap.pdf", subdir="exp10")
    savefig(
        plot_nan_count_heatmap(results),
        "exp10_nan_count_heatmap.png", subdir="exp10",
    )

    # Sensitivity sweep: sample size for D2 × E3
    print("\n  Running sensitivity sweep: samples for D2 × E3 …")
    from pathlib import Path
    sweep_samples(
        dgp="D2",
        encoder="E3",
        sample_values=SAMPLE_VALUES,
        n_factors=N_FACTORS,
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED,
        output_dir=Path(RESULTS_DIR / "exp10"),
        metrics_to_compute=set(ALL_METRICS.keys()),
    )

    print("Done.")


if __name__ == "__main__":
    main()
