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
    MAIN_METRICS, APX_METRICS,
    RESULTS_DIR,
)
from sensitivity import sweep_samples

# ── Configuration ──────────────────────────────────────────────────────────
DGPS = ["D1", "D2", "D3", "D4"]
ENCODERS = ["E1", "E2", "E3", "E7"]
SAMPLE_VALUES = [50, 100, 200, 500, 1000, 2000, 5000]
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)


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


def plot_sample_sensitivity_grid(results, metrics=None):
    """Grid of subplots: one per D×E combo."""
    if metrics is None:
        metrics = METRICS
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
        for m in metrics:
            c = get_color(m)
            ax.plot(SAMPLE_VALUES, r["means"][m], marker=get_marker(m),
                    color=c, label=display_name(m), markersize=4)
            ax.fill_between(SAMPLE_VALUES, r["ci_lo"][m], r["ci_hi"][m],
                            color=c, alpha=0.25)
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


def plot_sample_sensitivity_main(results, metrics=None):
    """Main-text 2x2 grid for key D×E combos (D1×E1, D1×E2, D2×E1, D1×E3)."""
    if metrics is None:
        metrics = METRICS_MAIN
    setup_plot_style()
    combos = ["D1×E1", "D1×E2", "D2×E1", "D1×E3"]
    combos = [k for k in combos if k in results]
    n = len(combos)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, key in enumerate(combos):
        ax = axes_flat[idx]
        r = results[key]
        for m in metrics:
            c = get_color(m)
            ax.plot(SAMPLE_VALUES, r["means"][m], marker=get_marker(m),
                    color=c, label=display_name(m), markersize=5)
            ax.fill_between(SAMPLE_VALUES, r["ci_lo"][m], r["ci_hi"][m],
                            color=c, alpha=0.25)
        ax.set_xlabel("n_samples")
        ax.set_title(key)
        ax.set_xscale("log")
        ax.set_xticks(SAMPLE_VALUES)
        ax.set_xticklabels([str(v) for v in SAMPLE_VALUES], fontsize=7)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    for idx in range(len(combos), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Metric score")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Sample sensitivity — key D×E combinations", y=1.01)
    fig.tight_layout()
    return fig


def plot_variance_heatmap(results, metrics=None):
    """Heatmap of metric std at the smallest sample size across D×E combos."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    keys = list(results.keys())
    small_idx = 0  # index into SAMPLE_VALUES for the smallest n
    data = np.full((len(keys), len(metrics)), np.nan)
    for i, key in enumerate(keys):
        for j, m in enumerate(metrics):
            data[i, j] = results[key]["stds"][m][small_idx]

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.9), max(4, len(keys) * 0.6)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([display_name(m) for m in metrics], rotation=45,
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


def plot_nan_count_heatmap(results, metrics=None):
    """Heatmap showing NaN counts at small n across combos."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    keys = list(results.keys())
    small_idx = 0
    data = np.zeros((len(keys), len(metrics)))
    for i, key in enumerate(keys):
        for j, m in enumerate(metrics):
            data[i, j] = results[key]["nan_counts"][m][small_idx]

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.9), max(4, len(keys) * 0.6)))
    im = ax.imshow(data, cmap="Reds", aspect="auto", vmin=0, vmax=N_SEEDS)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([display_name(m) for m in metrics], rotation=45,
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

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 10 – Sample sensitivity")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp10")
        results = data
        # Restore SAMPLE_VALUES for plotting
        global SAMPLE_VALUES
        SAMPLE_VALUES = config["sample_values"]
    else:
        results = run_all_combos()
        save_results("exp10", results, config={
            "sample_values": SAMPLE_VALUES,
            "dgps": DGPS,
            "encoders": ENCODERS,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            savefig(
                plot_sample_sensitivity_grid(results, metrics=mets),
                f"exp10_sample_sensitivity_grid_{tag}.{ext}", subdir="exp10",
            )
            savefig(
                plot_variance_heatmap(results, metrics=mets),
                f"exp10_variance_heatmap_{tag}.{ext}", subdir="exp10",
            )
            savefig(
                plot_nan_count_heatmap(results, metrics=mets),
                f"exp10_nan_count_heatmap_{tag}.{ext}", subdir="exp10",
            )

    # Main-text 2x2 grid
    for ext in ("pdf", "png"):
        savefig(
            plot_sample_sensitivity_main(results, metrics=METRICS_MAIN),
            f"exp10_sample_sensitivity_main.{ext}", subdir="exp10",
        )

    if not plot_only and not quick:
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
