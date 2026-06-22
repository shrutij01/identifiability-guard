"""
Experiment 4 – Do metrics respond to correlation magnitude rather than
entanglement structure?

2-D sweep: correlation strength ρ  ×  condition number κ.
DGP = D2 (correlated), Encoder = E3 (linearly entangled).

For each metric we produce a heat-map with ρ on one axis and κ on the other.
If a metric conflates correlation with entanglement, its heat-map will show
variation along both axes; a well-behaved metric should vary only along the
κ-axis (entanglement) and be constant along the ρ-axis (correlation).
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    plot_heatmap,
    savefig,
    setup_plot_style,
    make_registry,
    display_name,
    DEFAULT_N_SAMPLES, DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    MAIN_METRICS, APX_METRICS,
)

# ── Configuration ──────────────────────────────────────────────────────────
CORRELATION_VALUES = [-0.9, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9]
CONDITION_NUMBERS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
DGP = "D2"
ENCODER = "E3"
N_SAMPLES = 100  # d=2 needs n=100 for the full ρ range (PSD constraint)
# Use d=2 so that the full ρ range (-1, 1) is feasible for a uniform
# correlation matrix (PSD requires ρ > -1/(d-1); for d=2 this is ρ > -1).
N_FACTORS = 2
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Multi-d configuration
D_VALUES = [2, 5, 10]
N_SAMPLES_BY_D = {2: 100, 5: 1000, 10: 1000}


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_correlation_x_condition_number(
    corr_values=CORRELATION_VALUES,
    cn_values=CONDITION_NUMBERS,
    n_samples=N_SAMPLES,
    n_factors=N_FACTORS,
    n_seeds=N_SEEDS,
    base_seed=BASE_SEED,
):
    """
    2D parameter sweep.

    Returns
    -------
    score_grids : dict[str, np.ndarray]
        ``{metric_name: array of shape (len(corr_values), len(cn_values))}``
        Each cell is the mean across seeds.
    """
    registry = make_registry()
    score_grids = {m: np.full((len(corr_values), len(cn_values)), np.nan)
                   for m in METRICS}

    for i, rho in enumerate(corr_values):
        # Check if this ρ is feasible for a uniform correlation matrix of size d.
        # A d×d matrix with all off-diagonal entries ρ is PSD iff ρ > -1/(d-1).
        min_rho = -1.0 / (n_factors - 1) if n_factors > 1 else -1.0
        if rho <= min_rho:
            print(f"    ρ={rho:+.2f} — skipped (not PSD for d={n_factors}; "
                  f"need ρ > {min_rho:.2f})")
            continue

        for j, kappa in enumerate(cn_values):
            print(f"    ρ={rho:+.2f}, κ={kappa:.1f}")

            def eval_one_seed(seed, _rho=rho, _kappa=kappa):
                return evaluate_dgp_encoder(
                    DGP, ENCODER,
                    n_samples=n_samples, n_factors=n_factors, seed=seed,
                    dgp_kwargs={"correlation": _rho},
                    encoder_kwargs={"condition_number": _kappa},
                    registry=registry,
                )

            _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=n_seeds,
                                             base_seed=base_seed)
            for m in METRICS:
                score_grids[m][i, j] = agg.get(m, {}).get("mean", np.nan)

    return score_grids


def plot_2d_heatmaps(score_grids, corr_values, cn_values, metrics=None):
    """One heatmap per metric arranged in a grid figure."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    n_metrics = len(metrics)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten()

    row_labels = [rf"$\rho={r:+.2f}$" for r in corr_values]
    col_labels = [rf"$\kappa={c:.0f}$" for c in cn_values]

    for idx, m in enumerate(metrics):
        ax = axes_flat[idx]
        data = score_grids[m]
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(display_name(m), fontsize=10)
        # Annotate cells
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                val = data[ii, jj]
                txt = f"{val:.2f}" if np.isfinite(val) else "—"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        r"Correlation ($\rho$) vs Entanglement ($\kappa \in \{1, 2, \ldots, 50\}$)"
        f"\n({DGP} + {ENCODER}, d={N_FACTORS}, n={N_SAMPLES})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_2d_heatmaps_main(score_grids, corr_values, cn_values):
    """Main-text variant: 1x3 heatmaps for MCC-P, DCI-D, R² with diverging colormap."""
    focus = ["mcc_pearson", "dci_disentanglement", "r2"]
    focus = [m for m in focus if m in score_grids]
    setup_plot_style()
    n = len(focus)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    row_labels = [rf"$\rho={r:+.1f}$" for r in corr_values]
    col_labels = [rf"$\kappa={c:.0f}$" for c in cn_values]

    for ax, m in zip(axes, focus):
        data = score_grids[m]
        # Diverging colormap centered on 1 (perfect identifiability)
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(display_name(m), fontsize=11)
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                val = data[ii, jj]
                txt = f"{val:.2f}" if np.isfinite(val) else "—"
                color = "white" if (np.isfinite(val) and val < 0.3) else "black"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=6, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        r"Correlation ($\rho$) vs Entanglement ($\kappa$)"
        f"\n({DGP} + {ENCODER}, d={N_FACTORS})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 4 – 2D sweep: correlation × condition number")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp04")
        score_grids = data["grids"]
        corr_values = config["correlation_values"]
        cn_values = config["condition_numbers"]
    else:
        score_grids = sweep_correlation_x_condition_number()
        corr_values = CORRELATION_VALUES
        cn_values = CONDITION_NUMBERS
        save_results("exp04", {"grids": score_grids}, config={
            "correlation_values": CORRELATION_VALUES,
            "condition_numbers": CONDITION_NUMBERS,
            "dgp": DGP,
            "encoder": ENCODER,
            "n_samples": N_SAMPLES,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ("pdf", "png"):
            fig = plot_2d_heatmaps(score_grids, corr_values, cn_values, metrics=mets)
            savefig(fig, f"exp04_corr_x_condition_number_heatmaps_{tag}.{ext}", subdir="exp04")

    # Main-text 1x3 heatmap (key metrics only)
    for ext in ("pdf", "png"):
        fig = plot_2d_heatmaps_main(score_grids, corr_values, cn_values)
        savefig(fig, f"exp04_corr_x_condition_number_main.{ext}", subdir="exp04")

    # ── Multi-d sweeps ────────────────────────────────────────────────────
    for d in D_VALUES:
        if d == N_FACTORS:
            continue  # d=2 already done above
        n_samp = N_SAMPLES_BY_D.get(d, 1000)
        exp_key = f"exp04_d{d}"

        if plot_only:
            try:
                d_data, d_config = load_results(exp_key)
                d_score_grids = d_data["grids"]
                d_corr = d_config["correlation_values"]
                d_cn = d_config["condition_numbers"]
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping d={d} plots")
                continue
        else:
            print(f"\n  ── d={d} (n={n_samp}) ──")
            d_score_grids = sweep_correlation_x_condition_number(
                n_samples=n_samp, n_factors=d,
            )
            d_corr = CORRELATION_VALUES
            d_cn = CONDITION_NUMBERS
            save_results(exp_key, {"grids": d_score_grids}, config={
                "correlation_values": CORRELATION_VALUES,
                "condition_numbers": CONDITION_NUMBERS,
                "dgp": DGP,
                "encoder": ENCODER,
                "n_samples": n_samp,
                "n_factors": d,
                "n_seeds": N_SEEDS,
            })

        tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags:
            for ext in ("pdf", "png"):
                fig = plot_2d_heatmaps(d_score_grids, d_corr, d_cn, metrics=mets)
                savefig(fig, f"exp04_corr_x_condition_number_heatmaps_d{d}_{tag}.{ext}", subdir="exp04")

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
