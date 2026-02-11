"""
Experiment 4 – Do metrics respond to correlation magnitude rather than
entanglement structure?

2-D sweep: correlation strength ρ  ×  non-linearity strength α.
DGP = D2 (correlated), Encoder = E2 (elementwise non-linear).

For each metric we produce a heat-map with ρ on one axis and α on the other.
If a metric conflates correlation with entanglement, its heat-map will show
variation along both axes; a well-behaved metric should vary only along the
α-axis (entanglement) and be constant along the ρ-axis (correlation).
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
    ALL_METRICS,
)

# ── Configuration ──────────────────────────────────────────────────────────
CORRELATION_VALUES = [-0.9, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9]
NONLINEARITY_STRENGTHS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DGP = "D2"
ENCODER = "E2"
N_SAMPLES = DEFAULT_N_SAMPLES
# Use d=2 so that the full ρ range (-1, 1) is feasible for a uniform
# correlation matrix (PSD requires ρ > -1/(d-1); for d=2 this is ρ > -1).
N_FACTORS = 2
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_correlation_x_nonlinearity(
    corr_values=CORRELATION_VALUES,
    nl_values=NONLINEARITY_STRENGTHS,
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
        ``{metric_name: array of shape (len(corr_values), len(nl_values))}``
        Each cell is the mean across seeds.
    """
    registry = make_registry()
    score_grids = {m: np.full((len(corr_values), len(nl_values)), np.nan)
                   for m in METRICS}

    for i, rho in enumerate(corr_values):
        # Check if this ρ is feasible for a uniform correlation matrix of size d.
        # A d×d matrix with all off-diagonal entries ρ is PSD iff ρ > -1/(d-1).
        min_rho = -1.0 / (n_factors - 1) if n_factors > 1 else -1.0
        if rho <= min_rho:
            print(f"    ρ={rho:+.2f} — skipped (not PSD for d={n_factors}; "
                  f"need ρ > {min_rho:.2f})")
            continue

        for j, alpha in enumerate(nl_values):
            print(f"    ρ={rho:+.2f}, α={alpha:.2f}")

            def eval_one_seed(seed, _rho=rho, _alpha=alpha):
                return evaluate_dgp_encoder(
                    DGP, ENCODER,
                    n_samples=n_samples, n_factors=n_factors, seed=seed,
                    dgp_kwargs={"correlation": _rho},
                    encoder_kwargs={"nonlinearity_strength": _alpha},
                    registry=registry,
                )

            _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=n_seeds,
                                             base_seed=base_seed)
            for m in METRICS:
                score_grids[m][i, j] = agg.get(m, {}).get("mean", np.nan)

    return score_grids


def plot_2d_heatmaps(score_grids, corr_values, nl_values):
    """One heatmap per metric arranged in a grid figure."""
    setup_plot_style()
    n_metrics = len(METRICS)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten()

    row_labels = [f"ρ={r:+.2f}" for r in corr_values]
    col_labels = [f"α={a:.1f}" for a in nl_values]

    for idx, m in enumerate(METRICS):
        ax = axes_flat[idx]
        data = score_grids[m]
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
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
        "Correlation magnitude (ρ) vs entanglement strength (α)\n"
        f"({DGP} + {ENCODER}, d={N_FACTORS}, n={N_SAMPLES})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 4 – 2D sweep: correlation × nonlinearity")
    print("=" * 70)
    score_grids = sweep_correlation_x_nonlinearity()
    fig = plot_2d_heatmaps(score_grids, CORRELATION_VALUES, NONLINEARITY_STRENGTHS)
    savefig(fig, "exp04_corr_x_nonlinearity_heatmaps.pdf", subdir="exp04")
    savefig(
        plot_2d_heatmaps(score_grids, CORRELATION_VALUES, NONLINEARITY_STRENGTHS),
        "exp04_corr_x_nonlinearity_heatmaps.png", subdir="exp04",
    )
    print("Done.")


if __name__ == "__main__":
    main()
