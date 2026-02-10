"""
Experiment 7 – Is it desirable to compress the ground-truth in the case of
redundancies in latent space and how does this depend on predictability of
the retained variables?

2-D sweep: redundancy level r  ×  compression m.
DGPs = D3 (single-factor redundant) and D4 (multi-factor redundant).
Encoder = E4 (undercomplete linear, retains m < d factors).

For each metric we produce a heat-map with r on one axis and m on the other.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    evaluate_dgp_encoder,
    multi_seed_evaluate,
    savefig,
    setup_plot_style,
    make_registry,
    display_name,
    DEFAULT_N_SAMPLES, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    ALL_METRICS,
)

# ── Configuration ──────────────────────────────────────────────────────────
D_TOTAL = 10
R_VALUES = [1, 2, 3, 4]                    # number of redundant factors
M_VALUES = list(range(D_TOTAL - 1, 2, -1))  # compressed dimensionality
N_SAMPLES = DEFAULT_N_SAMPLES
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(ALL_METRICS.keys())
DGPS = ["D3", "D4"]


# ── Experiment logic ───────────────────────────────────────────────────────

def sweep_redundancy_x_compression(
    dgp_name: str,
    d_total=D_TOTAL,
    r_values=R_VALUES,
    m_values=M_VALUES,
):
    """
    2D sweep r × m for a given DGP.

    Returns
    -------
    score_grids : dict[str, np.ndarray]
        shape = (len(r_values), len(m_values))
    """
    registry = make_registry()
    score_grids = {met: np.full((len(r_values), len(m_values)), np.nan)
                   for met in METRICS}

    for i, r in enumerate(r_values):
        for j, m in enumerate(m_values):
            # Skip invalid combinations
            if m >= d_total:
                continue
            # For D3, r must be < d/2
            if r >= d_total / 2:
                continue
            print(f"    {dgp_name}  r={r}, m={m}")

            def eval_one_seed(seed, _r=r, _m=m):
                dgp_kwargs = {"r": _r}
                return evaluate_dgp_encoder(
                    dgp_name, "E4",
                    n_samples=N_SAMPLES, n_factors=d_total, seed=seed,
                    dgp_kwargs=dgp_kwargs,
                    encoder_kwargs={"m": _m},
                    registry=registry,
                )

            try:
                _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                                 base_seed=BASE_SEED)
                for met in METRICS:
                    score_grids[met][i, j] = agg.get(met, {}).get("mean", np.nan)
            except Exception as e:
                print(f"      ⚠ failed: {e}")

    return score_grids


def plot_redundancy_compression_heatmaps(
    dgp_name, score_grids, r_values, m_values,
):
    """One heatmap per metric."""
    setup_plot_style()
    n_metrics = len(METRICS)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten()

    row_labels = [f"r={r}" for r in r_values]
    col_labels = [f"m={m}" for m in m_values]

    for idx, met in enumerate(METRICS):
        ax = axes_flat[idx]
        data = score_grids[met]
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(display_name(met), fontsize=10)
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                val = data[ii, jj]
                txt = f"{val:.2f}" if np.isfinite(val) else "—"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Redundancy (r) vs compression (m) – {dgp_name} + E4, d={D_TOTAL}",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 7 – Redundancy vs compression (D3/D4 + E4)")
    print("=" * 70)

    for dgp in DGPS:
        print(f"  DGP: {dgp}")
        grids = sweep_redundancy_x_compression(dgp)
        fig = plot_redundancy_compression_heatmaps(dgp, grids, R_VALUES, M_VALUES)
        savefig(fig, f"exp07_redundancy_compression_{dgp}.pdf", subdir="exp07")
        savefig(
            plot_redundancy_compression_heatmaps(dgp, grids, R_VALUES, M_VALUES),
            f"exp07_redundancy_compression_{dgp}.png", subdir="exp07",
        )

    print("Done.")


if __name__ == "__main__":
    main()
