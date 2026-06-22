"""
Experiment 7 – Is it desirable to compress the ground-truth in the case of
redundancies in latent space and how does this depend on predictability of
the retained variables?

2-D sweep: redundancy level r  ×  compression m.
DGPs = D3 (single-factor redundant) and D4 (multi-factor redundant).
Encoder = E4 (undercomplete linear, retains m < d factors).

For each metric we produce a heat-map with r on one axis and m on the other.
"""

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
    MAIN_METRICS, APX_METRICS,
)

# ── Configuration ──────────────────────────────────────────────────────────
D_TOTAL = 10
R_VALUES = [1, 2, 3, 4]                    # number of redundant factors
M_VALUES = list(range(D_TOTAL - 1, 2, -1))  # compressed dimensionality
N_SAMPLES = DEFAULT_N_SAMPLES
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)
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
    dgp_name, score_grids, r_values, m_values, metrics=None,
):
    """One heatmap per metric."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    n_metrics = len(metrics)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten()

    row_labels = [f"r={r}" for r in r_values]
    col_labels = [f"m={m}" for m in m_values]

    for idx, met in enumerate(metrics):
        ax = axes_flat[idx]
        data = score_grids[met]
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
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


def plot_redundancy_main(dgp_name, score_grids, r_values, m_values):
    """Main-text variant: 1x3 heatmaps for key metrics with diverging colormap."""
    focus = ["mcc_pearson", "dci_disentanglement", "r2"]
    focus = [m for m in focus if m in score_grids]
    setup_plot_style()
    n = len(focus)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    row_labels = [f"r={r}" for r in r_values]
    col_labels = [f"m={m}" for m in m_values]

    for ax, met in zip(axes, focus):
        data = score_grids[met]
        # Diverging colormap centered on 1 (perfect = green, broken = red)
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(display_name(met), fontsize=11)
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                val = data[ii, jj]
                txt = f"{val:.2f}" if np.isfinite(val) else "—"
                color = "white" if (np.isfinite(val) and val < 0.3) else "black"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=6,
                        color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Redundancy (r) vs compression (m) — {dgp_name} + E4, d={D_TOTAL}",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 7 – Redundancy vs compression (D3/D4 + E4)")
    print("=" * 70)

    if plot_only:
        data, config = load_results("exp07")
        dgps = config["dgps"]
        r_values = config["r_values"]
        m_values = config["m_values"]
        for dgp in dgps:
            grids = data[dgp]["grids"]
            tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
            for tag, mets in tags:
                for ext in ["pdf", "png"]:
                    fig = plot_redundancy_compression_heatmaps(
                        dgp, grids, r_values, m_values, metrics=mets,
                    )
                    savefig(fig, f"exp07_redundancy_compression_{dgp}_{tag}.{ext}",
                            subdir="exp07")
            # Main-text 1x3 heatmap
            for ext in ["pdf", "png"]:
                fig = plot_redundancy_main(dgp, grids, r_values, m_values)
                savefig(fig, f"exp07_redundancy_main_{dgp}.{ext}", subdir="exp07")
    else:
        save_data = {}
        for dgp in DGPS:
            print(f"  DGP: {dgp}")
            grids = sweep_redundancy_x_compression(dgp)
            save_data[dgp] = {"grids": grids}
            tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
            for tag, mets in tags:
                for ext in ["pdf", "png"]:
                    fig = plot_redundancy_compression_heatmaps(
                        dgp, grids, R_VALUES, M_VALUES, metrics=mets,
                    )
                    savefig(fig, f"exp07_redundancy_compression_{dgp}_{tag}.{ext}",
                            subdir="exp07")
            # Main-text 1x3 heatmap
            for ext in ["pdf", "png"]:
                fig = plot_redundancy_main(dgp, grids, R_VALUES, M_VALUES)
                savefig(fig, f"exp07_redundancy_main_{dgp}.{ext}", subdir="exp07")
        save_results("exp07", save_data, config={
            "dgps": DGPS,
            "r_values": R_VALUES,
            "m_values": M_VALUES,
            "d_total": D_TOTAL,
            "n_samples": N_SAMPLES,
            "n_seeds": N_SEEDS,
        })

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
