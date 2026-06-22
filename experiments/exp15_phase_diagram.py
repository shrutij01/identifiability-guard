"""
Experiment 15 -- Phase diagram: where do metrics break?

Heatmap of metric reliability under null encoder E9 (output independent of
input -- all scores should be 0) as a function of m/d and m/n.

  * Green (score ≈ 0): metric is trustworthy at this operating point.
  * Red (score > 0): false positive -- metric reports identifiability
    where none exists.

This figure shows two failure modes:
  1. m/d > 1 causes structural misspecification (metrics don't know which
     codes to evaluate).
  2. m/n > 1 causes statistical misspecification (metrics can't reliably
     estimate anything).

Practical relevance: sparse autoencoders in mech-interp operate at m/n
ratios in the hundreds, often without awareness that metrics are far
outside their validity domain.
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
    DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    MAIN_METRICS, APX_METRICS,
)

# ── Configuration ──────────────────────────────────────────────────────────
DGP = "D1"
NULL_ENCODERS = ["E9", "E10"]  # E9 = random Gaussian, E10 = random Uniform
D_FIXED = 10
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Grid axes
MD_RATIOS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
MN_RATIOS = [0.01, 0.05, 0.10, 0.50, 1.00, 5.00]

# Derived: m = m/d * d, n = m / (m/n)
M_VALUES = [int(md * D_FIXED) for md in MD_RATIOS]  # [5, 10, 20, 50, 100, 200]

MIN_SAMPLES = 5  # skip cells with n < MIN_SAMPLES


# ── Experiment logic ───────────────────────────────────────────────────────

def compute_phase_diagram(null_encoder):
    """
    Compute metric scores for every (m/d, m/n) cell in the grid.

    Parameters
    ----------
    null_encoder : str
        Encoder key (e.g. ``"E9"`` or ``"E10"``).

    Returns
    -------
    score_grids : dict[str, np.ndarray]
        ``{metric: array of shape (len(MD_RATIOS), len(MN_RATIOS))}``
    """
    registry = make_registry()
    score_grids = {met: np.full((len(MD_RATIOS), len(MN_RATIOS)), np.nan)
                   for met in METRICS}

    for i, (md, m) in enumerate(zip(MD_RATIOS, M_VALUES)):
        for j, mn in enumerate(MN_RATIOS):
            n = int(round(m / mn))
            if n < MIN_SAMPLES:
                print(f"    m/d={md:.1f}, m/n={mn:.2f} → m={m}, n={n} — skipped (n < {MIN_SAMPLES})")
                continue
            print(f"    m/d={md:.1f}, m/n={mn:.2f} → m={m}, n={n}")

            def eval_one_seed(seed, _m=m, _n=n, _enc=null_encoder):
                return evaluate_dgp_encoder(
                    DGP, _enc,
                    n_samples=_n, n_factors=D_FIXED, seed=seed,
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


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_phase_diagram(score_grids, metrics=None, null_encoder="E9"):
    """One heatmap per metric: m/d vs m/n, colour = null encoder score."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    n_metrics = len(metrics)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    row_labels = [f"m/d={md:.1f}" for md in MD_RATIOS]
    col_labels = [f"m/n={mn:.2f}" for mn in MN_RATIOS]

    for idx, met in enumerate(metrics):
        ax = axes_flat[idx]
        data = score_grids[met]
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(display_name(met), fontsize=10)
        # Annotate cells
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                val = data[ii, jj]
                if np.isfinite(val):
                    txt = f"{val:.2f}"
                    color = "white" if val > 0.5 else "black"
                else:
                    txt = "—"
                    color = "grey"
                ax.text(jj, ii, txt, ha="center", va="center",
                        fontsize=6, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Phase diagram: metric reliability under null encoder ({null_encoder})\n"
        f"({DGP}, d={D_FIXED}) — "
        "green = trustworthy (≈0), red = inflated (broken)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_phase_diagram_single(met, score_grids, null_encoder="E9"):
    """Large single-metric heatmap for the paper figure."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    data = score_grids[met]
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    row_labels = [f"{md:.0f}" if md >= 1 else f"{md:.1f}" for md in MD_RATIOS]
    col_labels = [f"{mn:.2f}" for mn in MN_RATIOS]

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel(r"$m\,/\,n$", fontsize=12)
    ax.set_ylabel(r"$m\,/\,d$", fontsize=12)
    ax.set_title(f"{display_name(met)} under null encoder ({null_encoder})",
                 fontsize=13)

    # Annotate
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            val = data[ii, jj]
            if np.isfinite(val):
                txt = f"{val:.2f}"
                color = "white" if val > 0.5 else "black"
            else:
                txt = "—"
                color = "grey"
            ax.text(jj, ii, txt, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Null encoder score (should be 0)")
    fig.tight_layout()
    return fig


def plot_phase_diagram_main(score_grids, null_encoder="E9"):
    """Main-text 1×3 heatmaps: MCC-P, R², DCI-D with diverging colormap centered on 0."""
    focus = ["mcc_pearson", "r2", "dci_disentanglement"]
    focus = [m for m in focus if m in score_grids]
    setup_plot_style()
    n = len(focus)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    row_labels = [f"{md:.0f}" if md >= 1 else f"{md:.1f}" for md in MD_RATIOS]
    col_labels = [f"{mn:.2f}" for mn in MN_RATIOS]

    for ax, met in zip(axes, focus):
        data = score_grids[met]
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel(r"$m\,/\,n$", fontsize=11)
        ax.set_ylabel(r"$m\,/\,d$", fontsize=11)
        ax.set_title(display_name(met), fontsize=12)
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                val = data[ii, jj]
                if np.isfinite(val):
                    txt = f"{val:.2f}"
                    color = "white" if val > 0.5 else "black"
                else:
                    txt = "—"
                    color = "grey"
                ax.text(jj, ii, txt, ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Phase diagram: metric reliability under null encoder ({null_encoder})\n"
        "green = trustworthy (≈0), red = inflated",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 15 – Phase diagram: metric reliability vs m/d × m/n")
    print("=" * 70)

    for enc in NULL_ENCODERS:
        enc_lower = enc.lower()
        exp_key = f"exp15_{enc_lower}"
        print(f"\n  ── Null encoder: {enc} ──")

        if plot_only:
            try:
                data, config = load_results(exp_key)
                score_grids = data["grids"]
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping")
                continue
        else:
            score_grids = compute_phase_diagram(null_encoder=enc)
            save_results(exp_key, {"grids": score_grids}, config={
                "md_ratios": MD_RATIOS,
                "mn_ratios": MN_RATIOS,
                "m_values": M_VALUES,
                "d_fixed": D_FIXED,
                "dgp": DGP,
                "null_encoder": enc,
                "n_seeds": N_SEEDS,
            })

        # Grid figure (all metrics)
        tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags:
            for ext in ("pdf", "png"):
                fig = plot_phase_diagram(score_grids, metrics=mets, null_encoder=enc)
                savefig(fig, f"exp15_phase_diagram_{enc_lower}_{tag}.{ext}", subdir="exp15")

        # Main-text 1×3 phase diagram
        for ext in ("pdf", "png"):
            fig = plot_phase_diagram_main(score_grids, null_encoder=enc)
            savefig(fig, f"exp15_phase_diagram_{enc_lower}_main.{ext}", subdir="exp15")

        # Single-metric figures for key metrics
        for met in ["mcc_pearson", "mcc_spearman", "dci_disentanglement", "r2", "tmex"]:
            if met in METRICS:
                for ext in ("pdf", "png"):
                    fig = plot_phase_diagram_single(met, score_grids, null_encoder=enc)
                    savefig(fig, f"exp15_phase_{enc_lower}_{met}.{ext}", subdir="exp15")

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
