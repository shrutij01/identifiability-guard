"""
Experiment 15 -- Phase diagram: where do metrics break?

Heatmap of metric reliability under null encoders (output independent of
input -- all scores should be 0) as a function of m/d and m/n.

Fixed-m design
--------------
The representation width m is held FIXED at ``M_FIXED``; the two ratio
axes are realised by varying d and n independently:

  * rows:    m/d  →  d = m / (m/d)   (structural ratio; varies only d)
  * columns: m/n  →  n = m / (m/n)   (estimation ratio; varies only n)

This decoupling is essential.  The earlier design fixed d and derived
both m and n from the ratios, so every direction in the grid changed the
estimation budget and the two axes were confounded.  (That run was
additionally affected by a bug where the ``m`` kwarg was silently
dropped for E9/E10, pinning m = d in every cell.)

Theory (see paper appendix): under the null, E[MCC-P] ≈ sqrt(2 log m / n)
-- governed by n (strongly) and m (logarithmically), independent of d.
Predicted pattern: scores rise along rows (n shrinks), stay flat along
columns (only d changes).

Collapse sweep
--------------
The fixed-m grid cannot show how the score depends on m itself.  A
second sweep varies m ∈ COLLAPSE_M and n ∈ COLLAPSE_N at fixed d:
plotting the null MCC against sqrt(2 log m / n_eff) collapses all m onto
one curve, while plotting against m/n does not.  (n_eff = test-split
size, since the pipeline scores pure-statistic metrics on the 20% test
split.)

Practical relevance: sparse autoencoders in mech-interp operate at m/n
ratios in the hundreds, often without awareness that metrics are far
outside their validity domain.

  * Green (score ≈ 0): metric is trustworthy at this operating point.
  * Red (score > 0): false positive -- metric reports identifiability
    where none exists.
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
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
N_JOBS = 5  # parallel seed workers
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Phase-diagram grid: m fixed, rows vary d, columns vary n
M_FIXED = 50
MD_RATIOS = [0.5, 1.0, 2.0, 5.0, 10.0]
MN_RATIOS = [0.01, 0.05, 0.10, 0.50, 1.00]
D_VALUES = [int(round(M_FIXED / md)) for md in MD_RATIOS]  # [100, 50, 25, 10, 5]
N_VALUES = [int(round(M_FIXED / mn)) for mn in MN_RATIOS]  # [5000, 1000, 500, 100, 50]

# Collapse sweep: d fixed, m and n vary
COLLAPSE_D = 10
COLLAPSE_M = [10, 50, 200]
COLLAPSE_N = [20, 50, 100, 200, 500, 1000, 2000, 5000]
COLLAPSE_METRICS = {"mcc_pearson", "mcc_spearman"}


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

    for i, (md, d) in enumerate(zip(MD_RATIOS, D_VALUES)):
        for j, (mn, n) in enumerate(zip(MN_RATIOS, N_VALUES)):
            print(f"    m/d={md:.1f}, m/n={mn:.2f} → m={M_FIXED}, d={d}, n={n}")

            def eval_one_seed(seed, _d=d, _n=n, _enc=null_encoder):
                return evaluate_dgp_encoder(
                    DGP, _enc,
                    n_samples=_n, n_factors=_d, seed=seed,
                    encoder_kwargs={"m": M_FIXED},
                    metrics_to_compute=set(METRICS),
                    registry=registry,
                )

            try:
                _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                                base_seed=BASE_SEED, n_jobs=N_JOBS)
                for met in METRICS:
                    score_grids[met][i, j] = agg.get(met, {}).get("mean", np.nan)
            except Exception as e:
                print(f"      ⚠ failed: {e}")

    return score_grids


def compute_collapse_sweep(null_encoder):
    """
    Null MCC over an (m, n) grid at fixed d — data for the collapse plot.

    Returns
    -------
    grids : dict[str, np.ndarray]
        ``{f"{metric}_mean"/f"{metric}_std": array of shape
        (len(COLLAPSE_M), len(COLLAPSE_N))}``
    """
    registry = make_registry()
    grids = {}
    for met in sorted(COLLAPSE_METRICS):
        grids[f"{met}_mean"] = np.full((len(COLLAPSE_M), len(COLLAPSE_N)), np.nan)
        grids[f"{met}_std"] = np.full((len(COLLAPSE_M), len(COLLAPSE_N)), np.nan)

    for i, m in enumerate(COLLAPSE_M):
        for j, n in enumerate(COLLAPSE_N):
            print(f"    collapse: m={m}, n={n} (d={COLLAPSE_D})")

            def eval_one_seed(seed, _m=m, _n=n, _enc=null_encoder):
                return evaluate_dgp_encoder(
                    DGP, _enc,
                    n_samples=_n, n_factors=COLLAPSE_D, seed=seed,
                    encoder_kwargs={"m": _m},
                    metrics_to_compute=set(COLLAPSE_METRICS),
                    registry=registry,
                )

            try:
                _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                                base_seed=BASE_SEED, n_jobs=N_JOBS)
                for met in sorted(COLLAPSE_METRICS):
                    grids[f"{met}_mean"][i, j] = agg.get(met, {}).get("mean", np.nan)
                    grids[f"{met}_std"][i, j] = agg.get(met, {}).get("std", np.nan)
            except Exception as e:
                print(f"      ⚠ failed: {e}")

    return grids


# ── Plotting ──────────────────────────────────────────────────────────────

ROW_LABELS = [f"m/d={md:g} (d={d})" for md, d in zip(MD_RATIOS, D_VALUES)]
COL_LABELS = [f"m/n={mn:g} (n={n})" for mn, n in zip(MN_RATIOS, N_VALUES)]


def _annotate_cells(ax, data, fontsize=6, fontweight=None):
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
                    fontsize=fontsize, color=color, fontweight=fontweight)


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

    for idx, met in enumerate(metrics):
        ax = axes_flat[idx]
        data = score_grids[met]
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(COL_LABELS)))
        ax.set_xticklabels(COL_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ROW_LABELS)))
        ax.set_yticklabels(ROW_LABELS, fontsize=7)
        ax.set_title(display_name(met), fontsize=10)
        _annotate_cells(ax, data)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Phase diagram: metric reliability under null encoder ({null_encoder})\n"
        f"({DGP}, m={M_FIXED} fixed; rows vary d, columns vary n) — "
        "green = trustworthy (≈0), red = inflated (broken)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_phase_diagram_single(met, score_grids, null_encoder="E9"):
    """Large single-metric heatmap for inspection."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    data = score_grids[met]
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=9)
    ax.set_xlabel(r"$m\,/\,n$", fontsize=12)
    ax.set_ylabel(r"$m\,/\,d$", fontsize=12)
    ax.set_title(f"{display_name(met)} under null encoder ({null_encoder}), "
                 f"m={M_FIXED} fixed", fontsize=13)
    _annotate_cells(ax, data, fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Null encoder score (should be 0)")
    fig.tight_layout()
    return fig


def plot_phase_diagram_main(score_grids, null_encoder="E9"):
    """1×3 heatmaps for quick inspection: MCC-P, R², DCI-D."""
    focus = ["mcc_pearson", "r2", "dci_disentanglement"]
    focus = [m for m in focus if m in score_grids]
    setup_plot_style()
    n = len(focus)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, met in zip(axes, focus):
        data = score_grids[met]
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(COL_LABELS)))
        ax.set_xticklabels(COL_LABELS, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(ROW_LABELS)))
        ax.set_yticklabels(ROW_LABELS, fontsize=9)
        ax.set_xlabel(r"$m\,/\,n$", fontsize=11)
        ax.set_ylabel(r"$m\,/\,d$", fontsize=11)
        ax.set_title(display_name(met), fontsize=12)
        _annotate_cells(ax, data, fontsize=7, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Phase diagram: metric reliability under null encoder ({null_encoder})\n"
        f"m={M_FIXED} fixed; rows vary d, columns vary n — "
        "green = trustworthy (≈0), red = inflated",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_collapse(collapse_grids, null_encoder="E10"):
    """Diagnostic collapse plot: null MCC-P vs m/n and vs sqrt(2 log m / n_eff)."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    obs = collapse_grids["mcc_pearson_mean"]
    n_arr = np.array(COLLAPSE_N, dtype=float)
    n_eff = n_arr - (0.8 * n_arr).astype(int)

    for i, m in enumerate(COLLAPSE_M):
        axes[0].plot(m / n_arr, obs[i], marker="o", label=f"m={m}")
        bound = np.sqrt(2 * np.log(m) / n_eff)
        axes[1].plot(bound, obs[i], marker="o", label=f"m={m}")

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$m/n$")
    axes[0].set_title("vs. m/n — no collapse")
    xs = np.linspace(0, 1.7, 100)
    axes[1].plot(xs, np.minimum(xs, 1.0), ls="--", color="grey",
                 label=r"bound $\min(x, 1)$")
    axes[1].set_xlabel(r"$\sqrt{2 \log m / n_{\rm eff}}$")
    axes[1].set_title("vs. extreme-value bound — collapse")
    for ax in axes:
        ax.set_ylabel("Null MCC-P")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    fig.suptitle(f"Null MCC-P scaling ({null_encoder}, d={COLLAPSE_D})", y=1.0)
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 15 – Phase diagram: metric reliability vs m/d × m/n")
    print(f"  (m={M_FIXED} fixed; rows vary d={D_VALUES}, cols vary n={N_VALUES})")
    print("=" * 70)

    for enc in NULL_ENCODERS:
        enc_lower = enc.lower()
        exp_key = f"exp15_{enc_lower}"
        collapse_key = f"exp15_collapse_{enc_lower}"
        print(f"\n  ── Null encoder: {enc} ──")

        if plot_only:
            try:
                data, config = load_results(exp_key)
                score_grids = data["grids"]
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping")
                continue
            try:
                cdata, _ = load_results(collapse_key)
                collapse_grids = cdata["grids"]
            except FileNotFoundError:
                collapse_grids = None
        else:
            score_grids = compute_phase_diagram(null_encoder=enc)
            save_results(exp_key, {"grids": score_grids}, config={
                "m_fixed": M_FIXED,
                "md_ratios": MD_RATIOS,
                "mn_ratios": MN_RATIOS,
                "d_values": D_VALUES,
                "n_values": N_VALUES,
                "dgp": DGP,
                "null_encoder": enc,
                "n_seeds": N_SEEDS,
            })
            print(f"\n  ── Collapse sweep: {enc} ──")
            collapse_grids = compute_collapse_sweep(null_encoder=enc)
            save_results(collapse_key, {"grids": collapse_grids}, config={
                "d_fixed": COLLAPSE_D,
                "m_values": COLLAPSE_M,
                "n_values": COLLAPSE_N,
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

        # 1×3 phase diagram
        for ext in ("pdf", "png"):
            fig = plot_phase_diagram_main(score_grids, null_encoder=enc)
            savefig(fig, f"exp15_phase_diagram_{enc_lower}_main.{ext}", subdir="exp15")

        # Single-metric figures for key metrics
        for met in ["mcc_pearson", "mcc_spearman", "dci_disentanglement", "r2", "tmex"]:
            if met in METRICS:
                for ext in ("pdf", "png"):
                    fig = plot_phase_diagram_single(met, score_grids, null_encoder=enc)
                    savefig(fig, f"exp15_phase_{enc_lower}_{met}.{ext}", subdir="exp15")

        # Collapse diagnostic
        if collapse_grids is not None:
            for ext in ("pdf", "png"):
                fig = plot_collapse(collapse_grids, null_encoder=enc)
                savefig(fig, f"exp15_collapse_{enc_lower}.{ext}", subdir="exp15")

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
