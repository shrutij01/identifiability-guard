"""
Experiment 8 – What is the effect of the type of encoding of the factor in the
codes?

Compare D × E1 (elementwise linear baseline) against D × E5 (overcomplete
linear), D × E6 (overcomplete multicodes), and D × E8 (overcomplete disjoint
nonlinear subsets).

For each DGP (D1–D4) we produce a grouped bar chart where bars correspond to
encoders and each group of bars is one metric.

Extended: sweep d ∈ {5, 20, 50} to see how metrics scale with dimensionality.
For d ∈ {20, 50}, only D1 is used (isolates encoder effect, avoids combinatorial
explosion).
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
    get_color,
    DEFAULT_N_SAMPLES, DEFAULT_N_FACTORS, DEFAULT_N_SEEDS, DEFAULT_BASE_SEED,
    MAIN_METRICS, APX_METRICS,
)

# ── Configuration ──────────────────────────────────────────────────────────
DGPS = ["D1", "D2", "D3", "D4"]
ENCODERS = ["E1", "E5", "E6", "E8"]
N_SAMPLES = DEFAULT_N_SAMPLES
N_FACTORS = DEFAULT_N_FACTORS
N_SEEDS = DEFAULT_N_SEEDS
BASE_SEED = DEFAULT_BASE_SEED
METRICS = sorted(APX_METRICS)
METRICS_MAIN = sorted(MAIN_METRICS)

# Multi-d configuration
D_VALUES = [5, 20, 50]
N_SAMPLES_BY_D = {5: 1000, 20: 1000, 50: 4000}
# At large d, only run D1 to isolate encoder effect
DGPS_BY_D = {5: DGPS, 20: ["D1"], 50: ["D1"]}

ENCODER_COLORS = {
    "E1": "#1f77b4",
    "E3": "#8c564b",
    "E5": "#ff7f0e",
    "E6": "#2ca02c",
    "E7": "#d62728",
    "E8": "#9467bd",
}

# Ratio sweep configuration
RATIOS = [1.5, 2.0, 3.0, 5.0, 10.0]
RATIO_DGP = "D1"


# ── Experiment logic ───────────────────────────────────────────────────────

def evaluate_all_combinations(dgps=None, n_samples=N_SAMPLES, n_factors=N_FACTORS):
    """
    Returns
    -------
    scores : dict[dgp][encoder][metric] → float (mean over seeds)
    """
    if dgps is None:
        dgps = DGPS
    registry = make_registry()
    scores = {}
    for dgp in dgps:
        scores[dgp] = {}
        for enc in ENCODERS:
            print(f"    {dgp} × {enc}")

            def eval_one_seed(seed, _dgp=dgp, _enc=enc):
                return evaluate_dgp_encoder(
                    _dgp, _enc,
                    n_samples=n_samples, n_factors=n_factors, seed=seed,
                    registry=registry,
                )

            _raw, agg = multi_seed_evaluate(eval_one_seed, n_seeds=N_SEEDS,
                                             base_seed=BASE_SEED)
            scores[dgp][enc] = {
                met: agg.get(met, {}).get("mean", np.nan) for met in METRICS
            }
    return scores


def plot_encoding_type_effect(scores, metrics=None, d=N_FACTORS):
    """One figure per DGP: grouped bar chart comparing encoders for each metric."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    figs = {}
    dgps = list(scores.keys())
    for dgp in dgps:
        n_metrics = len(metrics)
        n_enc = len(ENCODERS)
        width = 0.8 / n_enc
        x = np.arange(n_metrics)

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, enc in enumerate(ENCODERS):
            vals = [scores[dgp][enc].get(m, np.nan) for m in metrics]
            offset = (i - n_enc / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=enc,
                   color=ENCODER_COLORS.get(enc, "#888"))

        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in metrics],
                           rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Metric score")
        ax.set_title(f"Effect of encoding type – {dgp}, d={d}")
        ax.legend(title="Encoder")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        figs[dgp] = fig
    return figs


def plot_encoding_type_combined(scores, metrics=None, d=N_FACTORS):
    """All DGPs in a single multi-panel figure."""
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    dgps = list(scores.keys())
    n_dgps = len(dgps)
    fig, axes = plt.subplots(1, n_dgps, figsize=(6 * n_dgps, 5), sharey=True)
    if n_dgps == 1:
        axes = [axes]

    n_metrics = len(metrics)
    n_enc = len(ENCODERS)
    width = 0.8 / n_enc
    x = np.arange(n_metrics)

    for ax, dgp in zip(axes, dgps):
        for i, enc in enumerate(ENCODERS):
            vals = [scores[dgp][enc].get(m, np.nan) for m in metrics]
            offset = (i - n_enc / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=enc,
                   color=ENCODER_COLORS.get(enc, "#888"))
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in metrics],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(dgp)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ENCODERS),
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Effect of encoding type: E1 vs E5, E6, E8\n"
        f"(d={d}, n={N_SAMPLES_BY_D.get(d, N_SAMPLES)})", y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_encoding_type_vs_d(all_d_scores, metrics=None):
    """For each encoder, show metric scores vs d (log scale x-axis).

    Parameters
    ----------
    all_d_scores : dict[int, dict[str, dict[str, float]]]
        ``{d: {encoder: {metric: score}}}``  — only D1 scores used.
    """
    if metrics is None:
        metrics = METRICS
    setup_plot_style()
    d_vals = sorted(all_d_scores.keys())
    n_enc = len(ENCODERS)
    fig, axes = plt.subplots(1, n_enc, figsize=(5.5 * n_enc, 5), sharey=True)
    if n_enc == 1:
        axes = [axes]

    for ax, enc in zip(axes, ENCODERS):
        for m in metrics:
            vals = []
            for d in d_vals:
                # Use D1 scores at each d
                enc_scores = all_d_scores[d].get("D1", {}).get(enc, {})
                vals.append(enc_scores.get(m, np.nan))
            c = get_color(m)
            ax.plot(d_vals, vals, marker="o", color=c,
                    label=display_name(m), markersize=5)
        ax.set_xlabel("Number of factors (d)")
        ax.set_xscale("log")
        ax.set_xticks(d_vals)
        ax.set_xticklabels([str(d) for d in d_vals])
        ax.set_title(f"Encoder {enc}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Metric score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Encoding type effect vs dimensionality (D1 only)\n"
        f"d ∈ {{{', '.join(str(d) for d in d_vals)}}}", y=1.02,
    )
    fig.tight_layout()
    return fig


def evaluate_ratio_sweep():
    """Sweep m/d ratio for each encoder on D1, d=N_FACTORS."""
    registry = make_registry()
    d = N_FACTORS
    ref_encs = ["E1", "E3"]
    overcomplete_encs = ["E5", "E6", "E7", "E8"]
    results = {}

    # Reference encoders at m/d = 1.0
    for ref_enc in ref_encs:
        print(f"    {ref_enc} @ m/d=1.00 (reference)")

        def eval_ref(seed, _enc=ref_enc):
            return evaluate_dgp_encoder(
                RATIO_DGP, _enc, n_samples=N_SAMPLES, n_factors=d,
                seed=seed, registry=registry,
                metrics_to_compute=set(METRICS_MAIN),
            )

        _raw, agg = multi_seed_evaluate(eval_ref, n_seeds=N_SEEDS, base_seed=BASE_SEED)
        results[ref_enc] = {
            "ratios": [1.0],
            "means": {met: [agg.get(met, {}).get("mean", np.nan)] for met in METRICS_MAIN},
            "ci_lo": {met: [agg.get(met, {}).get("ci_lower", np.nan)] for met in METRICS_MAIN},
            "ci_hi": {met: [agg.get(met, {}).get("ci_upper", np.nan)] for met in METRICS_MAIN},
        }

    # Overcomplete encoders at each ratio
    for enc in overcomplete_encs:
        seen = {}
        ratios_actual = []
        means = {met: [] for met in METRICS_MAIN}
        ci_lo = {met: [] for met in METRICS_MAIN}
        ci_hi = {met: [] for met in METRICS_MAIN}

        for ratio in RATIOS:
            if enc == "E8":
                cpf = max(2, round(ratio))
                actual_ratio = float(cpf)
                enc_kwargs = {"codes_per_factor": cpf}
            else:
                m = int(round(ratio * d))
                actual_ratio = m / d
                enc_kwargs = {"m": m}

            if actual_ratio in seen:
                continue
            seen[actual_ratio] = True
            ratios_actual.append(actual_ratio)

            print(f"    {enc} @ m/d={actual_ratio:.2f}")

            def eval_one(seed, _enc=enc, _kwargs=enc_kwargs):
                return evaluate_dgp_encoder(
                    RATIO_DGP, _enc, n_samples=N_SAMPLES, n_factors=d,
                    seed=seed, encoder_kwargs=_kwargs, registry=registry,
                    metrics_to_compute=set(METRICS_MAIN),
                )

            _raw, agg = multi_seed_evaluate(eval_one, n_seeds=N_SEEDS, base_seed=BASE_SEED)
            for met in METRICS_MAIN:
                means[met].append(agg.get(met, {}).get("mean", np.nan))
                ci_lo[met].append(agg.get(met, {}).get("ci_lower", np.nan))
                ci_hi[met].append(agg.get(met, {}).get("ci_upper", np.nan))

        results[enc] = {
            "ratios": ratios_actual,
            "means": means,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

    return results


def plot_ratio_sweep(sweep_results):
    """1x4 panels: metric score vs m/d ratio, one line per encoder."""
    setup_plot_style()
    metrics = METRICS_MAIN
    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(5 * n_met, 4.5), sharey=True)
    if n_met == 1:
        axes = [axes]

    max_ratio = max(
        r for enc_data in sweep_results.values() for r in enc_data["ratios"]
    )

    all_encs = list(sweep_results.keys())

    for ax, met in zip(axes, metrics):
        for enc in all_encs:
            enc_data = sweep_results[enc]
            ratios = enc_data["ratios"]
            means = [enc_data["means"][met][i] for i in range(len(ratios))]
            lo = [enc_data["ci_lo"][met][i] for i in range(len(ratios))]
            hi = [enc_data["ci_hi"][met][i] for i in range(len(ratios))]
            color = ENCODER_COLORS.get(enc, "#888")
            ax.plot(ratios, means, marker="o", color=color, label=enc, markersize=5)
            ax.fill_between(ratios, lo, hi, color=color, alpha=0.15)

        ax.axvline(1.0, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("m / d ratio")
        ax.set_title(display_name(met))
        ax.set_xlim(0.8, max_ratio + 0.2)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Metric score")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Metric score vs m/d ratio by encoder type ({RATIO_DGP}, d={N_FACTORS})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main(plot_only=False, quick=False):
    from results_io import save_results, load_results

    print("=" * 70)
    print("Experiment 8 – Encoding type effect (E1 vs E5, E6, E8)")
    print("=" * 70)

    # ── d=5 sweep (original, backward-compatible) ─────────────────────────
    if plot_only:
        data, config = load_results("exp08")
        scores = data["scores"]
    else:
        scores = evaluate_all_combinations()
        save_results("exp08", {"scores": scores}, config={
            "dgps": DGPS,
            "encoders": ENCODERS,
            "n_samples": N_SAMPLES,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
    for tag, mets in tags:
        for ext in ["pdf", "png"]:
            figs = plot_encoding_type_effect(scores, metrics=mets)
            for dgp, fig in figs.items():
                savefig(fig, f"exp08_encoding_type_{dgp}_{tag}.{ext}",
                        subdir="exp08")

            fig_combined = plot_encoding_type_combined(scores, metrics=mets)
            savefig(fig_combined, f"exp08_encoding_type_combined_{tag}.{ext}",
                    subdir="exp08")

    # ── Multi-d sweeps ────────────────────────────────────────────────────
    # Collect D1 scores for all d values (for vs-d plot)
    all_d_scores = {N_FACTORS: scores}

    for d in D_VALUES:
        if d == N_FACTORS:
            continue  # already done
        n_samp = N_SAMPLES_BY_D.get(d, N_SAMPLES)
        dgps_d = DGPS_BY_D.get(d, ["D1"])
        exp_key = f"exp08_d{d}"

        if plot_only:
            try:
                d_data, d_config = load_results(exp_key)
                d_scores = d_data["scores"]
                all_d_scores[d] = d_scores
            except FileNotFoundError:
                print(f"  No saved results for {exp_key}, skipping d={d} plots")
                continue
        else:
            print(f"\n  ── d={d} (n={n_samp}, DGPs={dgps_d}) ──")
            d_scores = evaluate_all_combinations(
                dgps=dgps_d, n_samples=n_samp, n_factors=d,
            )
            save_results(exp_key, {"scores": d_scores}, config={
                "dgps": dgps_d,
                "encoders": ENCODERS,
                "n_samples": n_samp,
                "n_factors": d,
                "n_seeds": N_SEEDS,
            })
            all_d_scores[d] = d_scores

        # Per-d bar chart plots
        tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags:
            for ext in ["pdf", "png"]:
                figs = plot_encoding_type_effect(d_scores, metrics=mets, d=d)
                for dgp, fig in figs.items():
                    savefig(fig, f"exp08_encoding_type_{dgp}_d{d}_{tag}.{ext}",
                            subdir="exp08")
                fig_combined = plot_encoding_type_combined(d_scores, metrics=mets, d=d)
                savefig(fig_combined, f"exp08_encoding_type_combined_d{d}_{tag}.{ext}",
                        subdir="exp08")

    # ── vs-d plot ─────────────────────────────────────────────────────────
    if len(all_d_scores) > 1:
        tags = [("main", METRICS_MAIN)] if quick else [("main", METRICS_MAIN), ("apx", METRICS)]
        for tag, mets in tags:
            for ext in ["pdf", "png"]:
                fig = plot_encoding_type_vs_d(all_d_scores, metrics=mets)
                savefig(fig, f"exp08_encoding_type_vs_d_{tag}.{ext}",
                        subdir="exp08")

    # ── Ratio sweep ──────────────────────────────────────────────────────
    exp_ratio_key = "exp08_ratio"
    ratio_results = None
    if plot_only:
        try:
            ratio_data, _ = load_results(exp_ratio_key)
            ratio_results = ratio_data["sweep"]
        except FileNotFoundError:
            print(f"  No saved results for {exp_ratio_key}, skipping ratio sweep plots")
    else:
        print("\n  ── Ratio sweep (D1, d=5) ──")
        ratio_results = evaluate_ratio_sweep()
        save_results(exp_ratio_key, {"sweep": ratio_results}, config={
            "ratios": RATIOS,
            "dgp": RATIO_DGP,
            "encoders": ENCODERS,
            "n_factors": N_FACTORS,
            "n_seeds": N_SEEDS,
        })

    if ratio_results is not None:
        for ext in ["pdf", "png"]:
            fig = plot_ratio_sweep(ratio_results)
            savefig(fig, f"exp08_ratio_sweep_main.{ext}", subdir="exp08")

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
