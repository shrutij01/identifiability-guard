"""Re-generate plots from saved results for all (or specific) experiments.

Usage
-----
    python experiments/plot_all.py                # all experiments
    python experiments/plot_all.py exp01 exp04    # specific experiments
    python experiments/plot_all.py --list         # show available experiments
"""

import sys
import importlib

EXPERIMENTS = [f"exp{i:02d}" for i in range(1, 15)]

MODULE_MAP = {
    "exp01": "exp01_invariance_across_dgps",
    "exp02": "exp02_nonlinearity_sensitivity",
    "exp03": "exp03_correlation_sign",
    "exp04": "exp04_correlation_vs_entanglement",
    "exp05": "exp05_predictability_vs_disentanglement",
    "exp06": "exp06_dropped_variables",
    "exp07": "exp07_redundancy_vs_compression",
    "exp08": "exp08_encoding_type",
    "exp09": "exp09_overcomplete_representations",
    "exp10": "exp10_sample_sensitivity",
    "exp11": "exp11_metric_inflation",
    "exp12": "exp12_ratio_replot",
    "exp13": "exp13_ratio_collapse",
    "exp14": "exp14_disentangle_ratios",
}


def main():
    if "--list" in sys.argv:
        for exp, mod in MODULE_MAP.items():
            print(f"  {exp}  ->  {mod}.py")
        return

    targets = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not targets:
        targets = EXPERIMENTS

    for exp in targets:
        if exp not in MODULE_MAP:
            print(f"Unknown experiment: {exp}")
            continue
        mod = importlib.import_module(MODULE_MAP[exp])
        print(f"\n{'=' * 70}")
        print(f"Plotting {exp} from saved results")
        print(f"{'=' * 70}")
        try:
            mod.main(plot_only=True)
        except FileNotFoundError as e:
            print(f"  Skipped ({e})")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
