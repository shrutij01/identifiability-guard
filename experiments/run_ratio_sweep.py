"""Run only the ratio sweep for exp08 and exp09."""
from results_io import save_results
import exp08_encoding_type as e8
import exp09_overcomplete_representations as e9

print("=== exp08 ratio sweep ===")
r8 = e8.evaluate_ratio_sweep()
save_results("exp08_ratio", {"sweep": r8}, config={
    "ratios": e8.RATIOS, "dgp": e8.RATIO_DGP,
    "encoders": e8.ENCODERS, "n_factors": e8.N_FACTORS, "n_seeds": e8.N_SEEDS,
})
for ext in ["pdf", "png"]:
    e8.savefig(e8.plot_ratio_sweep(r8), f"exp08_ratio_sweep_main.{ext}", subdir="exp08")

print("=== exp09 ratio sweep ===")
r9 = e9.evaluate_ratio_sweep()
save_results("exp09_ratio", {"sweep": r9}, config={
    "ratios": e9.RATIOS, "dgp": e9.RATIO_DGP,
    "encoders": e9.ENCODERS, "n_factors": e9.N_FACTORS, "n_seeds": e9.N_SEEDS,
})
for ext in ["pdf", "png"]:
    e9.savefig(e9.plot_ratio_sweep(r9), f"exp09_ratio_sweep_main.{ext}", subdir="exp09")

print("Done.")
