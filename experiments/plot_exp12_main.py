"""Quick script: re-plot exp12 ratio_6a with main metrics only."""
import numpy as np
from results_io import load_results
from utils import plot_metrics_vs_xaxis_with_ci, savefig

data, config = load_results("exp12")
d_total = config["d_total"]
M_MAX, M_MIN = 3.0, 0.3
m_lo = max(1, int(round(M_MIN * d_total)))
m_values = list(range(int(M_MAX * d_total), m_lo - 1, -1))
ratios = [m / d_total for m in m_values]
focus = ["r2", "dci_disentanglement", "mcc_pearson", "tmex"]

for dgp in config["dgps_6a"]:
    d = data["6a"][dgp]
    for ext in ("pdf", "png"):
        fig = plot_metrics_vs_xaxis_with_ci(
            ratios, d["means"], d["ci_lo"], d["ci_hi"],
            xlabel=r"$m\,/\,d$",
            title=f"Metric score vs m/d ({dgp}, vary m, d=10)",
            metrics_to_plot=focus,
        )
        fig.axes[0].axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.5)
        savefig(fig, f"exp12_ratio_6a_{dgp}_main.{ext}", subdir="exp12")
    print(f"  saved {dgp}")
