"""
Metric Validity Phase Diagram
Generates a publication-quality PDF for direct inclusion in LaTeX.
Usage: python plot_phase_diagram.py → outputs phase_diagram.pdf
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# ─── Style ───
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "STIX Two Text",
            "STIXGeneral",
            "CMU Serif",
            "Computer Modern Roman",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.linewidth": 0.6,
    }
)

# ─── Colors ───
GOOD = "#b8d8e8"
GOOD_D = "#0b4f6c"
MILD = "#fef3c7"
MILD_D = "#78650d"
MODERATE = "#fdcfa1"
MODERATE_D = "#8a4000"
SEVERE = "#dcb4ca"
SEVERE_D = "#6b1d3a"

CRL = "#33127A"
BEYOND = "#DE70A1"
TEAL = "#008080"
AMETHYST = "#9966CC"
BURGUNDY = "#800021"

BG = "#fdfcf9"
BORDER = "#2b2b2b"
BORDER_L = "#aaaaaa"
MUTED = "#555555"
PASS_C = "#1a6e3a"
FAIL_C = "#a02020"
APX_C = "#2a8a7a"  # muted teal-green for appendix metrics

# ─── Layout ───
COLS, ROWS = 3, 3
CELL_W, CELL_H = 2.8, 1.85
STRIP_W = 2.6
STRIP_GAP = 0.6

fig_w = COLS * CELL_W + STRIP_GAP + STRIP_W + 2.2
fig_h = ROWS * CELL_H + 2.0

fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-2.0, COLS * CELL_W + STRIP_GAP + STRIP_W + 0.4)
ax.set_ylim(-0.7, ROWS * CELL_H + 1.5)
ax.set_aspect("equal")
ax.axis("off")

# ─── CRL / Beyond CRL background shading ───
# CRL wash: row 1 (Matched) × cols 0–1 (Independent, Correlated)
# In grid coords: Matched is visual row 1 → y from CELL_H to 2*CELL_H
crl_wash = mpatches.FancyBboxPatch(
    (0, CELL_H),
    2 * CELL_W,
    CELL_H,
    boxstyle="round,pad=0.02",
    facecolor=CRL,
    edgecolor="none",
    alpha=0.06,
    zorder=0,
)
ax.add_patch(crl_wash)

# Beyond wash: Overcomplete (all cols) + Undercomplete (all cols) + Matched × col 2
# Overcomplete row: y from 2*CELL_H to 3*CELL_H
beyond_over = mpatches.FancyBboxPatch(
    (0, 2 * CELL_H),
    COLS * CELL_W,
    CELL_H,
    boxstyle="round,pad=0.02",
    facecolor=BEYOND,
    edgecolor="none",
    alpha=0.07,
    zorder=0,
)
ax.add_patch(beyond_over)

# Undercomplete row: y from 0 to CELL_H
beyond_under = mpatches.FancyBboxPatch(
    (0, 0),
    COLS * CELL_W,
    CELL_H,
    boxstyle="round,pad=0.02",
    facecolor=BEYOND,
    edgecolor="none",
    alpha=0.07,
    zorder=0,
)
ax.add_patch(beyond_under)

# Matched × Functionally constrained (col 2): y from CELL_H to 2*CELL_H
beyond_matched_fc = mpatches.FancyBboxPatch(
    (2 * CELL_W, CELL_H),
    CELL_W,
    CELL_H,
    boxstyle="round,pad=0.02",
    facecolor=BEYOND,
    edgecolor="none",
    alpha=0.07,
    zorder=0,
)
ax.add_patch(beyond_matched_fc)

# CRL / Beyond labels tucked into corners
ax.text(
    0.12,
    2 * CELL_H - 0.10,
    "Within CRL",
    ha="left",
    va="top",
    fontsize=6,
    fontweight="bold",
    color=CRL,
    alpha=0.7,
    zorder=1,
)
ax.text(
    COLS * CELL_W - 0.12,
    3 * CELL_H - 0.10,
    "Beyond CRL",
    ha="right",
    va="top",
    fontsize=6,
    fontweight="bold",
    color=BEYOND,
    alpha=0.7,
    zorder=1,
)

# ─── Cell data (two-tier: main_pass, main_fail, apx_pass) ───
cell_data = [
    # Row 0: Overcomplete (m > d)
    [
        {
            "bg": MILD,
            "bd": MILD_D,
            "title": "FN: distributed codes",
            "main_pass": "MCC-S",
            "main_fail": "MCC-P, DCI-D, R\u00b2",
            "apx_pass": "",
        },
        {
            "bg": MODERATE,
            "bd": MODERATE_D,
            "title": "FN: correlation +\ndistributed codes",
            "main_pass": "",
            "main_fail": "MCC-P, MCC-S, DCI-D, R\u00b2",
            "apx_pass": "",
        },
        {
            "bg": SEVERE,
            "bd": SEVERE_D,
            "title": "FN: redundancy +\ndistribution compound",
            "main_pass": "",
            "main_fail": "All coordinate-wise",
            "apx_pass": "",
        },
    ],
    # Row 1: Matched (m = d)
    [
        {
            "bg": GOOD,
            "bd": GOOD_D,
            "title": "Metrics calibrated",
            "main_pass": "MCC-P, MCC-S, R\u00b2, DCI-D",
            "main_fail": "",
            "apx_pass": "T-MEX, InfoM",
        },
        {
            "bg": MILD,
            "bd": MILD_D,
            "title": "FN: correlation\nconfound",
            "main_pass": "MCC-P, MCC-S, R\u00b2",
            "main_fail": "DCI-D",
            "apx_pass": "T-MEX",
        },
        {
            "bg": MODERATE,
            "bd": MODERATE_D,
            "title": "FN: predictability\nconflation",
            "main_pass": "MCC-P, MCC-S",
            "main_fail": "DCI-D, R\u00b2",
            "apx_pass": "",
        },
    ],
    # Row 2: Undercomplete (m < d)
    [
        {
            "bg": MILD,
            "bd": MILD_D,
            "title": "FN: partial recovery\npenalised",
            "main_pass": "MCC-P, MCC-S, R\u00b2",
            "main_fail": "DCI-D",
            "apx_pass": "T-MEX",
        },
        {
            "bg": MODERATE,
            "bd": MODERATE_D,
            "title": "FN: partial recovery +\ncorrelation confound",
            "main_pass": "MCC-S",
            "main_fail": "MCC-P, DCI-D, R\u00b2",
            "apx_pass": "",
        },
        {
            "bg": SEVERE,
            "bd": SEVERE_D,
            "title": "FN: lossless vs lossy\nindistinguishable",
            "main_pass": "",
            "main_fail": "All metrics",
            "apx_pass": "",
        },
    ],
]

col_labels = ["Independent", "Correlated", "Functionally\nconstrained"]
col_sub = ["(D1)", "(D2)", "(D3 / D4)"]
col_sub_colors = [CRL, CRL, BEYOND]
row_labels = ["Overcomplete", "Matched", "Undercomplete"]
row_sub = ["$m > d$", "$m = d$", "$m < d$"]
row_enc = ["(E5\u2013E8)", "(E1\u2013E3)", "(E4)"]
row_enc_colors = [BURGUNDY, TEAL, AMETHYST]


def draw_cell(ax, row, col, data):
    """Draw one cell with two-tier metric display."""
    x = col * CELL_W
    y = (ROWS - 1 - row) * CELL_H
    pad = 0.06

    rect = FancyBboxPatch(
        (x + pad, y + pad),
        CELL_W - 2 * pad,
        CELL_H - 2 * pad,
        boxstyle="round,pad=0.04",
        facecolor=data["bg"],
        edgecolor=BORDER_L,
        linewidth=0.7,
        zorder=2,
    )
    ax.add_patch(rect)

    cx = x + CELL_W / 2
    cy = y + CELL_H / 2

    # Title
    title = data["title"]
    n_title_lines = title.count("\n") + 1
    title_y = cy + 0.28 if n_title_lines == 1 else cy + 0.32
    ax.text(
        cx,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=data["bd"],
        linespacing=1.3,
        zorder=3,
    )

    # Build metric lines dynamically (no empty gaps)
    lines = []
    if data["main_pass"]:
        lines.append((data["main_pass"], PASS_C, 6.5))
    if data["main_fail"]:
        lines.append((data["main_fail"], FAIL_C, 6.5))
    if data["apx_pass"]:
        lines.append(("+ " + data["apx_pass"], APX_C, 5.5))

    # Position lines below title, packed tightly
    start_y = cy - 0.22
    spacing = 0.22
    for i, (text, color, fsize) in enumerate(lines):
        ax.text(
            cx,
            start_y - i * spacing,
            text,
            ha="center",
            va="center",
            fontsize=fsize,
            color=color,
            style="italic",
            zorder=3,
        )


# ─── Draw all cells ───
for ri in range(ROWS):
    for ci in range(COLS):
        draw_cell(ax, ri, ci, cell_data[ri][ci])

# ─── Outer border ───
outer = mpatches.FancyBboxPatch(
    (0, 0),
    COLS * CELL_W,
    ROWS * CELL_H,
    boxstyle="round,pad=0.02",
    facecolor="none",
    edgecolor=BORDER,
    linewidth=1.2,
    zorder=4,
)
ax.add_patch(outer)

# ─── Column headers ───
for ci in range(COLS):
    cx = ci * CELL_W + CELL_W / 2
    ax.text(
        cx,
        ROWS * CELL_H + 0.65,
        col_labels[ci],
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=BORDER,
        linespacing=1.15,
    )
    ax.text(
        cx,
        ROWS * CELL_H + 0.25,
        col_sub[ci],
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=col_sub_colors[ci],
    )

# ─── Row headers ───
for ri in range(ROWS):
    cy = (ROWS - 1 - ri) * CELL_H + CELL_H / 2
    ax.text(
        -0.2,
        cy + 0.22,
        row_labels[ri],
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=BORDER,
    )
    ax.text(
        -0.2,
        cy - 0.05,
        row_sub[ri],
        ha="right",
        va="center",
        fontsize=7.5,
        color=MUTED,
    )
    ax.text(
        -0.2,
        cy - 0.28,
        row_enc[ri],
        ha="right",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=row_enc_colors[ri],
    )

# ─── Axis labels ───
ax.text(
    COLS * CELL_W / 2,
    ROWS * CELL_H + 1.2,
    r"DGP COMPLEXITY $\longrightarrow$",
    ha="center",
    va="center",
    fontsize=10.5,
    fontweight="bold",
    color=BORDER,
)

ax.text(
    -1.7,
    ROWS * CELL_H / 2,
    r"$\longleftarrow$ DIMENSION MISMATCH $\longrightarrow$",
    ha="center",
    va="center",
    fontsize=10.5,
    fontweight="bold",
    color=BORDER,
    rotation=90,
)

# ─── Statistical validity strip (right side) ───
strip_x = COLS * CELL_W + STRIP_GAP
grid_h = ROWS * CELL_H  # total grid height

# Three zones stacked bottom-to-top
STRIP_GOOD = "#b8d8e8"    # blue
STRIP_MODERATE = "#fdcfa1"  # orange
STRIP_SEVERE = "#dcb4ca"    # purple-pink
STRIP_GOOD_D = "#0b4f6c"
STRIP_MOD_D = "#8a4000"
STRIP_SEV_D = "#6b1d3a"

zones = [
    {
        "frac": 0.33,
        "bg": STRIP_GOOD,
        "bd": STRIP_GOOD_D,
        "ratio": "m/n < 0.1",
        "body": "Metrics calibrated",
        "detail": "Low FP risk",
    },
    {
        "frac": 0.34,
        "bg": STRIP_MODERATE,
        "bd": STRIP_MOD_D,
        "ratio": r"0.1 $\leq$ m/n < 1",
        "body": "MCC-P, MCC-S inflate",
        "detail": "DCI-D mild FP",
    },
    {
        "frac": 0.33,
        "bg": STRIP_SEVERE,
        "bd": STRIP_SEV_D,
        "ratio": r"m/n $\geq$ 1",
        "body": "Most metrics report\nnoise as signal",
        "detail": "Sparse autoencoders\noperate here",
    },
]

y_cursor = 0.0
for zone in zones:
    zh = zone["frac"] * grid_h
    rect = FancyBboxPatch(
        (strip_x + 0.04, y_cursor + 0.04),
        STRIP_W - 0.08,
        zh - 0.08,
        boxstyle="round,pad=0.04",
        facecolor=zone["bg"],
        edgecolor=BORDER_L,
        linewidth=0.7,
        zorder=2,
    )
    ax.add_patch(rect)

    zcx = strip_x + STRIP_W / 2
    zcy = y_cursor + zh / 2

    # Ratio label (bold)
    ax.text(
        zcx,
        zcy + 0.35,
        zone["ratio"],
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=zone["bd"],
    )
    # Body text
    ax.text(
        zcx,
        zcy + 0.02,
        zone["body"],
        ha="center",
        va="center",
        fontsize=6.5,
        color=BORDER,
        linespacing=1.2,
    )
    # Detail text (italic, muted)
    ax.text(
        zcx,
        zcy - 0.32,
        zone["detail"],
        ha="center",
        va="center",
        fontsize=5.5,
        color=MUTED,
        style="italic",
        linespacing=1.2,
    )

    y_cursor += zh

# Strip outer border
strip_border = mpatches.FancyBboxPatch(
    (strip_x, 0),
    STRIP_W,
    grid_h,
    boxstyle="round,pad=0.02",
    facecolor="none",
    edgecolor=BORDER,
    linewidth=1.0,
    zorder=4,
)
ax.add_patch(strip_border)

# Upward arrow along right edge of strip
arrow_x = strip_x + STRIP_W + 0.2
ax.annotate(
    "",
    xy=(arrow_x, grid_h - 0.1),
    xytext=(arrow_x, 0.1),
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
)
ax.text(
    arrow_x + 0.08,
    grid_h / 2,
    r"m/n $\uparrow$",
    ha="left",
    va="center",
    fontsize=7,
    color=MUTED,
    rotation=90,
)

# Header above strip
ax.text(
    strip_x + STRIP_W / 2,
    grid_h + 0.65,
    "Statistical validity",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color=BORDER,
)
ax.text(
    strip_x + STRIP_W / 2,
    grid_h + 0.30,
    "(m/n ratio)",
    ha="center",
    va="center",
    fontsize=7.5,
    color=MUTED,
)

# Robust metrics note below strip
ax.text(
    strip_x + STRIP_W / 2,
    -0.30,
    r"R$^2$, T-MEX: robust to false-positive inflation",
    ha="center",
    va="center",
    fontsize=6,
    color=PASS_C,
    style="italic",
)

# ─── Legend (spans main grid width only) ───
legend_items = [
    (GOOD, GOOD_D, "Calibrated"),
    (MILD, MILD_D, "Some metrics fail"),
    (MODERATE, MODERATE_D, "Most metrics fail"),
    (SEVERE, SEVERE_D, "All metrics fail"),
]
leg_y = -0.55
total_legend_w = sum(len(l) * 0.07 + 0.55 for _, _, l in legend_items)
leg_x = (COLS * CELL_W - total_legend_w) / 2

for i, (bg, bd, label) in enumerate(legend_items):
    lx = leg_x + i * 2.1
    rect = FancyBboxPatch(
        (lx, leg_y - 0.1),
        0.2,
        0.2,
        boxstyle="round,pad=0.01",
        facecolor=bg,
        edgecolor=bd,
        linewidth=0.7,
    )
    ax.add_patch(rect)
    ax.text(
        lx + 0.32,
        leg_y,
        label,
        ha="left",
        va="center",
        fontsize=7.5,
        color=BORDER,
    )


# ─── Save ───
out_dir = Path(__file__).resolve().parent.parent / "experiments" / "runs"
out_dir.mkdir(parents=True, exist_ok=True)

fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(
        out_dir / f"phase_diagram.{ext}",
        bbox_inches="tight",
        dpi=300,
        facecolor=BG,
    )
print(f"Saved phase_diagram.pdf and phase_diagram.png to {out_dir}")
