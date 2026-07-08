"""
Master parameter reference for all experiments.

Every experiment's (d, m, n, m/d, m/n, d/n) range is recorded here so that
reviewers and collaborators can verify consistency at a glance.

Run this file to print the full table:

    python experiments/param_check.py

The key identity is:  m/n = (m/d) * (d/n)
Only two of the three ratios are independent.

Safe-zone rules of thumb:
  - m/n < 0.05  and  d/n < 0.05  →  safe (classical statistics)
  - 0.05 < ratio < 0.10          →  borderline
  - ratio > 0.10                  →  danger zone (finite-sample bias)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Encoder output dimension rules
# ---------------------------------------------------------------------------
# E1, E2, E3:  m = d       (elementwise / entangled, square)
# E4:          m < d       (undercomplete, m specified)
# E5, E6:     m = d + 1   (overcomplete, one extra)
# E7, E8:     m = 2d      (overcomplete, double)
# E9, E10:    m = d        (null encoders, default; overridable via m kwarg)


def _ratios(d, m, n):
    """Return (m/d, m/n, d/n) as floats, or None for variable entries."""
    md = m / d if (d and m) else None
    mn = m / n if (n and m) else None
    dn = d / n if (n and d) else None
    return md, mn, dn


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------
# Each entry: (exp, description, d, m, n, notes)
# For sweeps, the swept variable is shown as a list.

EXPERIMENTS = [
    # ── Exp 01: Invariance across DGPs ──
    {
        "exp": "exp01",
        "description": "Invariance across DGPs (D1-D4 x E1)",
        "d_values": [5, 10, 20],
        "encoders": ["E1"],
        "m_rule": "m = d",
        "n": 1000,
        "notes": "Multi-d sweep. m=d for E1, so m/d=1.0 always.",
    },

    # ── Exp 02: Nonlinearity sensitivity ──
    {
        "exp": "exp02",
        "description": "Nonlinearity sensitivity (D1 x E2-variant)",
        "d_values": [5, 10],
        "encoders": ["E2-variant (alpha sweep)"],
        "m_rule": "m = d",
        "n": 1000,
        "notes": "Sweep alpha in [0, 0.005, ..., 1.0] with symlog x-axis.",
    },

    # ── Exp 03: Correlation sign ──
    {
        "exp": "exp03",
        "description": "Correlation sign effect (D2 x {E1, E3})",
        "d_values": [2, 5, 10],
        "encoders": ["E1", "E3"],
        "m_rule": "m = d",
        "n_by_d": {2: 100, 5: 1000, 10: 1000},
        "notes": "d=2 uses n=100 (full rho range). PSD limits negative rho at d>2.",
    },

    # ── Exp 04: Correlation vs entanglement ──
    {
        "exp": "exp04",
        "description": "Correlation (rho) vs entanglement (kappa) heatmap",
        "d_values": [2, 5, 10],
        "encoders": ["E3"],
        "m_rule": "m = d",
        "n_by_d": {2: 100, 5: 1000, 10: 1000},
        "notes": "2D sweep: rho x kappa. kappa in {1..50}.",
    },

    # ── Exp 05: Predictability vs disentanglement ──
    {
        "exp": "exp05",
        "description": "Factor predictability vs disentanglement (D2 rho, D3 alpha)",
        "d_values": [5, 10],
        "encoders": ["E1"],
        "m_rule": "m = d",
        "n": 1000,
        "notes": "D2 panel: sweep rho. D3 panel: sweep alpha.",
    },

    # ── Exp 06: Dropped variables ──
    {
        "exp": "exp06",
        "description": "Dropped variables (E4, m < d)",
        "d_values": [10],
        "encoders": ["E4"],
        "m_rule": "m in {1, ..., d-1}",
        "n": 1000,
        "notes": "6a: sweep m, fixed d=10. 6b: sweep d for fixed m=3.",
    },

    # ── Exp 07: Redundancy vs compression ──
    {
        "exp": "exp07",
        "description": "Redundancy vs compression (D3/D4 x E4)",
        "d_values": [10],
        "encoders": ["E4"],
        "m_rule": "m in {3, ..., 9}",
        "n": 1000,
        "notes": "2D sweep: redundancy r x compression m.",
    },

    # ── Exp 08: Encoding type effect ──
    {
        "exp": "exp08",
        "description": "Encoding type effect (E1 vs E5, E6, E8)",
        "d_values": [5, 20, 50],
        "encoders": ["E1", "E5", "E6", "E8"],
        "m_rule": "E1: m=d, E5/E6: m=d+1, E8: m=2d",
        "n_by_d": {5: 1000, 20: 1000, 50: 4000},
        "notes": "D1 only at d=20,50. Full DGP set at d=5.",
    },

    # ── Exp 09: Overcomplete representations ──
    {
        "exp": "exp09",
        "description": "Overcomplete representations (E3 vs E5-E8)",
        "d_values": [5, 20, 50],
        "encoders": ["E3", "E5", "E6", "E7", "E8"],
        "m_rule": "E3: m=d, E5/E6: m=d+1, E7/E8: m=2d",
        "n_by_d": {5: 1000, 20: 1600, 50: 4000},
        "notes": "n=1600 at d=20 because max m=2d=40 (E7/E8), rule: n >= 40*m_max.",
    },

    # ── Exp 10: Sample sensitivity ──
    {
        "exp": "exp10",
        "description": "Sample sensitivity (D1-D4 x {E1,E2,E3,E7})",
        "d_values": [5],
        "encoders": ["E1", "E2", "E3", "E7"],
        "m_rule": "E1/E2/E3: m=d=5, E7: m=2d=10",
        "n_values": [50, 100, 200, 500, 1000, 2000, 5000],
        "notes": "E7 overcomplete arm gives m/n up to 0.20 at n=50.",
    },

    # ── Exp 11: Metric inflation (null encoders) ──
    {
        "exp": "exp11",
        "description": "Metric inflation under null encoders (E9, E10)",
        "d_values": [5],
        "encoders": ["E9", "E10"],
        "m_rule": "default m=d=5; high-m arms: m=50, m=200",
        "n_values": [50, 100, 200, 500, 1000, 2000, 5000, 10000],
        "notes": "High-m stress test: m=50 gives m/n=1.0 at n=50; m=200 gives m/n=4.0.",
    },

    # ── Exp 12: Ratio replot of exp06 ──
    {
        "exp": "exp12",
        "description": "Re-plot exp06 with m/d on x-axis",
        "d_values": [10],
        "encoders": ["E1/E4"],
        "m_rule": "sweep m/d from 0.1 to 1.0",
        "n": 1000,
        "notes": "No new compute; re-axes exp06 data. Also 6b: d sweep with fixed m=3.",
    },

    # ── Exp 13: Ratio collapse test ──
    {
        "exp": "exp13",
        "description": "Ratio collapse test: does d/n govern metric behaviour?",
        "d_values": [3, 5, 10, 20],
        "encoders": ["E1"],
        "m_rule": "m = d (E1)",
        "ratios": "d/n in {0.02, 0.05, 0.10, 0.20}",
        "notes": "n derived from d/ratio. 10 seeds per cell.",
    },

    # ── Exp 14: Disentangle m/n from d/n ──
    {
        "exp": "exp14",
        "description": "Disentangle m/n from d/n (E4, d=10)",
        "d_values": [10],
        "encoders": ["E4"],
        "sweep_a": "m in {1,3,5,7,9}, n=1000 (d/n=0.01 constant)",
        "sweep_b": "m=5, n in {2000,1000,500,200,100,50} (m/d=0.50 constant)",
        "notes": "Two orthogonal sweeps. 10 seeds per cell.",
    },

    # ── Exp 15: Phase diagram ──
    {
        "exp": "exp15",
        "description": "Phase diagram: metric reliability under null encoder (fixed m)",
        "d_values": [100, 50, 25, 10, 5],
        "encoders": ["E9", "E10"],
        "m_values": [50],
        "md_ratios": [0.5, 1.0, 2.0, 5.0, 10.0],
        "mn_ratios": [0.01, 0.05, 0.10, 0.50, 1.00, 2.00, 5.00],
        "notes": "5x7 heatmap. m=50 FIXED; rows vary d = m/(m/d), cols vary "
                 "n = m/(m/n) — axes are independent knobs. Plus collapse "
                 "sweep: d=10, m in {10,50,200}, n in {20..5000}.",
    },
]


# ---------------------------------------------------------------------------
# Ratio table printer
# ---------------------------------------------------------------------------

def _m_values_for_encoder(enc: str, d: int) -> list[int]:
    """Return the list of output dimensions m for an encoder at dimension d."""
    if enc in ("E1", "E2", "E3", "E9", "E10"):
        return [d]
    elif enc in ("E5", "E6"):
        return [d + 1]
    elif enc in ("E7", "E8"):
        return [2 * d]
    elif enc == "E4":
        return list(range(1, d))  # undercomplete
    return [d]


def print_ratio_table():
    """Print a comprehensive table of (d, m, n, m/d, m/n, d/n) for all experiments."""
    print("=" * 100)
    print(f"{'Exp':<7} {'Description':<50} {'d':>4} {'m':>6} {'n':>7} "
          f"{'m/d':>6} {'m/n':>7} {'d/n':>7}  Notes")
    print("-" * 100)

    for entry in EXPERIMENTS:
        exp = entry["exp"]
        desc = entry["description"][:48]
        notes = entry.get("notes", "")

        d_values = entry.get("d_values", [])
        n_fixed = entry.get("n", None)
        n_values = entry.get("n_values", None)
        n_by_d = entry.get("n_by_d", {})

        # For simple experiments with fixed n
        if n_fixed and not n_values:
            for d in d_values:
                n = n_by_d.get(d, n_fixed)
                encoders = entry.get("encoders", [])
                for enc_str in encoders:
                    # Try to get actual m values
                    enc_key = enc_str.split(" ")[0].split("/")[0].split("-")[0]
                    if enc_key in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"):
                        ms = _m_values_for_encoder(enc_key, d)
                    else:
                        ms = [d]  # fallback
                    for m in ms[:3]:  # show at most 3 m values
                        md, mn, dn = _ratios(d, m, n)
                        flag = ""
                        if mn and mn > 0.10:
                            flag = " !!!"
                        elif mn and mn > 0.05:
                            flag = " !"
                        print(f"{exp:<7} {desc:<50} {d:>4} {m:>6} {n:>7} "
                              f"{md:>6.2f} {mn:>7.4f} {dn:>7.4f}{flag}")
                    if len(ms) > 3:
                        print(f"{'':>7} {'':>50} {'...':>4}")
        elif n_values:
            # Sweep over n
            for d in d_values:
                n_min, n_max = min(n_values), max(n_values)
                encoders = entry.get("encoders", [])
                for enc_str in encoders:
                    enc_key = enc_str.split(" ")[0].split("/")[0].split("-")[0]
                    if enc_key in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"):
                        ms = _m_values_for_encoder(enc_key, d)
                    else:
                        ms = [d]
                    for m in ms[:2]:
                        md_val = m / d
                        mn_min = m / n_max
                        mn_max = m / n_min
                        dn_min = d / n_max
                        dn_max = d / n_min
                        flag = " !!!" if mn_max > 0.10 else (" !" if mn_max > 0.05 else "")
                        print(f"{exp:<7} {desc:<50} {d:>4} {m:>6} "
                              f"{n_min:>3}-{n_max:<4}"
                              f" {md_val:>5.2f} "
                              f"{mn_min:.4f}-{mn_max:.4f} "
                              f"{dn_min:.4f}-{dn_max:.4f}{flag}")
        else:
            # Special cases (ratio-based, phase diagram, etc.)
            print(f"{exp:<7} {desc:<50} "
                  f"{'var':>4} {'var':>6} {'var':>7} "
                  f"{'var':>6} {'var':>7} {'var':>7}  {notes[:40]}")

    print("=" * 100)
    print("\nFlags:  ! = borderline (m/n > 0.05),  !!! = danger zone (m/n > 0.10)")
    print("Identity: m/n = (m/d) * (d/n)")


# ---------------------------------------------------------------------------
# Detailed per-experiment ratio grids
# ---------------------------------------------------------------------------

def print_exp15_grid():
    """Print the full exp15 phase diagram grid (fixed-m design)."""
    md_ratios = [0.5, 1.0, 2.0, 5.0, 10.0]
    mn_ratios = [0.01, 0.05, 0.10, 0.50, 1.00, 2.00, 5.00]
    m = 50  # fixed; rows vary d, columns vary n

    print("\n\nExp15 Phase Diagram Grid (m=50 fixed, E9/E10 null encoders)")
    print("=" * 70)
    header = f"{'m/d':>6} {'d':>4}  " + "  ".join(f"{'m/n='+f'{mn:.2f}':>10}" for mn in mn_ratios)
    print(header)
    print("-" * 70)
    for md in md_ratios:
        d = int(round(m / md))
        row = f"{md:>6.1f} {d:>4}  "
        for mn in mn_ratios:
            n = int(round(m / mn))
            row += f"{'n='+str(n):>10}"
        print(row)
    print()


def print_exp13_grid():
    """Print the exp13 ratio collapse grid."""
    d_values = [3, 5, 10, 20]
    ratio_values = [0.02, 0.05, 0.10, 0.20]

    print("\n\nExp13 Ratio Collapse Grid (m=d, E1)")
    print("=" * 50)
    header = f"{'d/n':>6}  " + "  ".join(f"d={d:>3}" for d in d_values)
    print(header)
    print("-" * 50)
    for r in ratio_values:
        row = f"{r:>6.2f}  "
        for d in d_values:
            n = int(round(d / r))
            row += f"n={n:>4}  "
        print(row)
    print()


if __name__ == "__main__":
    print_ratio_table()
    print_exp15_grid()
    print_exp13_grid()
