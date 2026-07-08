# Handoff: rerun Experiment 15 (null-encoder phase diagram) on the cluster

> **Status 2026-07-08 (late)**: a high-effort code review of the diff passed
> (8 findings — all fixed or deliberately deferred; compute path verified
> bug-free), stale pre-fix results were archived to
> `experiments/runs_old/exp15_stale_pre_m_fix/`, and a LOCAL run of the full
> sweep may already be underway on the laptop. Before launching on the
> cluster, check whether `experiments/runs/exp15_e10/` already exists locally.
> Note: the 1×3 inspection figure is now saved as `..._1x3.{pdf,png}` (the
> `_main` name belongs to the METRICS_MAIN grid figure), and both plotters
> refuse results whose config lacks `m_fixed` (pre-redesign data).

## Goal
Rerun `experiments/exp15_phase_diagram.py` after a redesign + bug fix, check the
results against theory, and regenerate the paper figures. **CPU-only job — do
not request a GPU** (see "Compute" below).

## State of the code
Branch `sj/mcc`, with 4 files modified (make sure these changes are present on
the cluster checkout — they were uncommitted on the laptop, so pull the branch
after they've been committed & pushed, or rsync the working tree):

1. `src/identifiability_guard/evaluation/helpers.py` — **the critical fix**:
   `create_encoder_with_params` now forwards the `m` kwarg for null encoders
   E9/E10. Before, it was silently dropped, pinning m = d in every cell of the
   old phase diagram.
2. `experiments/exp15_phase_diagram.py` — redesigned experiment:
   - Fixed-m grid: `M_FIXED = 50`; rows vary d ∈ {100, 50, 25, 10, 5}
     (m/d ∈ {0.5, 1, 2, 5, 10}), columns vary n ∈ {5000, 1000, 500, 100, 50}
     (m/n ∈ {0.01, 0.05, 0.1, 0.5, 1}). The two ratio axes are now independent
     knobs (the old design derived both m and n from the ratios at fixed d,
     confounding the axes).
   - Collapse sweep: d = 10, m ∈ {10, 50, 200}, n ∈ {20, 50, 100, 200, 500,
     1000, 2000, 5000}, MCC-Pearson/Spearman only.
   - `N_SEEDS = 5`, seed-parallel with `N_JOBS = 5`.
3. `experiments/param_check.py` — grid documentation updated to match.
4. `experiments/plot_paper_figures.py` — `plot_exp15` now produces the E10 main
   heatmap, E9 appendix heatmap, and a 2-panel MCC collapse figure
   (`exp15_mcc_collapse`).

## Theory being tested
Under a null encoder (output independent of input), the expected null MCC is
governed by the extreme-value bound

    E[MCC-P] ≈ sqrt(2 · log m / n_eff),   n_eff = n − int(0.8·n)

(n_eff is the held-out 20% test split — the pipeline scores pure-statistic
metrics on the test split). It depends on n strongly, on m logarithmically,
and **not on d**.

## Compute
Everything is numpy/scipy/sklearn; torch appears only in
`src/identifiability_guard/metrics/mcc/_sinkhorn.py` on ≤100×50 matrices, so a
CPU-only torch build is fine and a GPU would sit idle. Data is tiny (n ≤ 5000,
dims ≤ 100). Ask for something like `--cpus-per-task=8 --mem=16G`, no `--gres`.
Total work: phase grid 2 encoders × 25 cells × 5 seeds (full APX metric set,
5 seeds run in parallel) + collapse sweep 2 × 24 cells × 5 seeds (MCC only,
cheap).

## Environment setup
From repo root (has `pyproject.toml`):

```bash
python -m venv .venv && source .venv/bin/activate   # or module load python / uv venv
pip install -e .
pip install matplotlib   # if not pulled in by the package deps
```

## Step 1 — verification probe (run before the full experiment)
```bash
cd experiments && python -c "
import time, numpy as np
from utils import evaluate_dgp_encoder, make_registry, multi_seed_evaluate, APX_METRICS

registry = make_registry()

# Verify the fix flows through the full pipeline: MCC must now depend on m
from identifiability_guard.evaluation.helpers import create_encoder_with_params
enc = create_encoder_with_params('E10', 25, 0, {'m': 50})
print('E10 with d=25, requested m=50 → encoder.m =', enc.m)

# Timing probe: worst cell (d=100, n=5000), one seed, full APX metric set
t0 = time.time()
res = evaluate_dgp_encoder('D1', 'E10', n_samples=5000, n_factors=100, seed=42,
                           encoder_kwargs={'m': 50},
                           metrics_to_compute=set(APX_METRICS), registry=registry)
t1 = time.time()
print(f'worst cell, 1 seed: {t1-t0:.1f}s')
print({k: round(v, 3) for k, v in sorted(res.items())})
" 2>&1 | grep -v Warning | tail -5
```
Pass criteria (reference values measured on the laptop, 2026-07-08):
- `encoder.m = 50` (if it prints 25, the helpers.py fix isn't on this checkout — stop).
- Scores for this cell: `mcc_pearson ≈ mcc_spearman ≈ 0.086` (theory:
  √(2·ln 50/1000) = 0.0885), `r2 ≈ 0.0`, `mig ≈ 0.002`, `dci ≈ 0.03`,
  `infom ≈ 0.06`. `mcc_rdc ≈ 0.96` and `tmex ≈ 0.96` are hugely inflated —
  that is the expected failure story for those metrics, not a bug.
- Timing: this worst cell took ~685 s for one seed on a laptop. Cells run
  sequentially with the 5 seeds in parallel (`N_JOBS = 5`), and per-cell cost
  drops roughly with n·d, so expect ~30 min/encoder for the phase grid +
  a few min for the collapse sweep ⇒ ~1–1.5 h wall total.
  `--cpus-per-task=8` is plenty (more cores won't help unless you also
  parallelize across cells).

## Step 2 — full run
```bash
cd experiments && python exp15_phase_diagram.py
```
Outputs:
- Results: `experiments/runs/exp15_e9/`, `exp15_e10/`, `exp15_collapse_e9/`,
  `exp15_collapse_e10/` (each a `results.npz` + config).
- Inspection figures via `savefig(..., subdir="exp15")`.

## Step 3 — validate against theory before declaring success
Load the grids (rows = m/d index, cols = m/n index) and check, for MCC-P under
both E9 and E10:
- **Flat down each column**: cells differ only in d, so scores should be ≈
  constant down a column. This flatness is the visible signature that the
  m-kwarg fix worked (the buggy run varied along d because m was pinned to d).
- **Rising along each row** with predicted values √(2·ln 50 / n_eff):
  n=5000 → ≈0.09, n=1000 → ≈0.20, n=500 → ≈0.28, n=100 → ≈0.63, n=50 → ≈0.88.
- **Collapse sweep**: the three m-curves should overlap when plotted against
  √(2·log m / n_eff) and track min(x, 1); they should NOT overlap vs m/n.
- Other APX metrics (R², DCI, …) carry no such prediction — don't treat their
  patterns as failures; the paper's quantitative claim is about MCC.

## Step 4 — paper figures
```bash
cd experiments && python plot_paper_figures.py
```
Regenerates the exp15 heatmaps (E10 main, E9 appendix) and `exp15_mcc_collapse`.

## Step 5 — bring results home
rsync `experiments/runs/exp15*` and the generated figure files back to the
laptop (runs/ artifacts are what `plot_paper_figures.py --plot-only` style
regeneration needs locally).
