## Running Experiments

Each experiment is a self-contained script in the `experiments/` directory.
Install the package first:

```bash
pip install -e .
```

### Metric sets

Every experiment produces two versions of each plot:

| Tag    | Metrics                                                          | Purpose          |
|--------|------------------------------------------------------------------|------------------|
| `main` | MCC-P, MCC-S, R², DCI-D                                         | Paper figures    |
| `apx`  | MCC-P, MCC-S, MCC-RDC, R², DCI-D, InfoM, T-MEX, MIG            | Appendix figures |

Outputs are saved to `experiments/runs/<expNN>/` as
`<name>_main.pdf`, `<name>_main.png`, `<name>_apx.pdf`, `<name>_apx.png`.

### Notation

Three Greek letters are used consistently across all experiments:

| Symbol | Concept        | Encoder / DGP parameter | Axis label               |
|--------|----------------|-------------------------|--------------------------|
| ρ      | Correlation    | D2 `correlation`        | Correlation (ρ)          |
| α      | Non-linearity  | E2 `nonlinearity_strength` | Non-linearity (α)     |
| κ      | Entanglement   | E3 `condition_number`   | Entanglement (κ)         |

### Individual experiments

```bash
# Exp 1 – Invariance across DGP types (D1–D4 × E1)
python experiments/exp01_invariance_across_dgps.py

# Exp 2 – Non-linearity sensitivity (D1 × E2, sweep α)
python experiments/exp02_nonlinearity_sensitivity.py

# Exp 3 – Correlation sign effect (D2 × {E1, E3}, sweep ρ)
python experiments/exp03_correlation_sign.py

# Exp 4 – 2D heatmap: correlation ρ × entanglement κ (D2 × E3)
python experiments/exp04_correlation_vs_entanglement.py

# Exp 5 – Predictability vs disentanglement (D2 correlation vs D3 non-linearity)
python experiments/exp05_predictability_vs_disentanglement.py

# Exp 6 – Dropped variables & dimension inflation (E4 undercomplete)
python experiments/exp06_dropped_variables.py

# Exp 7 – Redundancy vs compression (D3/D4 × E4)
python experiments/exp07_redundancy_vs_compression.py

# Exp 8 – Encoding type effect (E1 vs E5, E6, E8)
python experiments/exp08_encoding_type.py

# Exp 9 – Overcomplete representations (E2 vs E5–E8)
python experiments/exp09_overcomplete_representations.py

# Exp 10 – Sample sensitivity (D1–D4 × E1–E3, sweep n)
python experiments/exp10_sample_sensitivity.py

# Exp 11 – Null-encoder inflation (E9, E10)
python experiments/exp11_metric_inflation.py

# Exp 12 – Re-plot Exp 6 with structural ratio m/d
python experiments/exp12_ratio_replot.py

# Exp 13 – Ratio collapse: does d/n govern metric behaviour?
python experiments/exp13_ratio_collapse.py

# Exp 14 – Disentangle m/n from d/n
python experiments/exp14_disentangle_ratios.py
```

### Saving and loading results

Each experiment automatically saves its numeric results to compressed NPZ
files under `experiments/runs/<expNN>/`. This allows expensive computations
to be run once (e.g. on a cluster) and plots to be regenerated later without
recomputing.

```bash
# Normal run: compute + save results + generate plots
python experiments/exp01_invariance_across_dgps.py

# Re-plot from saved results (no computation)
python experiments/exp01_invariance_across_dgps.py --plot-only

# Re-plot ALL experiments from saved results
python experiments/plot_all.py

# Re-plot specific experiments
python experiments/plot_all.py exp01 exp04
```

Saved files per experiment:
- `experiments/runs/<expNN>/results.npz` — compressed numeric arrays
- `experiments/runs/<expNN>/config.json` — axis labels, parameter lists, metadata

### Run all experiments sequentially

```bash
for f in experiments/exp*.py; do python "$f"; done
```

### Combined heatmap (big_table.py)

Generate a comprehensive heatmap showing all DGP × Encoder × Metric combinations:

```bash
python experiments/big_table.py
python experiments/big_table.py --samples 10000 --factors 6 --seed 123
python experiments/big_table.py --output results/my_heatmap.png
```

### Sensitivity analysis (sensitivity.py)

Run parameter sweeps with multi-seed aggregation:

```bash
# Sweep sample sizes
python experiments/sensitivity.py \
    --sweep-samples 500,1000,2500,5000,10000 \
    --dgp D1 --encoder E1 --n-seeds 5

# Sweep correlation ρ (always uses D2)
python experiments/sensitivity.py \
    --sweep-correlation 0.0,0.3,0.5,0.7,0.9 \
    --encoder E3 --n-seeds 5

# Sweep number of factors
python experiments/sensitivity.py \
    --sweep-factors 3,5,7,10 \
    --dgp D2 --encoder E4 --n-seeds 5

# Sweep non-linearity α (E2 encoder)
python experiments/sensitivity.py \
    --sweep-nonlinearity 0.0,0.25,0.5,0.75,1.0 \
    --dgp D1 --n-seeds 5

# Sweep encoder non-linearity α (E2, explicit label)
python experiments/sensitivity.py \
    --sweep-encoder-nonlinearity 0.0,0.25,0.5,0.75,1.0 \
    --dgp D1 --n-seeds 5

# Compute all APX metrics
python experiments/sensitivity.py \
    --sweep-samples 500,1000,5000 \
    --all-metrics --n-seeds 5

# Select specific metrics
python experiments/sensitivity.py \
    --sweep-samples 1000,5000 \
    --metrics dci_disentanglement,mcc_pearson,r2
```

### Quick mode

Pass `--quick` to skip appendix plots and sensitivity sweeps (runs all seeds,
generates only the 4 main-metric figures):

```bash
# Single experiment, quick
python experiments/exp04_correlation_vs_entanglement.py --quick

# All via SLURM, quick
bash experiments/launch_experiments.sh --quick

# Specific experiments via SLURM, quick
bash experiments/launch_experiments.sh --quick exp01 exp04
```

### Cluster submission (SLURM)

The launch script generates and submits per-experiment SLURM jobs:

```bash
# Submit ALL experiments
bash experiments/launch_experiments.sh

# Submit a single experiment
bash experiments/launch_experiments.sh exp01

# Submit multiple experiments
bash experiments/launch_experiments.sh exp01 exp04 exp10

# Submit by tier (medium / heavy / long)
bash experiments/launch_experiments.sh medium
bash experiments/launch_experiments.sh heavy
```

The launcher resolves the repository root relative to its own location and
uses `.venv` by default. Override the paths or SLURM settings through
environment variables when needed:
```bash
PROJECT_ROOT=/path/to/identifiability-guard \
VENV_PATH=/path/to/venv/bin/activate \
SLURM_PARTITION=compute \
bash experiments/launch_experiments.sh exp01
```

Tier time limits:
| Tier   | Time  | Experiments                           |
|--------|-------|---------------------------------------|
| medium | 2–3 h | exp01, exp02, exp03, exp05, exp15     |
| heavy  | 3–5 h | exp04, exp06–exp09, exp11–exp14       |
| long   | 5–6 h | exp10                                 |

Generated job scripts are in `experiments/generated_jobs/`.
Logs are written to `experiments/logs/`.

After jobs complete, regenerate all plots locally:
```bash
cd experiments && python plot_all.py
```
