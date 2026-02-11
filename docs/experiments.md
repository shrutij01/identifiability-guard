## Running Experiments

Each experiment is a self-contained script in the `experiments/` directory.
Run them from the project root:

```bash
# Run all experiments sequentially
for f in experiments/exp*.py; do python "$f"; done

# Experiment 1 – Invariance across DGP types
python experiments/exp01_invariance_across_dgps.py

# Experiment 2 – Nonlinearity sensitivity (+ encoder NL sweep via sensitivity.py)
python experiments/exp02_nonlinearity_sensitivity.py

# Experiment 3 – Correlation sign effect (+ correlation sweeps via sensitivity.py)
python experiments/exp03_correlation_sign.py

# Experiment 4 – 2D heatmap: correlation × nonlinearity
python experiments/exp04_correlation_vs_entanglement.py

# Experiment 5 – Predictability vs disentanglement (+ NL & correlation sweeps)
python experiments/exp05_predictability_vs_disentanglement.py

# Experiment 6 – Dropped variables & dimension inflation (+ factor sweeps)
python experiments/exp06_dropped_variables.py

# Experiment 9 – Overcomplete representations
python experiments/exp09_overcomplete_representations.py

# Experiment 10 – Sample sensitivity grid (+ sample sweep D2×E3)
python experiments/exp10_sample_sensitivity.py

# Experiment 11 – Null-encoder inflation (+ sample sweep D1×E10)
python experiments/exp11_metric_inflation.py
```

Results (PDF + PNG plots) are saved to `results/experiments/<expNN>/`.

### Combined Heatmap (big_table.py)

Generate a comprehensive heatmap showing all DGP × Encoder × Metric combinations:

```bash
# Default settings (300 samples, 5 factors)
python experiments/big_table.py

# Custom configuration
python experiments/big_table.py --samples 10000 --factors 6 --seed 123

# Specify output path
python experiments/big_table.py --output results/my_heatmap.png
```

Output: a multi-panel figure with one heatmap per DGP, encoder × metric cells
annotated with scores, and a timing/memory profiling table.

### Sensitivity Analysis (sensitivity.py)

Run parameter sweeps with multi-seed aggregation and camera-ready plots:

```bash
# Sweep sample sizes
python experiments/sensitivity.py \
    --sweep-samples 500,1000,2500,5000,10000 \
    --dgp D1 --encoder E1 --n-seeds 5

# Sweep correlation values (always uses D2)
python experiments/sensitivity.py \
    --sweep-correlation 0.0,0.3,0.5,0.7,0.9 \
    --encoder E2 --n-seeds 5

# Sweep number of factors
python experiments/sensitivity.py \
    --sweep-factors 3,5,7,10 \
    --dgp D2 --encoder E4 --n-seeds 5

# Sweep nonlinearity strength (E2 encoder)
python experiments/sensitivity.py \
    --sweep-nonlinearity 0.0,0.25,0.5,0.75,1.0 \
    --dgp D1 --n-seeds 5

# Sweep encoder nonlinearity strength (E2, explicit label)
python experiments/sensitivity.py \
    --sweep-encoder-nonlinearity 0.0,0.25,0.5,0.75,1.0 \
    --dgp D1 --n-seeds 5

# Compute all metrics (default uses a fast subset)
python experiments/sensitivity.py \
    --sweep-samples 500,1000,5000 \
    --all-metrics --n-seeds 5

# Select specific metrics
python experiments/sensitivity.py \
    --sweep-samples 1000,5000 \
    --metrics dci_disentanglement,mcc_pearson,r2

# Custom output directory
python experiments/sensitivity.py \
    --sweep-samples 1000,5000 \
    --output-dir results/my_sweep
```

Each sweep produces:
- A JSON file with raw per-seed results and aggregated statistics
- A camera-ready PNG plot with 95% CI error bands (one panel per metric)
