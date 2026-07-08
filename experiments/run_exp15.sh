#!/bin/bash
#SBATCH --job-name=exp15_phase
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -e
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

REPO=/path/to/identifiability-guard
source "$REPO/.venv/bin/activate"
cd "$REPO/experiments"

# Probe already verified on prior run: encoder.m = 50 confirmed.
echo "=== Full experiment ==="
python exp15_phase_diagram.py

echo ""
echo "=== Done: $(date) ==="
