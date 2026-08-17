#!/bin/bash
#SBATCH --job-name=exp15_phase
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv/bin/activate}"

if [[ ! -f "${VENV_PATH}" ]]; then
    echo "Virtual environment activation script not found: ${VENV_PATH}" >&2
    echo "Set VENV_PATH to the activation script for your environment." >&2
    exit 1
fi

source "${VENV_PATH}"
cd "${PROJECT_ROOT}/experiments"

# Probe already verified on prior run: encoder.m = 50 confirmed.
echo "=== Full experiment ==="
python exp15_phase_diagram.py

echo ""
echo "=== Done: $(date) ==="
