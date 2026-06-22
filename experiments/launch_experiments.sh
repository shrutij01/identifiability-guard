#!/bin/bash

# Launch identifiability-guard experiments on SLURM.
#
# Experiments are grouped into tiers by expected runtime:
#   - medium (2-3h):  exp01, exp02, exp03, exp05, exp15
#   - heavy  (3-5h):  exp04, exp06, exp07, exp08, exp09, exp11, exp12, exp13, exp14
#   - long   (5-6h):  exp10
#
# Usage:
#   bash experiments/launch_experiments.sh             # submit all
#   bash experiments/launch_experiments.sh exp01       # submit one experiment
#   bash experiments/launch_experiments.sh exp01 exp04 # submit several
#   bash experiments/launch_experiments.sh fast        # submit a tier
#   bash experiments/launch_experiments.sh heavy       # submit a tier
#   bash experiments/launch_experiments.sh --quick             # all, quick sanity check
#   bash experiments/launch_experiments.sh --quick exp01 exp04 # specific, quick

# ---- Paths (edit these for your cluster) ----
PROJECT_ROOT="/home/mila/j/joshi.shruti/causalrepl_space/identifiability-guard"
VENV_PATH="${PROJECT_ROOT}/.venv/bin/activate"

# ---- Job settings ----
memory="8Gb"
default_cpus=4
partition="main"

# ---- Setup directories ----
mkdir -p "${PROJECT_ROOT}/experiments/generated_jobs"
mkdir -p "${PROJECT_ROOT}/experiments/logs"
mkdir -p "${PROJECT_ROOT}/experiments/runs"

counter=0

submit_job() {
    local job_name="$1"
    local time_limit="$2"
    local cmd="$3"
    local cpus="${EXP_CPUS[$job_name]:-${default_cpus}}"

    local script_name="${PROJECT_ROOT}/experiments/generated_jobs/job_${job_name}.sh"

    cat > "${script_name}" <<EOF
#!/bin/bash
#SBATCH --job-name=ig-${job_name}
#SBATCH --output=${PROJECT_ROOT}/experiments/logs/${job_name}_%j.out
#SBATCH --error=${PROJECT_ROOT}/experiments/logs/${job_name}_%j.err
#SBATCH --time=${time_limit}
#SBATCH --mem=${memory}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --partition=${partition}

module load python/3.9 2>/dev/null || true

source ${VENV_PATH}

export PYTHONPATH="${PROJECT_ROOT}:\$PYTHONPATH"
cd ${PROJECT_ROOT}/experiments

${cmd}
EOF

    chmod +x "${script_name}"
    echo "Submitting ${job_name} (${time_limit})..."
    sbatch "${script_name}"
    ((counter++))
}

# ---- Experiment definitions (name -> time_limit, tier) ----
declare -A EXP_TIME
EXP_TIME[exp01]="02:00:00"   # multi-d (d=5,10,20), n=1000
EXP_TIME[exp02]="03:00:00"   # multi-d (d=5,10), finer alpha grid, n=1000
EXP_TIME[exp03]="03:00:00"   # multi-d (d=2,5,10), n up to 1000
EXP_TIME[exp04]="04:00:00"   # multi-d (d=2,5,10), 2D sweep rho x kappa
EXP_TIME[exp05]="03:00:00"   # multi-d (d=5,10), n=1000
EXP_TIME[exp06]="03:00:00"   # d=10, n=1000
EXP_TIME[exp07]="03:00:00"   # d=10, n=1000
EXP_TIME[exp08]="04:00:00"   # multi-d (d=5,20,50), n up to 4000
EXP_TIME[exp09]="04:00:00"   # multi-d (d=5,20,50), n up to 4000
EXP_TIME[exp10]="06:00:00"   # 4 DGPs x 4 encoders (incl E7) x 7 n values
EXP_TIME[exp11]="05:00:00"   # null encoders + high-m arms, n up to 10000
EXP_TIME[exp12]="04:00:00"   # ratio replot (overcomplete + undercomplete)
EXP_TIME[exp13]="04:00:00"   # d=3,5,10,20 x 4 ratios, 10 seeds
EXP_TIME[exp14]="03:00:00"   # 2 sweeps, d=10, 10 seeds
EXP_TIME[exp15]="04:00:00"   # 6x6 phase diagram, d=10, null encoder

# Per-experiment CPU overrides (default: ${default_cpus}).
# exp12-14 use joblib parallelism across seeds; request enough CPUs.
declare -A EXP_CPUS
EXP_CPUS[exp12]=5   # N_SEEDS=5,  N_JOBS=-1
EXP_CPUS[exp13]=4   # N_SEEDS=10, N_JOBS=-1 (seeds run in batches)
EXP_CPUS[exp14]=4   # N_SEEDS=10, N_JOBS=-1 (seeds run in batches)
EXP_CPUS[exp15]=4   # N_SEEDS=5,  36 cells in 6x6 grid

declare -A EXP_MODULE
EXP_MODULE[exp01]="exp01_invariance_across_dgps"
EXP_MODULE[exp02]="exp02_nonlinearity_sensitivity"
EXP_MODULE[exp03]="exp03_correlation_sign"
EXP_MODULE[exp04]="exp04_correlation_vs_entanglement"
EXP_MODULE[exp05]="exp05_predictability_vs_disentanglement"
EXP_MODULE[exp06]="exp06_dropped_variables"
EXP_MODULE[exp07]="exp07_redundancy_vs_compression"
EXP_MODULE[exp08]="exp08_encoding_type"
EXP_MODULE[exp09]="exp09_overcomplete_representations"
EXP_MODULE[exp10]="exp10_sample_sensitivity"
EXP_MODULE[exp11]="exp11_metric_inflation"
EXP_MODULE[exp12]="exp12_ratio_replot"
EXP_MODULE[exp13]="exp13_ratio_collapse"
EXP_MODULE[exp14]="exp14_disentangle_ratios"
EXP_MODULE[exp15]="exp15_phase_diagram"

# Tier groupings
FAST=""
MEDIUM="exp01 exp02 exp03 exp05 exp15"
HEAVY="exp04 exp06 exp07 exp08 exp09 exp11 exp12 exp13 exp14"
LONG="exp10"
ALL_EXPS="${FAST} ${MEDIUM} ${HEAVY} ${LONG}"

# ---- Parse --quick flag ----
QUICK_FLAG=""
args=()
for arg in "$@"; do
    if [ "$arg" = "--quick" ]; then
        QUICK_FLAG="--quick"
    else
        args+=("$arg")
    fi
done
set -- "${args[@]}"

# ---- Determine which experiments to run ----
resolve_targets() {
    case "$1" in
        fast)   echo "${FAST}" ;;
        medium) echo "${MEDIUM}" ;;
        heavy)  echo "${HEAVY}" ;;
        long)   echo "${LONG}" ;;
        *)      echo "$1" ;;
    esac
}

targets=""
if [ ${#} -eq 0 ]; then
    targets="${ALL_EXPS}"
else
    for arg in "$@"; do
        targets="${targets} $(resolve_targets "$arg")"
    done
fi

if [ -n "${QUICK_FLAG}" ]; then
    echo "*** QUICK MODE: main plots only, skip sensitivity sweeps ***"
fi

# ---- Submit jobs ----
for exp in ${targets}; do
    time_limit="${EXP_TIME[$exp]}"
    module_name="${EXP_MODULE[$exp]}"
    if [ -z "${time_limit}" ] || [ -z "${module_name}" ]; then
        echo "Unknown experiment: ${exp}. Skipping."
        continue
    fi
    # In quick mode, halve the time limit (minimum 30 min)
    if [ -n "${QUICK_FLAG}" ]; then
        time_limit="00:30:00"
    fi
    submit_job "${exp}" "${time_limit}" \
        "python ${module_name}.py ${QUICK_FLAG}"
done

echo ""
echo "Submitted ${counter} job(s). Check logs in experiments/logs/"
echo "Results (NPZ) will be saved to experiments/runs/<expNN>/"
echo ""
echo "After jobs complete, regenerate all plots locally:"
echo "  cd ${PROJECT_ROOT}/experiments && python plot_all.py"
