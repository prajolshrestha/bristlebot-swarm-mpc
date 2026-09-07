#!/bin/bash -l
#SBATCH --job-name=run_distance
#SBATCH --array=0-419
#SBATCH --constraint=icx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# 900 s safety-margin sweep for paper fig 4 (d_rr / d_ro, obstacles ON,
# phototaxis) -- all new data under the tuned defaults, eight variants.
#   variant{01,02,04,06,07,08,10,11} x N{10,25,50,75,100,125,137} x seed{0..9}
#   = 8*7*10 = 560 runs.
# Task order = distance_comparison.run's loop (variant -> N -> seed), decoded
# inside distance_wrapper.py from the (now eight-entry) registry.
# After completion:
#   DIST_RESULTS_DIR=$PWD/../results/raw/distance_900s \
#     python distance_wrapper.py --merge --steps 9000

unset SLURM_EXPORT_ENV
set -uo pipefail

# Slurm resolves the relative --output path above against the submission
# directory, creating it if needed, so submitting from the wrong place does not
# fail -- it just scatters a stray results/logs/ there. Checking for results/logs
# cannot catch that (Slurm has already made it); check for a marker that only
# exists in src/experiments/simulation instead.
_sd="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -d "$_sd/behaviors_src" ] || [ ! -d "$_sd/comparison_src" ]; then
  echo "submitted from $_sd, expected src/experiments/simulation --" >&2
  echo "the relative --output path above would scatter logs there instead" >&2
  exit 1
fi

module add python 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acados_casadi
# Locate the repository without hardcoding a path: walk up from wherever the job
# was submitted until env.sh appears. Override with BB_REPO if you submit from
# outside the tree. SBATCH directives above cannot use variables, so their log
# paths are relative and land in the submit directory.
REPO="${BB_REPO:-}"
if [ -z "$REPO" ]; then
  d="${SLURM_SUBMIT_DIR:-$PWD}"
  while [ "$d" != "/" ] && [ ! -f "$d/env.sh" ]; do d="$(dirname "$d")"; done
  REPO="$d"
fi
if [ ! -f "$REPO/env.sh" ]; then
  echo "cannot locate the repository; set BB_REPO=/path/to/bristlebot-swarm-mpc" >&2
  exit 1
fi

source "$REPO/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

cd "$REPO/src/experiments/simulation/comparison_src"
export DIST_RESULTS_DIR="$PWD/../results/raw/distance_900s"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID start=$(date '+%F %T')"

srun python distance_wrapper.py --task-id "$SLURM_ARRAY_TASK_ID" --steps 9000 2>&1 \
  | grep -vE 'ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration'
rc=${PIPESTATUS[0]}

echo "done task=$SLURM_ARRAY_TASK_ID rc=$rc end=$(date '+%F %T')"
exit "$rc"
