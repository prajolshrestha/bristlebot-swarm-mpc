#!/bin/bash -l
#SBATCH --job-name=run_snapshots
#SBATCH --constraint=icx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# 900 s behavior snapshot figures (deployed variant, N=50): 4 behaviors x
# {0,2,12} obstacles = 12 sims sampled at t = 0/225/450/675/900 s, then the
# three obstacle-grouped renders (paper figs 5, A.1, A.2).
#
# The collector and renderer live in the deployed variant's experiments dir
# (they import its drawing helpers), so this job cds there; only the submission
# lives with the rest of the campaign. SPAN_TAG in
# collect_behavior_obstacle_snapshots.py derives the output dirs from SNAP_TIMES
# (behavior_obstacle_review_900s/, obstacle_grouped_figs_900s/), so this run
# cannot collide with the existing 60 s or 600 s sets.

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
export SNAP_TIMES=0,225,450,675,900

cd "$REPO/src/experiments/simulation/behaviors_src"
echo "host=$(hostname) start=$(date '+%F %T')"

python collect_behavior_obstacle_snapshots.py --workers 12 \
    --behaviors phototaxis phototaxis_no_reynolds phototaxis_orbital no_light \
    --n 50 2>&1 \
  | grep -vE 'ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration'
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "collect failed rc=$rc"; exit "$rc"; fi

python render_obstacle_grouped_figs.py
rc=$?

echo "done rc=$rc end=$(date '+%F %T')"
exit "$rc"
