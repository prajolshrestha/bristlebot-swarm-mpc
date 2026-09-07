#!/bin/bash -l
#SBATCH --job-name=timectl
#SBATCH --constraint=icx
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# Uncontended solve-time reference for the paper's real-time claims. The campaign
# measures solve time with one task per core on SHARED nodes, where a co-tenant
# can inflate it via memory-bandwidth contention and make the paper understate
# its real-time margin. This job re-measures on an EXCLUSIVELY held node, one run
# at a time, so the campaign numbers can be checked against an uncontended
# reference. Short runs suffice: solve time is a per-solve property.
#
# Submit:  sbatch run_timing.sh

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
echo "host=$(hostname) exclusive=yes start=$(date '+%F %T')"

srun python timing_control.py --steps 300 --out timing_control_exclusive.csv 2>&1 \
  | grep --line-buffered -vE 'ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration'
rc=${PIPESTATUS[0]}
echo "done rc=$rc end=$(date '+%F %T')"
exit "$rc"
