#!/bin/bash -l
#SBATCH --array=0-419
#SBATCH --constraint=icx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# 900 s EIGHT-variant benchmark (feasibility, solve time, collisions, jerk,
# target distance, polarization) for paper fig 3 + Table 2 -- all new data under
# the tuned defaults.
#   variant{01,02,04,06,07,08,10,11} x N{10,25,50,75,100,125,137} x seed{0..9}
#   = 8*7*10 = 560 per behavior, free space, reactive steering ON.
# One job per behavior (checkpoint keys carry the case name, so all three
# behaviors coexist in the same results dir):
#   sbatch --job-name=bench_pt  run_bench.sh phototaxis
#   sbatch --job-name=bench_po  run_bench.sh phototaxis_orbital
#   sbatch --job-name=bench_poc run_bench.sh phototaxis_orbital_contracting
# After each array: python bench_wrapper.py --aggregate --behavior <case> \
#                     --steps 9000 --results-dir ../results/raw/bench_900s \
#                     --variants "$VARIANTS"
# 5 h walltime: the slowest task (N=137, look-ahead variants) took ~1.7 h at
# 900 s, so the 4 h cap of bench600_job.sh leaves too little headroom.

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

BEH="${1:-phototaxis}"     # the paper campaign is phototaxis
# The task-id -> (variant, N, seed) mapping depends on this list, so the
# aggregate step MUST pass the identical string (see the header).
VARIANTS="Hard BF,Hard Dt-CBF,Slack BF,Slack Dt-CBF,LA Slack Dt-CBF,Slack Dt-HOCBF"

cd "$REPO/src/experiments/simulation/comparison_src"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID behavior=$BEH start=$(date '+%F %T')"

srun python bench_wrapper.py --task-id "$SLURM_ARRAY_TASK_ID" \
        --steps 9000 --results-dir ../results/raw/bench_900s \
        --behavior "$BEH" --variants "$VARIANTS" 2>&1 \
  | grep -vE 'ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration'
rc=${PIPESTATUS[0]}

echo "done task=$SLURM_ARRAY_TASK_ID rc=$rc end=$(date '+%F %T')"
exit "$rc"
