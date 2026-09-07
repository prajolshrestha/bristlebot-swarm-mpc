#!/bin/bash -l
#SBATCH --array=0-839
#SBATCH --constraint=icx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# 900 s obstacle-matrix campaign for ONE deterministic variant (all new data).
#   behavior{phototaxis, phototaxis_orbital, no_light, phototaxis_orbital_contracting}
#   x N{10,25,50,75,100,125,137}
#   x obstacles{0,2,12} x seed{0..9} = 4*7*3*10 = 840 runs, 900 s each.
# Decode of SLURM_ARRAY_TASK_ID (0..629), seed fastest:
#   seed = t%10; t/=10; OI = t%3; t/=3; NI = t%7; t/=7; BI = t%NBEH
#
# Submit one job per variant:
#   for v in 01 02 04 06 07 08 10 11; do
#     sbatch --job-name=mx$v run_matrix.sh $v
#   done
# Single-behavior slice (e.g. the no_light figure set, 3*7*10 = 210 runs):
#   sbatch --job-name=mx08nl --array=0-209 run_matrix.sh 08 no_light
#
# CSVs -> results_matrix_900s_tuned/ (this dir), raw .npz -> $WORK.

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

STEPS=9000
VAR="${1:?usage: sbatch --job-name=mxNN run_matrix.sh <variant> [behavior]}"
ONE_BEH="${2:-}"

if [ -n "$ONE_BEH" ]; then
  BEHAVIORS=("$ONE_BEH"); NBEH=1
else
  # All four conditions: the two tasks, the no-task control (no_light, needed by
  # Fig 7 and matrix900_summary.csv), and the contracting orbital variant. The
  # last is not read by any current matrix figure but is kept so the campaign
  # covers every behavior the paper discusses.
  BEHAVIORS=(phototaxis phototaxis_orbital no_light phototaxis_orbital_contracting); NBEH=4
fi
NS=(10 25 50 75 100 125 137)
OBS=(0 2 12)

t=$SLURM_ARRAY_TASK_ID
SEED=$(( t % 10 )); t=$(( t / 10 ))
OI=$(( t % 3 ));    t=$(( t / 3 ))
NI=$(( t % 7 ));    t=$(( t / 7 ))
BI=$(( t % NBEH ))
BEH=${BEHAVIORS[$BI]}; N=${NS[$NI]}; OB=${OBS[$OI]}

cd "$REPO/src/experiments/simulation/comparison_src"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID variant=$VAR behavior=$BEH N=$N obs=$OB seed=$SEED start=$(date '+%F %T')"

srun python matrix_wrapper.py --variant "$VAR" --behavior "$BEH" --n "$N" \
        --obstacles "$OB" --reactive on --seed "$SEED" --steps "$STEPS" \
        --checkpoint-every 500 --flush-every 200 2>&1 \
  | grep -vE 'ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration'
rc=${PIPESTATUS[0]}

echo "done task=$SLURM_ARRAY_TASK_ID rc=$rc end=$(date '+%F %T')"
exit "$rc"
