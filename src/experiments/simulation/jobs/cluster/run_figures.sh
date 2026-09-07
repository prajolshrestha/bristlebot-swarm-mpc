#!/bin/bash -l
#SBATCH --job-name=bbfigs
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%j.out
#
# Render paper figures into results/figures/. Plotting only - reads existing
# data, runs no simulation.
#
#   cd src/experiments/simulation
#   sbatch jobs/cluster/run_figures.sh [script.py ...]
#
# With no arguments it renders every script listed in DEFAULT_SCRIPTS below.

unset SLURM_EXPORT_ENV
set -uo pipefail

_sd="${SLURM_SUBMIT_DIR:-$PWD}"
[ -d "$_sd/behaviors_src" ] || { echo "submit from src/experiments/simulation" >&2; exit 1; }
cd "$_sd" || exit 1

module add python 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acados_casadi

REPO="${BB_REPO:-}"
if [ -z "$REPO" ]; then
  d="$PWD"
  while [ "$d" != "/" ] && [ ! -f "$d/env.sh" ]; do d="$(dirname "$d")"; done
  REPO="$d"
fi
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"
export MPLBACKEND=Agg


# The stored snapshots were collected at these times; the collector derives its
# output directory from them, so the renderer only finds the pickles when the same
# value is exported. Without it it silently reports "[skip]" for every arena.
export SNAP_TIMES=0,225,450,675,900

mkdir -p results/figures
rc=0

run() {  # run() <dir> <description> <command...>
  local d="$1"; local what="$2"; shift 2
  echo "=== $what ==="
  ( cd "$d" && "$@" ) || rc=$?
}

run behaviors_src   "robot_model"                     python robot_model.py
run behaviors_src   "behavior_snapshots_{free,2obs,12obs}" python render_obstacle_grouped_figs.py
run comparison_src  "safety_formulation_comparison"   python paper_comparison_figure.py
# --plot-only: without it this re-runs the whole distance campaign instead of
# plotting the CSV that is already on disk.
run comparison_src  "safety_margins"                  python distance_comparison.py --plot-only
run comparison_src  "order_nnd_by_behavior"           python fig_order_nnd.py --both
run comparison_src  "behavior_collisions_by_behavior" python matrix_figures.py --collisions2 --variant 05
echo "--- results/figures now holds ---"
ls -la results/figures
exit "$rc"
