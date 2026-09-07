#!/bin/bash -l
#SBATCH --job-name=bbvidrow
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%j.out
#
# Three one-row montages at the Elsevier recommended frame size (492x276, 750 kbps).
#
#   cd src/experiments/simulation
#   sbatch jobs/cluster/run_row_montages.sh
#
# This is a TRANSCODE, not a simulation: it windows the existing 900 s clips in
# results/videos/ down to their first 180 s, crops each to its arena, and lays the
# Nothing is re-simulated, so it runs in minutes rather than hours.

unset SLURM_EXPORT_ENV
set -uo pipefail

_sd="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -d "$_sd/behaviors_src" ]; then
  echo "submitted from $_sd, expected src/experiments/simulation --" >&2
  echo "the relative --output path above would scatter logs there instead" >&2
  exit 1
fi
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
if [ ! -f "$REPO/env.sh" ]; then
  echo "cannot locate the repository; set BB_REPO=/path/to/bristlebot-swarm-mpc" >&2
  exit 1
fi
source "$REPO/env.sh"
export MPLBACKEND=Agg

SRC="$REPO/src/experiments/simulation/results/videos"
missing=0
for b in phototaxis no_reynolds orbital no_stimulus; do
  for e in free sparse cluttered; do
    f="$SRC/demo_${b}_${e}_N50_900s_5x.mp4"
    [ -s "$f" ] || { echo "missing source clip: $(basename "$f")" >&2; missing=1; }
  done
done
[ "$missing" -eq 0 ] || { echo "run run_videos.sh first" >&2; exit 1; }

echo "host=$(hostname) start=$(date '+%F %T')"
python behaviors_src/make_row_montages.py "$@"
rc=$?
echo "end=$(date '+%F %T') rc=$rc"
exit "$rc"
