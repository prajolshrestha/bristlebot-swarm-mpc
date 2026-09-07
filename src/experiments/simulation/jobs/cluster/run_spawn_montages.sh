#!/bin/bash -l
#SBATCH --job-name=bbspwmon
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%j.out
#
# 2x2 montages of the door-spawn clips, matching random_initial_position/montage_180s:
# 2 s title card, first 180 s of each 900 s run at 5x, 2400x2700, 50 fps, CRF 24.
#
#   cd src/experiments/simulation
#   sbatch jobs/cluster/run_spawn_montages.sh
#
# Transcode only: it windows the existing 900 s clips, nothing is re-simulated.

unset SLURM_EXPORT_ENV
set -uo pipefail

_sd="${SLURM_SUBMIT_DIR:-$PWD}"
[ -d "$_sd/behaviors_src" ] || { echo "submit from src/experiments/simulation" >&2; exit 1; }
cd "$_sd" || exit 1

module add python 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acados_casadi
export MPLBACKEND=Agg

CLIPS="results/videos/spawn/original"
OUT="results/videos/spawn/montage_180s"
missing=0
for b in phototaxis no_reynolds orbital no_stimulus; do
  for e in free sparse cluttered; do
    f="$CLIPS/spawn_${b}_${e}_N137_900s_5x.mp4"
    [ -s "$f" ] || { echo "missing $(basename "$f")" >&2; missing=1; }
  done
done
[ "$missing" -eq 0 ] || { echo "run run_spawn_videos.sh first" >&2; exit 1; }

mkdir -p "$OUT"
echo "host=$(hostname) start=$(date '+%F %T')"
python behaviors_src/make_video_montages.py \
    --clip-dir "$CLIPS" --prefix spawn --n-robots 137 \
    --duration 900 --show-seconds 180 --speed 5 --fps 50 --crf 24 \
    --entry "Robots enter one at a time through the bottom-left corner." \
    --outdir "$OUT"
rc=$?
echo "end=$(date '+%F %T') rc=$rc"
[ "$rc" -eq 0 ] && ls -la "$OUT"
exit "$rc"
