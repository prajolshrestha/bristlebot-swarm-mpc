#!/bin/bash -l
#SBATCH --job-name=bbspawn
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# Door-spawn supplementary videos: 4 behaviors x 3 obstacle environments.
#
#   cd src/experiments/simulation
#   sbatch jobs/cluster/run_spawn_videos.sh
#
# Robots enter one at a time through a door in the bottom wall, one every
# SPAWN_INTERVAL seconds, up to NMAX. --force-spawn means a robot is admitted on
# every interval even when the doorway is still occupied, so the flow never
# stalls behind a jam.
#
# 900 s of simulation rendered at 5x (stride 1 at 50 fps = one frame per control
# step), 1200x1200, CRF 16 -- the same encode as the N50 clips.
#
# The simulation writes an npz first, then renders from it. A task whose MP4 is
# already on disk exits immediately, so a partially failed array can be refilled
# by resubmitting.

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
[ -f "$REPO/env.sh" ] || { echo "cannot locate the repository" >&2; exit 1; }
source "$REPO/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

FFMPEG_BIN="$(python -c 'import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())')"
mkdir -p "${TMPDIR:-/tmp}/bin.$$" && ln -sf "$FFMPEG_BIN" "${TMPDIR:-/tmp}/bin.$$/ffmpeg"
export PATH="${TMPDIR:-/tmp}/bin.$$:$PATH"

BEHAVIORS=(phototaxis no_reynolds orbital no_stimulus)
ENVS=(free sparse cluttered)
i="${SLURM_ARRAY_TASK_ID:-0}"
B="${BEHAVIORS[$((i / 3))]}"
E="${ENVS[$((i % 3))]}"

NMAX="${DOOR_NMAX:-137}"
DUR="${DOOR_DURATION:-900}"
INTERVAL="${DOOR_SPAWN_INTERVAL:-5}"
# corner_low = bottom-left corner. Its entry corridor sits at y = -0.38,
# below the obstacle band (|y| < 0.30), so the doorway stays clear even in
# the cluttered arena.
DOOR="${DOOR_NAME:-corner_low}"
# v_min matches model_bounds.v_min in the variant-05 config. The compiled
# solver already carries lbu = 0.01; this also lifts the controller-side
# clip, which is hardcoded to 0.0, so a jammed robot cannot stall at rest.
VMIN="${DOOR_V_MIN:-0.01}"

SIMDIR="$REPO/src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead/sim_engine"
RUNDIR="$REPO/src/experiments/simulation/results/videos/spawn/spawn_runs_${DOOR}_N${NMAX}_${DUR}s_i${INTERVAL}_v${VMIN}"
OUTDIR="$REPO/src/experiments/simulation/results/videos/spawn/original"
mkdir -p "$RUNDIR" "$OUTDIR"

OUT="$OUTDIR/spawn_${B}_${E}_N${NMAX}_${DUR}s_5x.mp4"
if [ -s "$OUT" ]; then
  echo "task $i: $(basename "$OUT") already present, skipping"
  exit 0
fi

cd "$SIMDIR"
echo "host=$(hostname) task=$i behavior=$B env=$E door=$DOOR nmax=$NMAX interval=${INTERVAL}s start=$(date '+%F %T')"

python spawn_sim.py --behavior "$B" --env "$E" --nmax "$NMAX" \
    --spawn-interval "$INTERVAL" --duration "$DUR" --force-spawn \
    --door "$DOOR" --v-min "$VMIN" --outdir "$RUNDIR" 2>&1 \
  | grep -vE 'ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration'
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "simulation failed rc=$rc" >&2; exit "$rc"; }

python spawn_render.py --behavior "$B" --env "$E" --rundir "$RUNDIR" \
    --outdir "$OUTDIR" --stride 1 --fps 50 --size 10 --dpi 120 --crf 16 --nmax "$NMAX" --hide-door
rc=$?

echo "end=$(date '+%F %T') rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
[ -s "$OUT" ] || { echo "expected $OUT, not written" >&2; rm -f "$OUT"; exit 1; }
ls -la "$OUT"

# The recording is only an intermediate. Keep it with KEEP_NPZ=1 if you expect
# to re-render (a re-render is minutes; re-simulating is hours).
if [ "${KEEP_NPZ:-0}" != "1" ]; then
  rm -f "$RUNDIR/${B}_${E}.npz"
  rmdir "$RUNDIR" 2>/dev/null || true
  echo "removed recording $RUNDIR/${B}_${E}.npz"
fi
