#!/bin/bash -l
#SBATCH --job-name=bbvid
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# Supplementary videos: 4 behaviors x 3 obstacle environments at N=50, rendered
# through the deployed LA Slack Dt-CBF viz engine.
#
#   cd src/experiments/simulation
#   sbatch jobs/cluster/run_videos.sh
#
# 900 s of simulation at 5x playback (--fps 50), so each clip is 180 s and stays
# inside Elsevier's 5 minute limit. Obstacle layouts come from seed 0, the same
# seed collect_behavior_obstacle_snapshots.py uses, so the arenas match Fig. 5
# and the appendix figures exactly.
#
# Skip-if-exists: a task whose MP4 is already on disk exits immediately, so a
# partially failed array can be refilled by resubmitting the same script.

unset SLURM_EXPORT_ENV
set -uo pipefail

_sd="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -d "$_sd/behaviors_src" ] || [ ! -d "$_sd/comparison_src" ]; then
  echo "submitted from $_sd, expected src/experiments/simulation --" >&2
  echo "the relative --output path above would scatter logs there instead" >&2
  exit 1
fi

module add python 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acados_casadi

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

# matplotlib writes MP4 through an "ffmpeg" on PATH. None is installed here, but
# imageio_ffmpeg ships a static binary; expose it under the expected name.
# Without this matplotlib silently falls back to the Pillow writer, which cannot
# write .mp4 and fails only after the whole simulation has run.
FFMPEG_BIN="$(python -c 'import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())')"
if [ ! -x "$FFMPEG_BIN" ]; then echo "no ffmpeg from imageio_ffmpeg" >&2; exit 1; fi
mkdir -p "${TMPDIR:-/tmp}/bin.$$" && ln -sf "$FFMPEG_BIN" "${TMPDIR:-/tmp}/bin.$$/ffmpeg"
export PATH="${TMPDIR:-/tmp}/bin.$$:$PATH"

BEHAVIORS=(phototaxis no_reynolds orbital no_stimulus)
ENVS=(free sparse cluttered)
i="${SLURM_ARRAY_TASK_ID:-0}"
B="${BEHAVIORS[$((i / 3))]}"
E="${ENVS[$((i % 3))]}"

# VID_DURATION lets a smoke run render a few seconds instead of the full 900 s.
# The duration is part of the output name, so smoke clips never collide with the
# real ones and the skip-if-exists check stays honest.
DUR="${VID_DURATION:-900}"
VARIANT="$REPO/src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead"
OUT="$REPO/src/experiments/simulation/results/videos/random_initial_position/original/demo_${B}_${E}_N50_${DUR}s_5x.mp4"
if [ -s "$OUT" ]; then
  echo "task $i: $(basename "$OUT") already present, skipping"
  exit 0
fi

cd "$VARIANT/sim_engine"
echo "host=$(hostname) task=$i behavior=$B env=$E start=$(date '+%F %T')"

python make_demo_videos.py --duration "$DUR" --fps 50 --speed 5 --n 50 \
    --behavior "$B" --env "$E" 2>&1 \
  | grep -vE 'ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration'
rc=${PIPESTATUS[0]}

echo "end=$(date '+%F %T') rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
[ -s "$OUT" ] || { echo "expected $OUT, not written or empty" >&2; rm -f "$OUT"; exit 1; }
ls -la "$OUT"
