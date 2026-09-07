#!/usr/bin/env bash
# Fig 4 safety margins: same six formulations, obstacles on.
# Cluster equivalent: jobs/cluster/run_distance.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
banner "distance: Fig 4 safety margins"

export DIST_RESULTS_DIR="$E/results/raw/distance_900s"
cd "$CMP"
IDS=$(python "$LOCAL_DIR/_task_ids.py" distance --ns "$NS" --seeds "$SEEDS")
[ -z "$IDS" ] && { echo "no task ids produced; check NS/SEEDS" >&2; exit 1; }
echo "    $(echo "$IDS" | wc -w) runs"

echo "$IDS" | xargs -P "$JOBS" -I{} \
  env PYTHONWARNINGS=ignore python distance_wrapper.py --task-id {} \
      --steps "$STEPS" --seeds "$SEEDS" 2>&1 | grep -vE "$FILT" | grep -E "task|FAILED" || true

python distance_wrapper.py --merge --seeds "$SEEDS" --steps "$STEPS" --allow-partial 2>&1 \
  | grep -vE "$FILT" | tail -3
