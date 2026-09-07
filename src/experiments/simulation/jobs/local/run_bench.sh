#!/usr/bin/env bash
# Fig 3 + comparison table: six formulations under phototaxis, free space.
# Cluster equivalent: jobs/cluster/run_bench.sh
#
#   ./run_bench.sh                       # reduced protocol
#   STEPS=9000 SEEDS=10 NS="10 25 50 75 100 125 137" ./run_bench.sh   # full
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
banner "bench: Fig 3 + comparison table"

VARIANTS="Hard BF,Hard Dt-CBF,Slack BF,Slack Dt-CBF,LA Slack Dt-CBF,Slack Dt-HOCBF"
RES="$E/results/raw/bench_900s"
cd "$CMP"

# --task-id indexes the FULL grid, so a subset needs its ids computed, not a
# seq. See _task_ids.py; a plain seq would rerun one variant instead of slicing
# across all six.
IDS=$(python "$LOCAL_DIR/_task_ids.py" bench --ns "$NS" --seeds "$SEEDS")
[ -z "$IDS" ] && { echo "no task ids produced; check NS/SEEDS" >&2; exit 1; }
echo "    $(echo "$IDS" | wc -w) runs"

echo "$IDS" | xargs -P "$JOBS" -I{} \
  env PYTHONWARNINGS=ignore python bench_wrapper.py --task-id {} \
      --steps "$STEPS" --behavior phototaxis --variants "$VARIANTS" \
      --results-dir "$RES" 2>&1 | grep -vE "$FILT" | grep -E "task|FAILED" || true

python bench_wrapper.py --aggregate --behavior phototaxis --variants "$VARIANTS" \
       --steps "$STEPS" --results-dir "$RES" --allow-partial 2>&1 | grep -vE "$FILT" | tail -3
python paper_comparison_figure.py --case phototaxis --res-dir "$RES" 2>&1 | tail -2
