#!/usr/bin/env bash
# Figs 5, A.1, A.2: behavior snapshot sheets. Only 12 simulations, so this one
# is affordable at the full 900 s protocol even on a laptop.
# Cluster equivalent: jobs/cluster/run_snapshots.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export SNAP_TIMES="${SNAP_TIMES:-0,225,450,675,900}"
banner "snapshots: Figs 5, A.1, A.2"

cd "$BEH"
python collect_behavior_obstacle_snapshots.py --workers "$JOBS" \
    --behaviors phototaxis phototaxis_no_reynolds phototaxis_orbital no_light \
    --n 50 2>&1 | grep -vE "$FILT" | tail -3
python render_obstacle_grouped_figs.py 2>&1 | tail -3
