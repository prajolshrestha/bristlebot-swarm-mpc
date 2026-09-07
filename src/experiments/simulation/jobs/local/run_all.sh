#!/usr/bin/env bash
# Everything, in order, then the tables. See README for the runtime warning.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
for s in run_bench.sh run_distance.sh run_matrix.sh run_snapshots.sh; do
    echo; bash "$D/$s" || echo "!! $s returned non-zero"
done
echo; echo "=== tables ==="
mkdir -p "$E/results/tables"
cd "$CMP"
python make_paper_table.py     > "$E/results/tables/table2_safety_comparison.tex" && echo "  table2_safety_comparison.tex"
python swarm_occupancy_table.py > "$E/results/tables/table_occupancy.tex"        && echo "  table_occupancy.tex"
echo; echo "=== figures ==="
find "$E/results/figures" -name "*.pdf" | sed "s|$E/results/figures/|  |" | sort
