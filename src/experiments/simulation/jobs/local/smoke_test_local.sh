#!/usr/bin/env bash
# Local equivalent of ../smoke_test.sh: proves the wiring without Slurm.
#
#   ./smoke_test_local.sh          # a couple of minutes
#
# Four checks, all cheap:
#   1. every variant's solver loads and steps, each in a FRESH process
#   2. one short simulation through the campaign driver
#   3. both table generators emit LaTeX
#   4. every figure script imports and resolves its data paths
#
# It does NOT run the bench aggregate. `--aggregate --allow-partial` does not
# skip missing runs, it SIMULATES them in-process, so on a laptop it would
# quietly start the whole 420-run grid. Aggregation belongs to run_bench.sh,
# after the campaign has actually produced its checkpoints.
#
# The fresh process per variant is mandatory, not tidiness: every variant ships
# a package called expert_src, so importing a second one in the same interpreter
# silently reuses the first variant's controller and fails confusingly.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rc=0
echo "=== local smoke test ==="

echo
echo "### 1. every variant solver loads and steps"
cat > /tmp/_smoke_local_worker.py <<'PY'
import contextlib, importlib.util, io, os, sys
import numpy as np
vdir, label = sys.argv[1], sys.argv[2]
sys.path.insert(0, vdir)
with contextlib.redirect_stdout(io.StringIO()):
    spec = importlib.util.spec_from_file_location(
        "eng", os.path.join(vdir, "sim_engine", "headless_sim_engine.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.SPAWN_SEED = 0; np.random.seed(0)
    sim = m.HeadlessSim(n_robots=10, behavior_mode=m.SwarmBehavior("phototaxis"),
                        enable_obstacles=False, enable_collision=True)
    for _ in range(30):
        sim.step()
print(f"   ok      {label:<14} 30 steps, collisions={sim.total_collisions}")
PY
cd "$CMP"
python - > /tmp/_smoke_local_variants.txt <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import auto_benchmarking_parallel as ABP
for folder, label in ABP.SAFETY_LEVEL_FOLDERS:
    print(f"{os.path.join(ABP.ROOT_DIR, folder)}\t{label}")
PY
while IFS=$'\t' read -r vdir label; do
    out=$(python /tmp/_smoke_local_worker.py "$vdir" "$label" 2>&1 | grep -vE "$FILT")
    if echo "$out" | grep -q "   ok "; then echo "$out"
    else echo "   FAIL    $label"; echo "$out" | tail -2 | sed 's/^/           /'; rc=1; fi
done < /tmp/_smoke_local_variants.txt

echo
echo "### 2. one run through the campaign driver"
RES="$E/results/raw/smoke_local"
rm -rf "$RES"
if python bench_wrapper.py --task-id 0 --steps 30 --results-dir "$RES" \
        --behavior phototaxis 2>&1 | grep -vE "$FILT" | grep -q "OK ->"; then
    echo "   ok      bench_wrapper wrote a checkpoint"
else
    echo "   FAIL    bench_wrapper"; rc=1
fi
rm -rf "$RES"

echo
echo "### 3. tables"
python make_paper_table.py > /tmp/_t1.txt 2>&1
grep -qE '^\\\\|&' /tmp/_t1.txt && echo "   ok      comparison table" \
    || { echo "   note    comparison table needs an aggregated CSV (run run_bench.sh first)"; }
python swarm_occupancy_table.py > /tmp/_t2.txt 2>&1
grep -q "begin{table}" /tmp/_t2.txt && echo "   ok      occupancy table" \
    || { echo "   FAIL    occupancy table"; tail -2 /tmp/_t2.txt | sed 's/^/           /'; rc=1; }

echo
echo "### 4. figure scripts import and resolve their paths"
for m in paper_comparison_figure matrix_figures; do
    (cd "$CMP" && python -c "import $m" 2>/dev/null) \
        && echo "   ok      $m" || { echo "   FAIL    $m"; rc=1; }
done
for m in fig_coordination render_obstacle_grouped_figs collect_behavior_obstacle_snapshots; do
    (cd "$BEH" && python -c "import $m" 2>/dev/null) \
        && echo "   ok      $m" || { echo "   FAIL    $m"; rc=1; }
done

rm -f /tmp/_smoke_local_worker.py /tmp/_smoke_local_variants.txt /tmp/_t1.txt /tmp/_t2.txt
echo
echo "=== LOCAL SMOKE rc=$rc ==="
exit "$rc"
