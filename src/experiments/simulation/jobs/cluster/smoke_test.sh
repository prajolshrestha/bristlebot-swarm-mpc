#!/bin/bash -l
#SBATCH --job-name=bbsmoke
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# End-to-end smoke test of the experiments tree. Proves the wiring, not the
# science: every variant's solver loads and steps, then the analysis chain runs
# from campaign through aggregate to figure and table.
#
#   sbatch src/experiments/simulation/jobs/cluster/smoke_test.sh
#
# Each variant is checked in a FRESH SUBPROCESS. This is mandatory, not tidiness:
# every variant ships a package called expert_src, so importing a second variant
# in the same interpreter silently reuses the first one's controller from
# sys.modules and fails with a confusing ImportError. The real harness
# (auto_benchmarking_parallel.sim_worker_task) forks per run for the same reason.

unset SLURM_EXPORT_ENV
set -uo pipefail

# Slurm resolves the relative --output path above against the submission
# directory, creating it if needed, so submitting from the wrong place does not
# fail -- it just scatters a stray results/logs/ there. Checking for results/logs
# cannot catch that (Slurm has already made it); check for a marker that only
# exists in src/experiments/simulation instead.
_sd="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -d "$_sd/behaviors_src" ] || [ ! -d "$_sd/comparison_src" ]; then
  echo "submitted from $_sd, expected src/experiments/simulation --" >&2
  echo "the relative --output path above would scatter logs there instead" >&2
  exit 1
fi

module add python 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acados_casadi
# Locate the repository without hardcoding a path: walk up from wherever the job
# was submitted until env.sh appears. Override with BB_REPO if you submit from
# outside the tree. SBATCH directives above cannot use variables, so their log
# paths are relative and land in the submit directory.
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
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
E="$REPO/src/experiments/simulation"
FILT='ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration|SQP_RTI'
rc_all=0

cd "$E/comparison_src"

echo "### 1. every variant solver loads and steps (fresh process each)"
cat > /tmp/_bbsmoke_worker.py <<'PY'
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
python - <<'PY' > /tmp/_bbsmoke_variants.txt
import os, sys
sys.path.insert(0, os.getcwd())
import auto_benchmarking_parallel as ABP
for folder, label in ABP.SAFETY_LEVEL_FOLDERS:
    print(f"{os.path.join(ABP.ROOT_DIR, folder)}\t{label}")
PY
while IFS=$'\t' read -r vdir label; do
  out=$(python /tmp/_bbsmoke_worker.py "$vdir" "$label" 2>&1 | grep -vE "$FILT")
  if echo "$out" | grep -q "   ok "; then
    echo "$out"
  else
    echo "   FAIL    $label"
    echo "$out" | tail -2 | sed 's/^/           /'
    rc_all=1
  fi
done < /tmp/_bbsmoke_variants.txt

echo; echo "### 2. one run through the campaign driver"
# One task per variant, not an aggregate: --aggregate --allow-partial simulates
# every missing run in the 420-task grid, which turns a smoke test into a
# campaign. jobs/local/run_bench.sh and jobs/cluster/run_bench.sh are for the
# campaign itself.
python bench_wrapper.py --task-id 0 --steps 30 --behavior phototaxis \
       --variants "Hard BF,Hard Dt-CBF,Slack BF,Slack Dt-CBF,LA Slack Dt-CBF,Slack Dt-HOCBF" \
       --results-dir "$E/results/raw/bench_900s" 2>&1 | grep -vE "$FILT" | tail -1 || rc_all=1

echo; echo "### 3. tables"
python make_paper_table.py > /tmp/_tab1.txt 2>&1 || rc_all=1
if grep -qE "^\\\\|&" /tmp/_tab1.txt; then echo "   ok      comparison table ($(wc -l < /tmp/_tab1.txt) lines)"
else echo "   FAIL    comparison table"; tail -2 /tmp/_tab1.txt | sed 's/^/           /'; rc_all=1; fi
python swarm_occupancy_table.py > /tmp/_tab2.txt 2>&1 || rc_all=1
if grep -q "begin{table}" /tmp/_tab2.txt; then echo "   ok      occupancy table ($(wc -l < /tmp/_tab2.txt) lines)"
else echo "   FAIL    occupancy table"; tail -2 /tmp/_tab2.txt | sed 's/^/           /'; rc_all=1; fi
rm -f /tmp/_tab1.txt /tmp/_tab2.txt

echo; echo "### 4. artefacts, by destination"
for d in csv figures; do
  echo "   results/$d/"
  find "$E/results/$d" -type f 2>/dev/null | sed "s|$E/results/$d/|      |" | sort
done
echo "   results/raw/  ($(find "$E/results/raw" -type f 2>/dev/null | wc -l) files: checkpoints and per-run traces, gitignored)"

rm -f /tmp/_bbsmoke_worker.py /tmp/_bbsmoke_variants.txt
echo "=== SMOKE rc=$rc_all ==="
exit "$rc_all"
