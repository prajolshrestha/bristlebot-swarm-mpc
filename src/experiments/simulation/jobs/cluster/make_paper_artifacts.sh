#!/bin/bash -l
#SBATCH --job-name=bbfigs
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#
# Aggregate the finished campaigns into the CSVs the paper's figures and tables
# are computed from, then build the two tables.
#
#   sbatch src/experiments/simulation/jobs/cluster/make_paper_artifacts.sh
#   sbatch src/experiments/simulation/jobs/cluster/run_figures.sh
#
# Figures are the other script's job. Keeping them apart matters because the
# aggregation reads results/raw/ and takes minutes, while the figures read only
# the aggregated CSVs and take seconds, so they can be re-rendered freely after
# a styling change without touching the data.
#
# Requires the four campaigns to have drained. Safe to re-run: each step
# recomputes from the CSVs on disk and overwrites its own output.

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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
E="$REPO/src/experiments/simulation"
FILT='ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration|SQP_RTI'
VARIANTS="Hard BF,Hard Dt-CBF,Slack BF,Slack Dt-CBF,LA Slack Dt-CBF,Slack Dt-HOCBF"
rc=0
step () { echo; echo "### $*"; }

cd "$E/comparison_src"

step "1/4 bench aggregate -> results/csv"
python bench_wrapper.py --aggregate --behavior phototaxis --variants "$VARIANTS" \
       --results-dir ../results/raw/bench_900s 2>&1 | grep -vE "$FILT" | tail -4 || rc=1

step "2/4 distance merge -> results/csv"
DIST_RESULTS_DIR="$E/results/raw/distance_900s" \
  python distance_wrapper.py --merge 2>&1 | grep -vE "$FILT" | tail -3 || rc=1

step "3/4 matrix summary"
python matrix_figures.py --summary --variant 05 2>&1 | grep -vE "$FILT" | tail -3 || rc=1

step "4/4 tables"
mkdir -p "$E/results/tables"
python make_paper_table.py > "$E/results/tables/table2_safety_comparison.tex" 2>/dev/null \
  && echo "   table2_safety_comparison.tex ($(wc -l < "$E/results/tables/table2_safety_comparison.tex") lines)" || rc=1
python swarm_occupancy_table.py > "$E/results/tables/table_occupancy.tex" 2>/dev/null \
  && echo "   table_occupancy.tex ($(wc -l < "$E/results/tables/table_occupancy.tex") lines)" || rc=1

step "verifying the aggregates the figures depend on"
miss=0
for f in phototaxis_metrics_vs_density.csv phototaxis_distances_vs_density.csv; do
  if [ -s "$E/results/csv/$f" ]; then
    echo "   ok      $f ($(($(wc -l < "$E/results/csv/$f") - 1)) rows)"
  else
    echo "   MISSING $f"; miss=$((miss+1))
  fi
done
if [ -d "$E/results/csv/matrix" ]; then
  n=$(find "$E/results/csv/matrix" -name "*.csv" | wc -l)
  echo "   ok      matrix/ ($n csv)"
  [ "$n" -eq 0 ] && miss=$((miss+1))
else
  echo "   MISSING matrix/"; miss=$((miss+1))
fi

step "verifying the tables"
for f in table2_safety_comparison.tex table_occupancy.tex; do
  if [ -s "$E/results/tables/$f" ]; then echo "   ok      $f"
  else echo "   MISSING $f"; miss=$((miss+1)); fi
done

step "figures"
# Built by run_figures.sh, not here. Reported so a run makes the state obvious,
# but never fails this job: the aggregates above are what it is responsible for.
built=0
for f in robot_model safety_formulation_comparison safety_margins \
         behavior_snapshots_free behavior_snapshots_2obs behavior_snapshots_12obs \
         order_nnd_by_behavior behavior_collisions_by_behavior; do
  if [ -f "$E/results/figures/$f.pdf" ]; then built=$((built+1)); else echo "   not built  $f"; fi
done
echo "   $built/8 data-derived figures present"
[ "$built" -lt 8 ] && echo "   run: sbatch jobs/cluster/run_figures.sh"

[ "$miss" -gt 0 ] && rc=1

echo
echo "=== ARTIFACTS rc=$rc missing=$miss ==="
exit "$rc"
