# Shared setup for the local runners. Sourced, not executed.
#
# The cluster jobs in jobs/cluster/ need three things a workstation does
# not have: `module add`, $SLURM_ARRAY_TASK_ID, and `srun`. Everything below
# them is plain Python that takes explicit arguments, so the array is replaced
# by a shell loop and nothing else changes.

set -uo pipefail

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute: the
# runners cd into comparison_src, after which a relative dirname would point
# at the wrong directory and helper lookups would silently return nothing.
REPO="$(cd "$LOCAL_DIR/../../../../.." && pwd)"
E="$REPO/src/experiments/simulation"
CMP="$E/comparison_src"
BEH="$E/behaviors_src"

# conda, if the caller has not already activated the environment
if [ -z "${CONDA_PREFIX:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${ENV_NAME:-acados_casadi}"
    else
        echo "conda not found; activate the environment first" >&2; exit 1
    fi
fi
source "$REPO/env.sh"

# One solve per process, so the parallelism below is the only parallelism.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

# Reduced protocol by default: a full 900 s x 10-seed x 7-density campaign is
# ~1400 CPU-hours, which is about a week on eight cores. Override any of these
# to move toward the published protocol.
STEPS="${STEPS:-900}"                       # 90 s runs; the paper uses 9000
SEEDS="${SEEDS:-3}"                         # the paper uses 10
NS="${NS:-10 50 137}"                       # the paper uses 10 25 50 75 100 125 137
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VARIANT="${VARIANT:-05}"

FILT='ACADOS_MINSTEP|compiled without OpenMP|QP solver returned error|QP iteration|SQP_RTI'

banner () {
    echo "=== $* ==="
    echo "    steps=$STEPS  seeds=$SEEDS  N={$NS}  parallel=$JOBS"
    [ "$STEPS" -lt 9000 ] && echo "    NOTE: reduced protocol, shapes reproduce but numbers will not match the paper"
    return 0
}
