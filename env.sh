# Source this after setup:  source env.sh
#
# acados is built inside the conda environment, so activating the environment is
# normally enough (its activate.d hook exports these). This file makes the same
# exports explicit for shells and batch jobs that do not run the hook.
#
# setup.sh rewrites this file with the location it installed acados to. Override
# for a different checkout:
#   ACADOS_SOURCE_DIR=/path/to/acados source env.sh

if [ -z "${ACADOS_SOURCE_DIR:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/acados" ]; then
    export ACADOS_SOURCE_DIR="$CONDA_PREFIX/acados"
  elif [ -d "$HOME/acados" ]; then
    export ACADOS_SOURCE_DIR="$HOME/acados"
  else
    echo "env.sh: cannot find acados. Activate the conda env first, or set" >&2
    echo "        ACADOS_SOURCE_DIR to your acados checkout." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/lib64:${LD_LIBRARY_PATH:-}"
