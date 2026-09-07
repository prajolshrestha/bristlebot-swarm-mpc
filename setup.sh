#!/usr/bin/env bash
# setup.sh - bootstrap acados + Python env for bristlebot-swarm-mpc
#
# Usage:
#   ./setup.sh                         # full install (acados + conda env + pip deps)
#   ./setup.sh --compile-solvers       # also build the six OCP solvers used by the paper
#   ./setup.sh --skip-acados           # only (re)create env and install pip packages
#   ./setup.sh --verify                # check tools, env vars, and Python imports
#
# After setup:
#   source env.sh
#   conda activate acados_casadi       # if using the default conda env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACADOS_DIR="${ACADOS_DIR:-$HOME/acados}"
ENV_NAME="${ENV_NAME:-acados_casadi}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SKIP_ACADOS=0
COMPILE_SOLVERS=0
VERIFY_ONLY=0
INSTALL_OPTIONAL=0

usage() {
    sed -n '2,12p' "$0"
    echo
    echo "Options:"
    echo "  --acados-dir PATH     acados source/install root (default: \$HOME/acados)"
    echo "  --env-name NAME       conda env name (default: acados_casadi)"
    echo "  --python-version VER  conda Python version (default: 3.10)"
    echo "  --skip-acados         skip cloning/building acados; only set up Python env"
    echo "  --compile-solvers     compile all generate_solver_via_acados.py targets"
    echo "  --optional            also install requirements-optional.txt"
    echo "  --verify              run checks only; do not install anything"
    echo "  -h, --help            show this help"
}

log() { printf '==> %s\n' "$*"; }
warn() { printf '!! %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --acados-dir) ACADOS_DIR="$2"; shift 2 ;;
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --skip-acados) SKIP_ACADOS=1; shift ;;
        --compile-solvers) COMPILE_SOLVERS=1; shift ;;
        --optional) INSTALL_OPTIONAL=1; shift ;;
        --verify) VERIFY_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) warn "Unknown option: $1"; usage; exit 1 ;;
    esac
done

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        warn "Missing required command: $1"
        return 1
    fi
}

check_system_tools() {
    local missing=0
    for cmd in git cmake make gcc g++; do
        require_cmd "$cmd" || missing=1
    done
    if [[ $missing -ne 0 ]]; then
        warn "Install build tools first. On Ubuntu/WSL:"
        warn "  sudo apt update && sudo apt install -y build-essential cmake git"
        return 1
    fi
}

write_env_sh() {
    cat > "$REPO_ROOT/env.sh" <<EOF
# Source this file after setup:  source env.sh
# Written by setup.sh with the acados location resolved at install time.
# Override for a different checkout:  ACADOS_SOURCE_DIR=/path source env.sh
export ACADOS_SOURCE_DIR="\${ACADOS_SOURCE_DIR:-$ACADOS_DIR}"
export LD_LIBRARY_PATH="\$ACADOS_SOURCE_DIR/lib:\$ACADOS_SOURCE_DIR/lib64:\${LD_LIBRARY_PATH:-}"
EOF
    log "Wrote $REPO_ROOT/env.sh"
}

verify_python_imports() {
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env.sh"
    python - <<'PY'
import importlib
mods = [
    "numpy",
    "scipy",
    "matplotlib",
    "yaml",
    "casadi",
    "acados_template",
]
failed = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {exc}")
if failed:
    raise SystemExit("Import check failed:\n  " + "\n  ".join(failed))
print("All core imports OK.")
PY
}

verify_acados_lib() {
    if [[ ! -f "$ACADOS_DIR/lib/libacados.so" ]]; then
        warn "acados shared library not found at $ACADOS_DIR/lib/libacados.so"
        return 1
    fi
    log "Found acados library at $ACADOS_DIR/lib/libacados.so"
}

verify_repo_solvers() {
    local missing=0 v
    for v in 01_hard_bf 02_hard_dt_cbf 03_slacked_bf 04_slacked_dt_cbf \
             05_slacked_dt_cbf_with_lookahead 06_slacked_dt_hocbf; do
        if compgen -G "$REPO_ROOT/src/deterministic_bristlebot_swarm/$v/expert_src/lib/*.so" > /dev/null; then
            log "Solver present: $v"
        else
            warn "Solver not compiled: $v"
            missing=$((missing + 1))
        fi
    done
    if [[ $missing -gt 0 ]]; then
        warn "$missing of 6 solvers missing (run with --compile-solvers)"
    fi
}

run_verify() {
    # Verification inspects an existing install, so build tools are advisory here:
    # cmake and a compiler are only needed by install_acados. On clusters they
    # usually come from a module, and demanding them would fail a healthy install.
    log "Verifying system tools (build tools advisory)..."
    check_system_tools || warn "build tools missing; only matters if you rebuild acados"
    log "Verifying acados install at $ACADOS_DIR..."
    verify_acados_lib || true
    write_env_sh
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME" 2>/dev/null || warn "Conda env '$ENV_NAME' not found"
    fi
    log "Verifying Python imports..."
    verify_python_imports
    verify_repo_solvers
    log "Verification complete."
}

install_acados() {
    if [[ -d "$ACADOS_DIR/.git" ]]; then
        log "Updating existing acados repo at $ACADOS_DIR"
        git -C "$ACADOS_DIR" pull --ff-only || warn "git pull failed; continuing with existing checkout"
        git -C "$ACADOS_DIR" submodule update --init --recursive
    else
        log "Cloning acados into $ACADOS_DIR"
        git clone https://github.com/acados/acados.git "$ACADOS_DIR"
        git -C "$ACADOS_DIR" submodule update --init --recursive
    fi

    log "Building acados (Release)..."
    mkdir -p "$ACADOS_DIR/build"
    cmake -S "$ACADOS_DIR" -B "$ACADOS_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DACADOS_WITH_QPOASES=ON \
        -DACADOS_WITH_OSQP=ON
    cmake --build "$ACADOS_DIR/build" --target install -j"$(nproc)"
    verify_acados_lib
}

setup_conda_env() {
    if ! command -v conda >/dev/null 2>&1; then
        warn "conda not found. Install Miniconda/Anaconda or create a venv manually:"
        warn "  python${PYTHON_VERSION} -m venv .venv && source .venv/bin/activate"
        warn "  pip install -r requirements.txt"
        warn "  pip install -e $ACADOS_DIR/interfaces/acados_template"
        return 1
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        log "Using existing conda env: $ENV_NAME"
    else
        log "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
        conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
    fi

    conda activate "$ENV_NAME"
    python -m pip install --upgrade pip wheel
}

install_python_packages() {
    write_env_sh
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env.sh"

    log "Installing acados_template (editable) from $ACADOS_DIR"
    python -m pip install -e "$ACADOS_DIR/interfaces/acados_template"

    log "Installing core requirements"
    python -m pip install -r "$REPO_ROOT/requirements.txt"

    if [[ $INSTALL_OPTIONAL -eq 1 ]]; then
        log "Installing optional requirements"
        python -m pip install -r "$REPO_ROOT/requirements-optional.txt"
    fi

    verify_python_imports
}

compile_all_solvers() {
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env.sh"
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    fi

    local generators=(
        "src/deterministic_bristlebot_swarm/01_hard_bf/expert_src/generate_solver_via_acados.py"
        "src/deterministic_bristlebot_swarm/02_hard_dt_cbf/expert_src/generate_solver_via_acados.py"
        "src/deterministic_bristlebot_swarm/03_slacked_bf/expert_src/generate_solver_via_acados.py"
        "src/deterministic_bristlebot_swarm/04_slacked_dt_cbf/expert_src/generate_solver_via_acados.py"
        "src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead/expert_src/generate_solver_via_acados.py"
        "src/deterministic_bristlebot_swarm/06_slacked_dt_hocbf/expert_src/generate_solver_via_acados.py"
    )

    for rel in "${generators[@]}"; do
        local dir="$REPO_ROOT/$(dirname "$rel")"
        log "Compiling solver in $dir"
        (cd "$dir" && python "$(basename "$rel")")
    done
}

main() {
    cd "$REPO_ROOT"

    if [[ $VERIFY_ONLY -eq 1 ]]; then
        run_verify
        exit 0
    fi

    check_system_tools

    if [[ $SKIP_ACADOS -eq 0 ]]; then
        install_acados
    else
        log "Skipping acados build (--skip-acados)"
        if [[ ! -d "$ACADOS_DIR" ]]; then
            warn "ACADOS_DIR does not exist: $ACADOS_DIR"
            exit 1
        fi
    fi

    setup_conda_env
    install_python_packages

    if [[ $COMPILE_SOLVERS -eq 1 ]]; then
        compile_all_solvers
    fi

    cat <<EOF

Setup finished.

  source $REPO_ROOT/env.sh
  conda activate $ENV_NAME

Quick checks:
  sbatch src/experiments/smoke_test.sh        # verifies every solver and the analysis chain
  ./setup.sh --verify                         # re-check tools, env and solvers

If solvers are missing, rerun:  ./setup.sh --skip-acados --compile-solvers
  Reproducing the paper: see src/experiments/README.md

EOF
}

main "$@"
