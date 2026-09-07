#!/usr/bin/env python3
"""Sweeps the communication radius and neighbour cap."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import importlib.util
import io
import json
import multiprocessing
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

import os

_CMP_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "comparison_src"))
if _CMP_SRC not in sys.path:
    sys.path.insert(0, _CMP_SRC)
from auto_benchmarking_parallel import SAFETY_LEVEL_FOLDERS
from comparison_config import CASES, DT, STEPS, STEADY_STATE_STEPS

SWEEP_LOCK = Path("/tmp/dmpc_comm_sweep.lock")
RESULTS_CSV = SCRIPT_DIR / "results_sweep" / "comm_k_r_sweep.csv"

D_SAFE_M = 0.07
HEADROOM = 30
DEFAULT_K = [3, 5, 8, 10]
DEFAULT_R = [0.12, 0.15, 0.18, 0.21]

VARIANT_DIRS = [
    ROOT_DIR / "src/deterministic_bristlebot_swarm/01_hard_bf",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/02_hard_dt_cbf",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/03_slacked_bf",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/04_slacked_dt_cbf",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/06_slacked_dt_hocbf",
]

COLLISION_SIMS = [
    v / "sim_engine/headless_sim_engine.py"
    for v in VARIANT_DIRS
]


def _n_list() -> list[int]:
    from occupancy_config import benchmark_n_list
    return benchmark_n_list(d_safe_cm=7.0, headroom=HEADROOM)


def _snapshot() -> dict[str, str]:
    snap: dict[str, str] = {}
    occ = SCRIPT_DIR / "occupancy_config.py"
    snap[str(occ)] = occ.read_text()
    for vdir in VARIANT_DIRS:
        yp = vdir / "expert_src/config/solver_ellipse_mpc.yaml"
        if yp.exists():
            snap[str(yp)] = yp.read_text()
    for sp in COLLISION_SIMS:
        if sp.exists():
            snap[str(sp)] = sp.read_text()
    return snap


def _restore(snap: dict[str, str]) -> None:
    for path, text in snap.items():
        Path(path).write_text(text)


def _patch_d_safe() -> None:
    for vdir in VARIANT_DIRS:
        yp = vdir / "expert_src/config/solver_ellipse_mpc.yaml"
        jp = vdir / "expert_src/lib/acados_ellipse_tracking_mpc_solver_config.json"
        if yp.exists():
            data = yaml.safe_load(yp.read_text())
            data.setdefault("collision", {})["d_safe"] = D_SAFE_M
            data.setdefault("obstacle_avoidance", {})["d_safe_obs"] = D_SAFE_M
            yp.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        if jp.exists():
            cfg = json.loads(jp.read_text())
            tc = cfg.setdefault("trajectory_coupled_dmpc", {})
            tc["d_safe"] = D_SAFE_M
            tc["d_safe_obs"] = D_SAFE_M
            jp.write_text(json.dumps(cfg, indent=2))


def _patch_k(k: int) -> None:
    occ = SCRIPT_DIR / "occupancy_config.py"
    text = occ.read_text()
    text = re.sub(r"^MAX_NEIGHBOURS:\s*int\s*=\s*\d+", f"MAX_NEIGHBOURS: int = {k}", text, flags=re.M)
    occ.write_text(text)
    for vdir in VARIANT_DIRS:
        yp = vdir / "expert_src/config/solver_ellipse_mpc.yaml"
        data = yaml.safe_load(yp.read_text())
        data.setdefault("collision", {})["max_neighbours"] = k
        yp.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def _patch_comm_radius(r: float) -> None:
    occ = SCRIPT_DIR / "occupancy_config.py"
    text = occ.read_text()
    text = re.sub(
        r"^COMM_RADIUS_M:\s*float\s*=\s*[\d.]+",
        f"COMM_RADIUS_M: float = {r}",
        text,
        flags=re.M,
    )
    occ.write_text(text)
    pat = re.compile(
        r"^(COMM_RADIUS\s*=\s*)[\d.]+(.*)$",
        re.MULTILINE,
    )
    for sp in COLLISION_SIMS:
        if not sp.exists():
            continue
        t = sp.read_text()
        t = pat.sub(rf"\g<1>{r}\2", t, count=1)
        sp.write_text(t)


def _recompile_all() -> None:
    for vdir in VARIANT_DIRS:
        expert = vdir / "expert_src"
        subprocess.run(
            [sys.executable, "generate_solver_via_acados.py"],
            cwd=str(expert),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )


def _mean_mpc_neighbors(sim, k: int, comm_r: float) -> float:
    """Mean number of neighbours passed to MPC per robot this step."""
    import numpy as np
    counts = []
    n = len(sim.robots)
    for i in range(n):
        pos_i = sim.robots[i].pos
        kk = min(k + 1, n)
        dists, indices = sim.kdtree.query(pos_i, k=kk, distance_upper_bound=comm_r)
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
            dists = [float(dists)]
        c = 0
        for d, j in zip(dists, indices):
            if j != i and j < n and d <= comm_r:
                c += 1
        counts.append(min(c, k))
    return float(sum(counts) / len(counts)) if counts else 0.0


def sim_worker(args):
    folder, label, n_robots, case_name, k, comm_r = args
    folder_path = ROOT_DIR / folder
    sim_engine_dir = folder_path / "sim_engine"
    orig_sys_path = list(sys.path)
    sys.path.insert(0, str(folder_path))
    sys.path.insert(0, str(sim_engine_dir))

    sim_path = sim_engine_dir / "headless_sim_engine.py"
    if not sim_path.is_file():
        sys.path = orig_sys_path
        return {"success": False, "error": "no headless sim"}

    try:
        spec = importlib.util.spec_from_file_location("dynamic_sim_module", str(sim_path))
        module = importlib.util.module_from_spec(spec)
        for key in list(sys.modules.keys()):
            if any(n in key for n in ["dynamic_sim_module", "headless_sim_engine", "swarm_dynamics", "expert_src"]):
                del sys.modules[key]
        spec.loader.exec_module(module)

        module.COMM_RADIUS = comm_r

        collision_history = []
        feasibility_statuses = []
        neighbor_counts = []

        with contextlib.redirect_stdout(io.StringIO()):
            sim = module.HeadlessSim(
                n_robots=n_robots,
                behavior_mode=module.SwarmBehavior(case_name),
                enable_obstacles=False,
                enable_collision=True,
            )
            for step_idx in range(STEPS):
                sim.step()
                collision_history.append(len(sim.active_collisions))
                feasibility_statuses.append(list(sim.solver_statuses))
                neighbor_counts.append(_mean_mpc_neighbors(sim, k, comm_r))

        import numpy as np
        all_times = [t for r_times in sim.solve_times for t in r_times]
        feas = np.array(feasibility_statuses)
        rate = 100.0 * np.sum(feas == 0) / feas.size if feas.size else 0.0

        sys.path = orig_sys_path
        return {
            "success": True,
            "label": label,
            "n_robots": n_robots,
            "case_name": case_name,
            "k": k,
            "comm_radius_m": comm_r,
            "collision_frames": int(sum(collision_history)),
            "feasibility_rate_pct": float(rate),
            "mean_solve_time_us": float(np.mean(all_times)) if all_times else 0.0,
            "mean_mpc_neighbors": float(np.mean(neighbor_counts)),
            "slot_fill_pct": 100.0 * float(np.mean(neighbor_counts)) / min(k, max(1, n_robots - 1)),
        }
    except Exception as e:
        sys.path = orig_sys_path
        return {"success": False, "error": str(e), "label": label, "k": k, "comm_radius_m": comm_r}


def _run_grid(k_values: list[int], r_values: list[float], workers: int) -> list[dict]:
    n_list = _n_list()
    print(f"N_LIST = {n_list}", flush=True)
    all_rows: list[dict] = []

    for k in k_values:
        print(f"\n=== K = {k} (recompiling solvers) ===", flush=True)
        _patch_k(k)
        _recompile_all()

        for r in r_values:
            print(f"  R = {r:.2f} m", flush=True)
            _patch_comm_radius(r)
            tasks = [
                (folder, label, n, case["name"], k, r)
                for case in CASES
                for folder, label in SAFETY_LEVEL_FOLDERS
                for n in n_list
            ]
            t0 = time.time()
            with multiprocessing.Pool(processes=workers) as pool:
                for res in pool.imap_unordered(sim_worker, tasks):
                    if res.get("success"):
                        res["n_operating"] = n_list[-1]
                        res["is_operating_max"] = int(res["n_robots"] == n_list[-1])
                        all_rows.append(res)
                    else:
                        print(f"    FAIL {res}", flush=True)
            print(f"    {len(tasks)} tasks in {time.time()-t0:.1f}s", flush=True)

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K)
    parser.add_argument("--r-values", type=float, nargs="+", default=DEFAULT_R)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 4))
    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_CSV,
        help="Output CSV path (default: results_sweep/comm_k_r_sweep.csv)",
    )
    args = parser.parse_args()

    results_csv = args.results_file if args.results_file.is_absolute() else SCRIPT_DIR / args.results_file
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_LOCK.open("w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        snap = _snapshot()
        try:
            _patch_d_safe()
            rows = _run_grid(args.k_values, args.r_values, args.workers)
            if rows:
                with results_csv.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                print(f"\nSaved {len(rows)} rows → {results_csv}", flush=True)
        finally:
            _restore(snap)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
