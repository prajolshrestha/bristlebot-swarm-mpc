#!/usr/bin/env python3
"""Sweeps the safety distance and the movement headroom."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))


_CMP_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "comparison_src"))
if _CMP_SRC not in sys.path:
    sys.path.insert(0, _CMP_SRC)
from comparison_config import CASES, DT, STEPS, STEADY_STATE_STEPS
from occupancy_config import (
    GEOMETRIC_N_MAX,
    MAX_NEIGHBOURS,
    benchmark_n_list,
    buffer_area_cm2,
    geometric_n_max,
    occupancy_pct,
)

from auto_benchmarking_parallel import SAFETY_LEVEL_FOLDERS, sim_worker_task

D_SAFE_VALUES_CM = sorted(GEOMETRIC_N_MAX.keys())
DEFAULT_HEADROOMS = [27, 30, 37, 50]
SWEEP_LOCK = Path("/tmp/dmpc_occupancy_sweep.lock")
RESULTS_ROOT = SCRIPT_DIR / "results_sweep"

VARIANT_DIRS = [
    ROOT_DIR / "src/deterministic_bristlebot_swarm/01_hard_bf/expert_src",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/02_hard_dt_cbf/expert_src",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/03_slacked_bf/expert_src",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/04_slacked_dt_cbf/expert_src",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead/expert_src",
    ROOT_DIR / "src/deterministic_bristlebot_swarm/06_slacked_dt_hocbf/expert_src",
]


def _yaml_path(expert_src: Path) -> Path:
    return expert_src / "config/solver_ellipse_mpc.yaml"


def _json_path(expert_src: Path) -> Path:
    return expert_src / "lib/acados_ellipse_tracking_mpc_solver_config.json"


def _read_text(path: Path) -> str:
    return path.read_text()


def _snapshot_configs() -> dict[str, str]:
    snap: dict[str, str] = {}
    for expert in VARIANT_DIRS:
        for p in (_yaml_path(expert), _json_path(expert)):
            if p.exists():
                snap[str(p)] = _read_text(p)
    return snap


def _restore_configs(snap: dict[str, str]) -> None:
    for path_str, text in snap.items():
        Path(path_str).write_text(text)


def _patch_d_safe(d_safe_m: float) -> None:
    for expert in VARIANT_DIRS:
        yaml_path = _yaml_path(expert)
        if yaml_path.exists():
            data = yaml.safe_load(yaml_path.read_text())
            data.setdefault("collision", {})["d_safe"] = d_safe_m
            data.setdefault("obstacle_avoidance", {})["d_safe_obs"] = d_safe_m
            yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

        json_path = _json_path(expert)
        if json_path.exists():
            cfg = json.loads(json_path.read_text())
            tc = cfg.setdefault("trajectory_coupled_dmpc", {})
            tc["d_safe"] = d_safe_m
            tc["d_safe_obs"] = d_safe_m
            json_path.write_text(json.dumps(cfg, indent=2))


def _collision_total(history: list[int]) -> int:
    return int(sum(history))


def _run_parallel(tasks: list[tuple], workers: int) -> list[dict]:
    results: list[dict] = []
    with multiprocessing.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(sim_worker_task, tasks):
            if res.get("success"):
                results.append(res)
            else:
                print(
                    f"  [FAIL] {res.get('label','?')} N={res.get('n_robots','?')} "
                    f"{res.get('case_name','?')}: {res.get('error','unknown')}",
                    flush=True,
                )
    return results


def _mean_neighbors_estimate(n_robots: int) -> float:
    """Upper-bound comm load proxy: min(K, N-1) averaged over swarm."""
    return float(min(MAX_NEIGHBOURS, max(0, n_robots - 1)))


def _run_headroom(
    d_safe_cm: float,
    headroom: int,
    workers: int,
) -> list[dict]:
    d_safe_m = d_safe_cm / 100.0
    n_geo = geometric_n_max(d_safe_cm)
    n_list = benchmark_n_list(d_safe_cm=d_safe_cm, headroom=headroom)
    n_operating = n_list[-1]
    buf_occ = occupancy_pct(n_operating * buffer_area_cm2(d_safe_cm))

    print(
        f"\n  headroom={headroom:2d}  N_geo={n_geo:3d}  N_operating={n_operating:3d}  "
        f"buffer_occ={buf_occ:.1f}%  N_LIST={n_list}",
        flush=True,
    )

    tasks = []
    for case in CASES:
        for folder, label in SAFETY_LEVEL_FOLDERS:
            for n in n_list:
                tasks.append((folder, label, n, case["name"]))

    t0 = time.time()
    results = _run_parallel(tasks, workers)
    elapsed = time.time() - t0
    print(f"  completed {len(results)}/{len(tasks)} tasks in {elapsed:.1f}s", flush=True)

    rows: list[dict] = []
    for res in results:
        n = res["n_robots"]
        rows.append({
            "d_safe_cm": d_safe_cm,
            "headroom": headroom,
            "n_geo": n_geo,
            "n_operating": n_operating,
            "buffer_occ_pct": round(buf_occ, 2),
            "n_robots": n,
            "safety_level": res["label"],
            "case_name": res["case_name"],
            "collision_frames": int(res.get("collisions_events", _collision_total(res["collision_history"]))),
            "feasibility_rate_pct": round(res["feasibility_rate"], 2),
            "mean_solve_time_us": round(res["mean_us"], 2),
            "max_solve_time_ms": round(res["max_ms"], 4),
            "mean_neighbors": _mean_neighbors_estimate(n),
            "mean_target_distance_m": round(res["mean_distance"], 4),
            "mean_polarization": round(res["mean_polarization"], 4),
            "mean_jerk": round(res["mean_jerk"], 4),
            "is_operating_max": int(n == n_operating),
        })
    return rows


def _write_rows(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def run_d_safe_sweep(
    d_safe_cm: float,
    headrooms: list[int],
    workers: int,
) -> Path:
    out_csv = RESULTS_ROOT / f"dsafe_{int(d_safe_cm)}cm.csv"
    print("=" * 72, flush=True)
    print(f"  D_safe = {d_safe_cm:.0f} cm  |  headrooms = {headrooms}", flush=True)
    print("=" * 72, flush=True)

    snap = _snapshot_configs()
    try:
        _patch_d_safe(d_safe_cm / 100.0)
        all_rows: list[dict] = []
        for headroom in headrooms:
            all_rows.extend(_run_headroom(d_safe_cm, headroom, workers))
        if all_rows:
            if out_csv.exists():
                out_csv.unlink()
            _write_rows(out_csv, all_rows)
        print(f"  → saved {len(all_rows)} rows to {out_csv}", flush=True)
        return out_csv
    finally:
        _restore_configs(snap)


def main() -> None:
    parser = argparse.ArgumentParser(description="D_safe × headroom occupancy sweep")
    parser.add_argument("--d-safe-cm", type=float, default=None, help="Single D_safe [cm]")
    parser.add_argument("--all-d-safe", action="store_true", help="Sweep all table D_safe values")
    parser.add_argument(
        "--headrooms", type=int, nargs="+", default=DEFAULT_HEADROOMS,
        help=f"Movement headroom values (default: {DEFAULT_HEADROOMS})",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, multiprocessing.cpu_count() - 4),
        help="Parallel simulation workers",
    )
    args = parser.parse_args()

    if args.all_d_safe:
        d_safe_list = D_SAFE_VALUES_CM
    elif args.d_safe_cm is not None:
        d_safe_list = [args.d_safe_cm]
    else:
        parser.error("Specify --d-safe-cm or --all-d-safe")

    SWEEP_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_LOCK.open("w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        for d_safe_cm in d_safe_list:
            run_d_safe_sweep(d_safe_cm, args.headrooms, args.workers)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
