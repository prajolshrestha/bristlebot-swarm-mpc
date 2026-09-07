#!/usr/bin/env python3
"""Sweeps the navigation look-ahead distance."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

import os

_CMP_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "comparison_src"))
if _CMP_SRC not in sys.path:
    sys.path.insert(0, _CMP_SRC)
from auto_benchmarking_parallel import SAFETY_LEVEL_FOLDERS
from comparison_config import CASES, DT

SWEEP_STEPS = 60
SWEEP_TIME_SEC = SWEEP_STEPS * DT
EARLY_COLLISION_STEPS = 30

LOOKAHEAD_VALUES_M = [0.04, 0.07, 0.10, 0.13, 0.16]
DEFAULT_N_ROBOTS = 25
RESULTS_CSV = SCRIPT_DIR / "results_sweep" / "lookahead_sweep.csv"


def _collision_total(history: list[int]) -> int:
    return int(sum(history))


def _early_collisions(history: list[int], through_step: int) -> int:
    """Sum collision counts over steps [0, through_step)."""
    return int(sum(history[:through_step]))


def sim_worker_task(args: tuple) -> dict:
    folder, label, n_robots, case_name, lookahead_m, steps = args
    folder_path = ROOT_DIR / folder
    sim_engine_dir = folder_path / "sim_engine"

    orig_sys_path = list(sys.path)
    sys.path.insert(0, str(folder_path))
    sys.path.insert(0, str(sim_engine_dir))

    sim_path = sim_engine_dir / "headless_sim_engine.py"
    if not sim_path.is_file():
        sys.path = orig_sys_path
        return {
            "success": False,
            "error": "No headless sim engine found",
            "label": label,
            "n_robots": n_robots,
            "case_name": case_name,
            "lookahead_m": lookahead_m,
        }

    try:
        spec = importlib.util.spec_from_file_location("dynamic_sim_module", str(sim_path))
        module = importlib.util.module_from_spec(spec)

        for key in list(sys.modules.keys()):
            if any(
                name in key
                for name in [
                    "dynamic_sim_module",
                    "headless_sim_engine",
                    "swarm_dynamics",
                    "light_source_model",
                    "obstacle_model",
                    "collision_response",
                    "expert_src",
                ]
            ):
                del sys.modules[key]

        spec.loader.exec_module(module)

        module.LOOKAHEAD_DIST = float(lookahead_m)

        collision_history: list[int] = []
        feasibility_statuses: list[list[int]] = []

        with contextlib.redirect_stdout(io.StringIO()):
            sim = module.HeadlessSim(
                n_robots=n_robots,
                behavior_mode=module.SwarmBehavior(case_name),
                enable_obstacles=False,
                enable_collision=True,
            )
            for _ in range(steps):
                sim.step()
                collision_history.append(len(sim.active_collisions))
                feasibility_statuses.append(list(sim.solver_statuses))

        all_times_us = [t for r_times in sim.solve_times for t in r_times]
        mean_us = float(np.mean(all_times_us)) if all_times_us else 0.0
        max_ms = float(np.max(all_times_us) / 1000.0) if all_times_us else 0.0

        feas = np.array(feasibility_statuses)
        total_solves = feas.size
        successful = int(np.sum(feas == 0))
        feasibility_rate = (successful / total_solves) * 100.0 if total_solves > 0 else 0.0

        sys.path = orig_sys_path

        return {
            "success": True,
            "folder": folder,
            "label": label,
            "n_robots": n_robots,
            "case_name": case_name,
            "lookahead_m": lookahead_m,
            "steps": steps,
            "collision_history": collision_history,
            "feasibility_rate": feasibility_rate,
            "mean_us": mean_us,
            "max_ms": max_ms,
        }

    except Exception as e:
        sys.path = orig_sys_path
        return {
            "success": False,
            "error": str(e),
            "label": label,
            "n_robots": n_robots,
            "case_name": case_name,
            "lookahead_m": lookahead_m,
        }


def _build_tasks(n_robots: int, lookaheads: list[float], steps: int) -> list[tuple]:
    tasks = []
    for la in lookaheads:
        for case in CASES:
            for folder, label in SAFETY_LEVEL_FOLDERS:
                tasks.append((folder, label, n_robots, case["name"], la, steps))
    return tasks


def _result_row(res: dict) -> dict:
    hist = res["collision_history"]
    return {
        "lookahead_m": res["lookahead_m"],
        "lookahead_cm": round(res["lookahead_m"] * 100.0, 2),
        "n_robots": res["n_robots"],
        "safety_level": res["label"],
        "case_name": res["case_name"],
        "sim_steps": res["steps"],
        "sim_time_sec": round(res["steps"] * DT, 1),
        "total_collisions": _collision_total(hist),
        "early_collisions_steps_0_30": _early_collisions(hist, EARLY_COLLISION_STEPS),
        "feasibility_rate_pct": round(res["feasibility_rate"], 2),
        "mean_solve_time_us": round(res["mean_us"], 2),
        "max_solve_time_ms": round(res["max_ms"], 4),
        "production_baseline": int(res["lookahead_m"] == 0.07),
    }


def run_sweep(n_robots: int, workers: int, out_csv: Path) -> Path:
    lookaheads = LOOKAHEAD_VALUES_M
    tasks = _build_tasks(n_robots, lookaheads, SWEEP_STEPS)

    print("=" * 72)
    print("  Lookahead distance sweep (LOOKAHEAD_DIST)")
    print(f"  Values [m]: {lookaheads}")
    print(f"  N={n_robots}  cases={[c['name'] for c in CASES]}")
    print(f"  STEPS={SWEEP_STEPS} ({SWEEP_TIME_SEC:.1f}s sim), short run, not full {60/DT:.0f} steps")
    print(f"  Variants: {[l for _, l in SAFETY_LEVEL_FOLDERS]}")
    print(f"  Tasks: {len(tasks)}  workers: {workers}")
    print("=" * 72)

    t0 = time.time()
    rows: list[dict] = []
    ok = 0
    with multiprocessing.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(sim_worker_task, tasks):
            if res.get("success"):
                ok += 1
                rows.append(_result_row(res))
            else:
                print(
                    f"  [FAIL] la={res.get('lookahead_m')} {res.get('label')} "
                    f"N={res.get('n_robots')} {res.get('case_name')}: {res.get('error')}",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\nCompleted {ok}/{len(tasks)} in {elapsed:.1f}s")

    rows.sort(
        key=lambda r: (
            r["case_name"],
            r["safety_level"],
            r["lookahead_m"],
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="LOOKAHEAD_DIST sweep for DMPC variants")
    parser.add_argument("--n-robots", type=int, default=DEFAULT_N_ROBOTS)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_CSV,
        help="Output CSV path",
    )
    args = parser.parse_args()
    run_sweep(args.n_robots, args.workers, args.out)


if __name__ == "__main__":
    main()
