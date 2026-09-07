#!/usr/bin/env python3
"""Runs one benchmark task, or aggregates a finished campaign."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import re
import pickle
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

N_LIST_600 = [10, 25, 50, 75, 100, 125, 137]
DEFAULT_RESULTS = os.path.join(SCRIPT_DIR, "..", "results", "raw", "bench_900s")


def _san(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s))


def apply_overrides(ABP, steps, results_dir):
    """Rebind the module globals every harness function reads (valid because the
    from-import at auto_benchmarking_parallel.py:36 binds them into ABP's namespace)."""
    ABP.STEPS = int(steps)
    ABP.SIM_TIME_SEC = float(steps) * float(ABP.DT)
    ABP.STEADY_STATE_STEPS = int(steps) // 2
    ABP.N_LIST = list(N_LIST_600)
    ABP.RESULTS_DIR = results_dir
    _RES = os.path.join(SCRIPT_DIR, "..", "results")
    ABP.CSV_DIR = os.path.abspath(os.path.join(_RES, "csv"))
    ABP.FIG_DIR = os.path.abspath(os.path.join(_RES, "figures"))
    os.makedirs(ABP.CSV_DIR, exist_ok=True)
    os.makedirs(ABP.FIG_DIR, exist_ok=True)


def build_tasks(ABP, enable_obstacles=False):
    """EXACT task order of run_parallel_benchmark (auto_benchmarking_parallel.py:732-738)."""
    all_tasks = []
    for case in ABP.CASES:
        case_name = case["name"]
        for folder, label in ABP.SAFETY_LEVEL_FOLDERS:
            for n in ABP.N_LIST:
                for seed in ABP.SEEDS:
                    all_tasks.append((folder, label, n, case_name, enable_obstacles, seed))
    return all_tasks


def ckpt_path(ABP, case_name, label, n, seed):
    return os.path.join(ABP.RESULTS_DIR, "_ckpt",
                        f"{_san(case_name)}__{_san(label)}__N{n}__s{seed}.pkl")


def run_one(ABP, k):
    tasks = build_tasks(ABP)
    if not 0 <= k < len(tasks):
        sys.exit(f"task-id {k} out of range 0..{len(tasks) - 1}")
    folder, label, n, case_name, enable_obstacles, seed = tasks[k]
    fp = ckpt_path(ABP, case_name, label, n, seed)
    if os.path.exists(fp):
        print(f"[task {k}] cached, skipping: {os.path.basename(fp)}", flush=True)
        return
    print(f"[task {k}] {label} | N={n} | {case_name} | seed={seed} | steps={ABP.STEPS}", flush=True)
    res = ABP.sim_worker_task(tasks[k])
    if not res.get("success"):
        print(f"[task {k}] FAILED: {res.get('error', 'unknown')}", flush=True)
        sys.exit(1)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp + ".tmp", "wb") as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(fp + ".tmp", fp)
    print(f"[task {k}] OK -> {os.path.basename(fp)}", flush=True)


def aggregate(ABP, workers, allow_partial, plots=True):
    tasks = build_tasks(ABP)
    missing = [t for t in tasks
               if not os.path.exists(ckpt_path(ABP, t[3], t[1], t[2], t[5]))]
    print(f"[aggregate] {len(tasks) - len(missing)}/{len(tasks)} checkpoints present")
    if missing and not allow_partial:
        for t in missing[:15]:
            print(f"  missing: {t[1]} | N={t[2]} | {t[3]} | seed={t[5]}")
        sys.exit(f"{len(missing)} runs missing -- resubmit the array, or pass "
                 f"--allow-partial to simulate the remainder here")
    ABP.SAVE_CSV = True
    ABP.SAVE_PLOTS = plots
    ABP.run_parallel_benchmark(enable_obstacles=False, workers_override=workers)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-id", type=int, default=None)
    mode.add_argument("--aggregate", action="store_true")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--behavior", default=None,
                    help="restrict the sweep to one case (e.g. phototaxis_orbital); "
                         "checkpoint keys carry the case name so it coexists in the same dir")
    ap.add_argument("--variants", default=None,
                    help="comma-separated variant LABELS to run (e.g. 'Hard BF,Hybrid v2'); "
                         "default = the harness registry. The task-id order depends on this "
                         "list, so array tasks and --aggregate must pass the SAME value.")
    args = ap.parse_args()

    import auto_benchmarking_parallel as ABP
    apply_overrides(ABP, args.steps, os.path.abspath(args.results_dir))
    if args.variants:
        want = [s.strip() for s in args.variants.split(",") if s.strip()]
        registry = {label: folder for folder, label in ABP.SAFETY_LEVEL_FOLDERS}
        unknown = [w for w in want if w not in registry]
        if unknown:
            sys.exit(f"unknown variant label(s) {unknown}; registry has {sorted(registry)}")
        ABP.SAFETY_LEVEL_FOLDERS = [(registry[w], w) for w in want]
    if args.behavior:
        _disp = {"phototaxis": "Phototaxis", "phototaxis_orbital": "Orbital",
                 "phototaxis_orbital_contracting": "Orbital (contracting)"}.get(
                     args.behavior, args.behavior.replace("_", " ").title())
        ABP.CASES = [{"name": args.behavior, "display_name": _disp, "suffix": args.behavior}]

    if args.aggregate:
        aggregate(ABP, args.workers, args.allow_partial, plots=(args.behavior is None))
    else:
        run_one(ABP, args.task_id)


if __name__ == "__main__":
    main()
