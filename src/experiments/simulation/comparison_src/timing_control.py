#!/usr/bin/env python3
"""Measures solve time on an idle machine, one solve at a time."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import csv
import time
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

VARIANTS = [
    ("01_hard_bf", "Hard BF"),
    ("02_hard_dt_cbf", "Hard Dt-CBF"),
    ("03_slacked_bf", "Slack BF"),
    ("04_slacked_dt_cbf", "Slack Dt-CBF"),
    ("05_slacked_dt_cbf_with_lookahead", "LA Slack Dt-CBF"),
    ("06_slacked_dt_hocbf", "Slack Dt-HOCBF"),
]
NS = [10, 25, 50, 75, 100, 125, 137]

WORKER = r'''
import os, sys, io, contextlib, importlib, numpy as np, json
vdir, n, steps, seed, outjson = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, vdir)
with contextlib.redirect_stdout(io.StringIO()):
    m = importlib.import_module("sim_engine.headless_sim_engine")
    m.SPAWN_SEED = seed
    sim = m.HeadlessSim(n_robots=n, enable_obstacles=False, enable_collision=True)
    for _ in range(steps):
        sim.step()
t = []
for r in sim.solve_times:
    t.extend(r)
t = np.asarray(t, float)
with open(outjson, "w") as fh:
    json.dump({"n_solves": int(t.size),
               "mean_us": float(t.mean()) if t.size else 0.0,
               "p50_us": float(np.percentile(t, 50)) if t.size else 0.0,
               "p99_us": float(np.percentile(t, 99)) if t.size else 0.0,
               "max_ms": float(t.max() / 1000.0) if t.size else 0.0}, fh)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="timing_control_exclusive.csv")
    a = ap.parse_args()

    wpath = os.path.join(SCRIPT_DIR, "_timing_worker.py")
    with open(wpath, "w") as fh:
        fh.write(WORKER)

    out = os.path.join(SCRIPT_DIR, a.out)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["safety_level", "n_robots", "steps", "seed", "n_solves",
                    "mean_solve_time_us", "p50_solve_time_us", "p99_solve_time_us",
                    "max_solve_time_ms", "wall_s"])
        for folder, label in VARIANTS:
            vdir = os.path.join(ROOT_DIR, "deterministic_mpc", folder)
            for n in NS:
                t0 = time.time()
                rpath = os.path.join(SCRIPT_DIR, f"_timing_result_{label.replace(' ', '_')}_{n}.json")
                if os.path.exists(rpath):
                    os.remove(rpath)
                p = subprocess.run([sys.executable, wpath, vdir, str(n),
                                    str(a.steps), str(a.seed), rpath],
                                   capture_output=True, text=True)
                wall = time.time() - t0
                if p.returncode != 0 or not os.path.exists(rpath):
                    print(f"[FAIL] {label} N={n} rc={p.returncode}", flush=True)
                    print((p.stderr or p.stdout or "")[-500:], flush=True)
                    continue
                import json
                with open(rpath) as rfh:
                    d = json.load(rfh)
                os.remove(rpath)
                w.writerow([label, n, a.steps, a.seed, d["n_solves"],
                            f"{d['mean_us']:.1f}", f"{d['p50_us']:.1f}",
                            f"{d['p99_us']:.1f}", f"{d['max_ms']:.4f}", f"{wall:.1f}"])
                fh.flush()
                print(f"{label:<14} N={n:<4} mean={d['mean_us']/1000:.3f} ms  "
                      f"p99={d['p99_us']/1000:.3f} ms  ({wall:.0f} s wall)", flush=True)
    os.remove(wpath)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
