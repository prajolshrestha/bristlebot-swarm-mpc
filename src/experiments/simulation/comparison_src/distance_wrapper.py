#!/usr/bin/env python3
"""Runs one safety-margin task, or merges a finished sweep."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import re
import csv
import pickle
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _san(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s))


def build_tasks(DC, seeds=10):
    """EXACT task order of DC.run (distance_comparison.py all_tasks loop)."""
    return [(folder, lab, n, s)
            for folder, lab in DC.SAFETY_LEVEL_FOLDERS
            for n in DC.DIST_NS for s in range(seeds)]


def frag_path(DC, label, n, seed):
    return os.path.join(DC.RESULTS_DIR, "_ckpt", f"{_san(label)}__N{n}__s{seed}.pkl")


def run_one(DC, k, seeds):
    tasks = build_tasks(DC, seeds)
    if not 0 <= k < len(tasks):
        sys.exit(f"task-id {k} out of range 0..{len(tasks) - 1}")
    folder, label, n, seed = tasks[k]
    fp = frag_path(DC, label, n, seed)
    if os.path.exists(fp):
        print(f"[task {k}] cached, skipping: {os.path.basename(fp)}", flush=True)
        return
    print(f"[task {k}] {label} | N={n} | seed={seed} | steps={DC.STEPS} | obstacles ON",
          flush=True)
    r = DC._worker(tasks[k])
    if not r.get("ok"):
        print(f"[task {k}] FAILED: {r.get('err', 'unknown')}", flush=True)
        sys.exit(1)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp + ".tmp", "wb") as fh:
        pickle.dump(r, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(fp + ".tmp", fp)
    print(f"[task {k}] OK min_rr={r['min_rr']:.4f} min_ro={r['min_ro']:.4f} "
          f"-> {os.path.basename(fp)}", flush=True)


def merge(DC, seeds, allow_partial):
    tasks = build_tasks(DC, seeds)
    frags, missing = [], []
    for t in tasks:
        fp = frag_path(DC, t[1], t[2], t[3])
        if os.path.exists(fp):
            with open(fp, "rb") as fh:
                frags.append(pickle.load(fh))
        else:
            missing.append(t)
    print(f"[merge] {len(frags)}/{len(tasks)} fragments present")
    if missing and not allow_partial:
        for t in missing[:15]:
            print(f"  missing: {t[1]} | N={t[2]} | seed={t[3]}")
        sys.exit(f"{len(missing)} runs missing -- resubmit the array, or pass "
                 f"--allow-partial to simulate the remainder in-process")
    os.makedirs(DC.RESULTS_DIR, exist_ok=True)
    partial = os.path.join(DC.RESULTS_DIR, "_partial_distances.csv")
    with open(partial + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_robots", "seed", "ok", "min_rr", "min_ro"])
        for r in frags:
            w.writerow([r["label"], r["n_robots"], r["seed"], int(r.get("ok", False)),
                        r.get("min_rr", ""), r.get("min_ro", "")])
    os.replace(partial + ".tmp", partial)
    print(f"[merge] wrote {partial}")
    DC.run(seeds=seeds, workers=1)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-id", type=int, default=None)
    mode.add_argument("--merge", action="store_true")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    if "DIST_RESULTS_DIR" not in os.environ:
        os.environ["DIST_RESULTS_DIR"] = os.path.join(SCRIPT_DIR, "..", "results", "raw", "distance_900s")
    import distance_comparison as DC
    DC.STEPS = int(args.steps)

    if args.merge:
        merge(DC, args.seeds, args.allow_partial)
    else:
        run_one(DC, args.task_id, args.seeds)


if __name__ == "__main__":
    main()
