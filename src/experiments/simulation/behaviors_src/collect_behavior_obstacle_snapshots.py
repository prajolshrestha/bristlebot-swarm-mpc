"""Simulates each behavior in each environment and saves snapshots along the way."""
from __future__ import annotations
import os
import sys
import csv
import pickle
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VARIANT_ROOT = os.path.abspath(os.path.join(
    _SCRIPT_DIR, "..", "..", "..", "deterministic_bristlebot_swarm",
    "05_slacked_dt_cbf_with_lookahead"))
_SIM_ENGINE_DIR = os.path.join(_VARIANT_ROOT, "sim_engine")
_COMPARISON_CODE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "comparison_src"))
for _p in (_VARIANT_ROOT, _SIM_ENGINE_DIR, _COMPARISON_CODE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_TIMES = (0, 15, 30, 45, 60)
if "SNAP_TIMES" in os.environ:
    SNAPSHOT_TIMES_SEC = tuple(int(float(t)) for t in os.environ["SNAP_TIMES"].split(","))
else:
    SNAPSHOT_TIMES_SEC = _DEFAULT_TIMES
SPAN_TAG = "" if SNAPSHOT_TIMES_SEC == _DEFAULT_TIMES else f"_{SNAPSHOT_TIMES_SEC[-1]}s"

OUT_DIR = os.path.abspath(os.path.join(
    _SCRIPT_DIR, "..", "results", "raw", "behavior_obstacle_review" + SPAN_TAG))
CKPT_DIR = os.path.join(OUT_DIR, "_ckpt")
CSV_PATH = os.path.join(OUT_DIR, "snapshot_data.csv")

BEHAVIORS = [
    "phototaxis",
    "phototaxis_no_reynolds",
    "no_light",
    "phototaxis_orbital",
    "phototaxis_orbital_contracting",
]
N_LIST = [10, 25, 50, 75, 100, 125, 137]
OBSTACLE_COUNTS = [0, 2, 12]
LAYOUT_SEED = 0


def _ckpt_path(behavior, count, n):
    return os.path.join(CKPT_DIR, f"{behavior}__{count}obs__N{n}.pkl")


def run_one(behavior, count, n):
    """Run one sim, return its snapshot list (or load it from the checkpoint)."""
    cp = _ckpt_path(behavior, count, n)
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)

    import plot_behaviors_for_paper as pb
    import sim_engine.headless_sim_engine as H
    from sim_engine.swarm_dynamics import SwarmBehavior

    H.OBSTACLE_DEFS = H.build_obstacle_defs(count, seed=LAYOUT_SEED)
    steps = tuple(int(round(t / pb.DT)) for t in SNAPSHOT_TIMES_SEC)

    devnull = os.open(os.devnull, os.O_WRONLY)
    s1, s2 = os.dup(1), os.dup(2)
    os.dup2(devnull, 1); os.dup2(devnull, 2)
    try:
        sim = pb.HeadlessSim(n_robots=n, behavior_mode=SwarmBehavior(behavior),
                             enable_obstacles=(count > 0))
        snaps = pb._collect_sim_snapshots(sim, steps=steps)
    finally:
        os.dup2(s1, 1); os.dup2(s2, 2)
        for fd in (devnull, s1, s2):
            os.close(fd)

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump(snaps, f)
    return snaps


def _csv_rows(behavior, count, n, snaps):
    """Flatten snaps to long-format CSV rows (kind = robot/obstacle/light)."""
    rows = []
    for t_sec, snap in zip(SNAPSHOT_TIMES_SEC, snaps):
        pos = np.asarray(snap["positions"])
        head = np.asarray(snap["headings"])
        coll = set(snap.get("colliding_indices", []))
        viol = set(snap.get("in_violation", []))
        for i in range(len(pos)):
            rows.append(dict(behavior=behavior, n_robots=n, obstacle_count=count,
                             t_sec=t_sec, kind="robot", id=i,
                             x=f"{pos[i,0]:.5f}", y=f"{pos[i,1]:.5f}",
                             heading_rad=f"{head[i]:.5f}",
                             colliding=int(i in coll), in_violation=int(i in viol),
                             radius=""))
        for k, ob in enumerate(snap.get("obstacles", [])):
            op = np.asarray(ob["pos"])
            rows.append(dict(behavior=behavior, n_robots=n, obstacle_count=count,
                             t_sec=t_sec, kind="obstacle", id=k,
                             x=f"{op[0]:.5f}", y=f"{op[1]:.5f}", heading_rad="",
                             colliding="", in_violation="", radius=f"{ob['radius']:.5f}"))
        for k, src in enumerate(snap.get("field_sources") or []):
            sp = np.asarray(src)
            rows.append(dict(behavior=behavior, n_robots=n, obstacle_count=count,
                             t_sec=t_sec, kind="light", id=k,
                             x=f"{sp[0]:.5f}", y=f"{sp[1]:.5f}", heading_rad="",
                             colliding="", in_violation="", radius=""))
    return rows


def _worker(args):
    behavior, count, n = args
    try:
        snaps = run_one(behavior, count, n)
        return {"ok": True, "task": args, "rows": _csv_rows(behavior, count, n, snaps)}
    except Exception as exc:
        import traceback
        return {"ok": False, "task": args, "err": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--behaviors", nargs="*", default=BEHAVIORS)
    ap.add_argument("--n", nargs="*", type=int, default=N_LIST)
    ap.add_argument("--counts", nargs="*", type=int, default=OBSTACLE_COUNTS)
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    tasks = [(b, c, n) for b in args.behaviors for c in args.counts for n in args.n]
    print(f"[collect] {len(tasks)} tasks "
          f"({len(args.behaviors)} beh x {len(args.counts)} counts x {len(args.n)} N), "
          f"workers={args.workers}", flush=True)

    all_rows = []
    fields = ["behavior", "n_robots", "obstacle_count", "t_sec", "kind", "id",
              "x", "y", "heading_rad", "colliding", "in_violation", "radius"]

    if args.workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_worker, tasks), 1):
                tag = "OK " if r["ok"] else "FAIL"
                print(f"  [{i}/{len(tasks)}] {tag} {r['task']}", flush=True)
                if not r["ok"]:
                    print(r["err"].splitlines()[-1], flush=True)
                results.append(r)

    for r in results:
        if r["ok"]:
            all_rows.extend(r["rows"])
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    ok = sum(1 for r in results if r["ok"])
    print(f"[collect] done: {ok}/{len(tasks)} ok; wrote {CSV_PATH} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
