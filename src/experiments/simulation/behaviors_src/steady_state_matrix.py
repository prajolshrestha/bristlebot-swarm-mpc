#!/usr/bin/env python3
"""Simulates one behavior at one density and logs the per-step metrics."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import csv
import time
import pickle
import argparse
import importlib
import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "deterministic_bristlebot_swarm"))
VARIANT_DIR = {
    "01": os.path.join(REPO, "01_hard_bf"),
    "02": os.path.join(REPO, "02_hard_dt_cbf"),
    "03": os.path.join(REPO, "03_slacked_bf"),
    "04": os.path.join(REPO, "04_slacked_dt_cbf"),
    "05": os.path.join(REPO, "05_slacked_dt_cbf_with_lookahead"),
    "06": os.path.join(REPO, "06_slacked_dt_hocbf"),
}
OUT_DIR = os.environ.get("MATRIX_OUT_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "paper_exp_10seed_results", "obstacle_matrix")
RAW_DIR = os.environ.get("MATRIX_RAW_DIR", os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "matrix_npz")))

CSV_HEADER = ["step", "t_sec", "min_dist_m", "avg_min_sep_m", "mean_dist_m",
              "coherence", "local_polarization", "comm_neighbor_mean_dist_m",
              "local_nematic", "global_nematic", "cum_collisions", "active_collisions",
              "safety_violations", "mean_path_len_m"]


def load_engine(variant):
    vdir = VARIANT_DIR[variant]
    if vdir not in sys.path:
        sys.path.insert(0, vdir)
    Engine = importlib.import_module("sim_engine.headless_sim_engine")
    SwarmBehavior = importlib.import_module("sim_engine.swarm_dynamics").SwarmBehavior
    return Engine, SwarmBehavior


def build_obstacle_defs(count, seed):
    """Identical layout to the engines' build_obstacle_defs, replicated here so
    it is variant-independent (variant 01's engine lacks the helper)."""
    if count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    lim, min_sep = 0.30, 0.12
    pts = []
    for _ in range(int(count)):
        cand = None
        for _t in range(2000):
            c = rng.uniform(-lim, lim, 2)
            if all(np.linalg.norm(c - p) >= min_sep for p in pts):
                cand = c
                break
        pts.append(cand if cand is not None else rng.uniform(-lim, lim, 2))
    return [{"type": "static", "pos": [float(p[0]), float(p[1])], "active": True} for p in pts]


def neighbor_metrics(pos, vel, theta, radius):
    """One KD-tree pass -> (local_polarization, local_nematic, comm_neighbor_mean_dist)."""
    tree = cKDTree(pos)
    nb = tree.query_ball_point(pos, radius)
    speeds = np.linalg.norm(vel, axis=1)
    speeds = np.where(speeds < 1e-9, 1.0, speeds)
    uv = vel / speeds[:, None]
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    glob_nem = float(np.hypot(c2.mean(), s2.mean()))
    lp, nem, cdist = [], [], []
    for i, idx in enumerate(nb):
        lp.append(np.linalg.norm(uv[idx].mean(0)))
        nem.append(float(np.hypot(c2[idx].mean(), s2[idx].mean())))
        others = [j for j in idx if j != i]
        if others:
            cdist.append(float(np.linalg.norm(pos[others] - pos[i], axis=1).mean()))
    return (float(np.mean(lp)), float(np.mean(nem)),
            float(np.mean(cdist)) if cdist else float("nan"), glob_nem)


RAW_SLOTS = ("raw_pos", "raw_vel", "raw_theta")
_ARR_SLOTS = RAW_SLOTS + ("dist_accum", "prev_pos")


class Extra:
    __slots__ = ("local_pol", "comm_dist", "nematic", "glob_nem", "cum_coll", "active_coll",
                 "safety_viol", "mean_path", "dist_accum", "prev_pos",
                 "raw_pos", "raw_vel", "raw_theta")

    def __init__(self):
        self.local_pol, self.comm_dist, self.nematic = [], [], []
        self.glob_nem = []
        self.cum_coll, self.active_coll, self.safety_viol = [], [], []
        self.mean_path = []
        self.dist_accum = None
        self.prev_pos = None
        self.raw_pos = None
        self.raw_vel = None
        self.raw_theta = None


def build_sim(Engine, SwarmBehavior, n, behavior, n_obs, seed, reactive):
    Engine.SPAWN_SEED = int(seed)
    Engine.ENABLE_OBSTACLES = bool(n_obs > 0)
    Engine.OBSTACLE_DEFS = build_obstacle_defs(n_obs, seed)
    sim = Engine.HeadlessSim(
        n_robots=n,
        behavior_mode=SwarmBehavior(behavior),
        max_consensus=Engine.MAX_CONSENSUS,
        enable_obstacles=bool(n_obs > 0),
        enable_collision=bool(reactive),
    )
    return sim


def log_step(sim, extra, comm_radius, d_safe, step):
    pos = np.array([r.pos for r in sim.robots])
    vel = np.array([r.velocity for r in sim.robots])
    th = np.array([float(r.theta) for r in sim.robots])
    lp, nem, cdist, gnem = neighbor_metrics(pos, vel, th, comm_radius)
    extra.local_pol.append(lp)
    extra.nematic.append(nem)
    extra.comm_dist.append(cdist)
    extra.glob_nem.append(gnem)
    extra.cum_coll.append(int(getattr(sim, "total_collisions", 0)))
    extra.active_coll.append(int(len(getattr(sim, "active_collisions", []))))
    extra.safety_viol.append(int(len(cKDTree(pos).query_pairs(d_safe))))
    extra.dist_accum += np.linalg.norm(pos - extra.prev_pos, axis=1)
    extra.prev_pos = pos.copy()
    extra.mean_path.append(float(extra.dist_accum.mean()))
    extra.raw_pos[step] = pos
    extra.raw_vel[step] = vel
    extra.raw_theta[step] = th


def save_raw(path, extra, args, comm_radius, d_safe, dt):
    """One compressed .npz per run: full per-robot pos/vel/heading time series
    (float32) + metadata, so any observable can be recomputed without re-running."""
    tmp = path[:-4] + ".tmp"
    np.savez_compressed(
        tmp, pos=extra.raw_pos, vel=extra.raw_vel, theta=extra.raw_theta,
        variant=args.variant, behavior=args.behavior, n=args.n,
        obstacles=args.obstacles, seed=args.seed, steps=args.steps,
        reactive=args.reactive, dt=dt, comm_radius=comm_radius, d_safe=d_safe)
    os.replace(tmp + ".npz", path)


def write_csv(path, sim, extra, upto):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        dt = sim_dt(sim)
        for k in range(upto):
            w.writerow([k, round((k + 1) * dt, 2),
                        round(sim.min_dist_log[k], 5), round(sim.avg_min_sep_log[k], 5),
                        round(sim.mean_dist_log[k], 5), round(sim.coherence_log[k], 5),
                        round(extra.local_pol[k], 5),
                        round(extra.comm_dist[k], 5) if extra.comm_dist[k] == extra.comm_dist[k] else "",
                        round(extra.nematic[k], 5),
                        round(extra.glob_nem[k], 5),
                        extra.cum_coll[k], extra.active_coll[k], extra.safety_viol[k],
                        round(extra.mean_path[k], 5)])
    os.replace(tmp, path)


def sim_dt(sim):
    """Ts of the sim's own engine module (DT is a module global, not a sim attr)."""
    eng = sys.modules.get(type(sim).__module__)
    return float(getattr(eng, "DT", 0.1))


def save_ckpt(path, sim, extra, step):
    state = {
        "step": step, "current_time": sim.current_time,
        "positions": np.array([r.pos for r in sim.robots]),
        "thetas": np.array([float(r.theta) for r in sim.robots]),
        "velocities": np.array([r.velocity for r in sim.robots]),
        "ang_velocities": np.array([float(r.ang_velocity) for r in sim.robots]),
        "logs": {k: list(getattr(sim, k)) for k in
                 ("min_dist_log", "avg_min_sep_log", "mean_dist_log", "coherence_log")},
        "extra": {s: list(getattr(extra, s)) for s in extra.__slots__
                  if s not in _ARR_SLOTS},
        "dist_accum": None if extra.dist_accum is None else np.asarray(extra.dist_accum, float),
        "prev_pos": None if extra.prev_pos is None else np.asarray(extra.prev_pos, float),
        "raw_pos": extra.raw_pos[:step], "raw_vel": extra.raw_vel[:step],
        "raw_theta": extra.raw_theta[:step],
        "total_collisions": int(getattr(sim, "total_collisions", 0)),
        "active_collisions": [tuple(p) for p in getattr(sim, "active_collisions", [])],
        "field": getattr(sim, "field", None),
    }
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
    except Exception:
        state["field"] = None
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
    os.replace(tmp, path)


def restore_ckpt(sim, extra, state):
    for i, r in enumerate(sim.robots):
        r.pos = np.asarray(state["positions"][i], float).copy()
        r.theta = float(state["thetas"][i])
        r.velocity = np.asarray(state["velocities"][i], float).copy()
        r.ang_velocity = float(state["ang_velocities"][i])
    for k, v in state["logs"].items():
        setattr(sim, k, list(v))
    for s, v in state["extra"].items():
        setattr(extra, s, list(v))
    if state.get("dist_accum") is not None:
        extra.dist_accum = np.asarray(state["dist_accum"], float).copy()
    if state.get("prev_pos") is not None:
        extra.prev_pos = np.asarray(state["prev_pos"], float).copy()
    if state.get("raw_pos") is not None:
        st = int(state["step"])
        extra.raw_pos[:st] = state["raw_pos"]
        extra.raw_vel[:st] = state["raw_vel"]
        extra.raw_theta[:st] = state["raw_theta"]
    sim.total_collisions = state["total_collisions"]
    if hasattr(sim, "active_collisions"):
        sim.active_collisions = set(frozenset(p) for p in state["active_collisions"])
    sim.current_time = state["current_time"]
    if state.get("field") is not None:
        sim.field = state["field"]
    return int(state["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANT_DIR), required=True)
    ap.add_argument("--behavior", default="phototaxis",
                    choices=["phototaxis", "phototaxis_orbital",
                             "phototaxis_orbital_contracting", "no_light"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--obstacles", type=int, choices=[0, 2, 12], required=True)
    ap.add_argument("--reactive", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--flush-every", type=int, default=200)
    args = ap.parse_args()

    Engine, SwarmBehavior = load_engine(args.variant)
    comm_radius = float(Engine.COMM_RADIUS)
    d_safe = float(getattr(Engine, "D_SAFE", 0.07))
    reactive = (args.reactive == "on")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    tag = (f"v{args.variant}_{args.behavior}_N{args.n}_obs{args.obstacles}"
           f"_reac{args.reactive}_seed{args.seed}_steps{args.steps}")
    final = os.path.join(OUT_DIR, tag + ".csv")
    raw_path = os.path.join(RAW_DIR, tag + ".npz")
    partial = os.path.join(RAW_DIR, tag + ".csv.partial")
    ckpt = os.path.join(RAW_DIR, "." + tag + ".ckpt.pkl")
    if os.path.exists(final) and os.path.exists(raw_path):
        print(f"[{tag}] cached, skipping", flush=True)
        return

    sim = build_sim(Engine, SwarmBehavior, args.n, args.behavior, args.obstacles, args.seed, reactive)
    extra = Extra()
    extra.dist_accum = np.zeros(args.n)
    extra.prev_pos = np.array([r.pos for r in sim.robots], float)
    extra.raw_pos = np.zeros((args.steps, args.n, 2), np.float32)
    extra.raw_vel = np.zeros((args.steps, args.n, 2), np.float32)
    extra.raw_theta = np.zeros((args.steps, args.n), np.float32)
    start = 0
    if os.path.exists(ckpt):
        try:
            with open(ckpt, "rb") as f:
                start = restore_ckpt(sim, extra, pickle.load(f))
            print(f"[{tag}] resumed at step {start}", flush=True)
        except Exception as exc:
            print(f"[{tag}] checkpoint unreadable ({exc}); fresh", flush=True)
            start = 0

    dt = sim_dt(sim)
    print(f"[{tag}] steps {start}..{args.steps} ({args.steps*dt:.0f}s), "
          f"COMM_RADIUS={comm_radius}, D_SAFE={d_safe}, reactive={reactive}", flush=True)
    t0 = time.perf_counter()
    for s in range(start, args.steps):
        sim.step()
        log_step(sim, extra, comm_radius, d_safe, s)
        done = s + 1
        if done % args.flush_every == 0:
            write_csv(partial, sim, extra, len(sim.min_dist_log))
        if done % args.checkpoint_every == 0:
            save_ckpt(ckpt, sim, extra, done)
        if done % 1000 == 0:
            rate = (done - start) / max(1e-9, time.perf_counter() - t0)
            print(f"[{tag}] step {done}/{args.steps} ({rate:.1f}/s, "
                  f"coll={getattr(sim,'total_collisions',0)})", flush=True)

    write_csv(final, sim, extra, len(sim.min_dist_log))
    save_raw(raw_path, extra, args, comm_radius, d_safe, dt)
    for p in (partial, ckpt):
        if os.path.exists(p):
            os.remove(p)
    print(f"[{tag}] DONE -> {final} + {os.path.basename(raw_path)} "
          f"({time.perf_counter()-t0:.0f}s, coll={getattr(sim,'total_collisions',0)})", flush=True)


if __name__ == "__main__":
    main()
