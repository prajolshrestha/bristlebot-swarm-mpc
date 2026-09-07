#!/usr/bin/env python3
"""Searches for the best safety gain, and look-ahead length where one exists."""
import argparse
import contextlib
import csv
import signal
import io
import itertools
import multiprocessing
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
SWARM = os.path.abspath(os.path.join(HERE, "..", ".."))
PYEXE = sys.executable

VARIANTS = {
    "02": dict(path="src/deterministic_bristlebot_swarm/02_hard_dt_cbf",
               gamma_keys=["cbf_alpha"], has_lookahead=False, family="det"),
    "03": dict(path="src/deterministic_bristlebot_swarm/03_slacked_bf",
               gamma_keys=["w_soft"], has_lookahead=False, family="det"),
    "04": dict(path="src/deterministic_bristlebot_swarm/04_slacked_dt_cbf",
               gamma_keys=["cbf_alpha"], has_lookahead=False, family="det"),
    "05": dict(path="src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead",
               gamma_keys=["cbf_alpha"], has_lookahead=True, family="det"),
    "06": dict(path="src/deterministic_bristlebot_swarm/06_slacked_dt_hocbf",
               gamma_keys=["hocbf_gamma1", "hocbf_gamma2"], has_lookahead=False, family="det"),
}


def set_lookahead(ycfg, spec, value):
    """Write lookahead_dist only for the variant that has the handle."""
    if spec.get("has_lookahead"):
        yaml_set(ycfg, "lookahead_dist", value)


def lookahead_keys(spec):
    return ["lookahead_dist"] if spec.get("has_lookahead") else []

GAMMAS = [0.3, 0.5, 0.7, 0.9]
L_FRACS = [0.3, 0.5, 0.7, 0.9]

SCREEN_NS = {"det": [50, 100, 137], "sto": [50, 100, 120]}
FULL_NS = {"det": [10, 25, 50, 75, 100, 125, 137], "sto": [10, 25, 50, 75, 100, 120]}
STEPS = 600


def yaml_get(path, key):
    for line in open(path):
        m = re.match(rf"^(\s+){re.escape(key)}:\s*([0-9.eE+-]+)", line)
        if m:
            return float(m.group(2))
    raise KeyError(f"{key} not found in {path}")


def yaml_set(path, key, value):
    out, hit = [], False
    for line in open(path):
        m = re.match(rf"^(\s+){re.escape(key)}:\s*([0-9.eE+-]+)(.*)$", line)
        if m:
            out.append(f"{m.group(1)}{key}: {value}{m.group(3)}\n")
            hit = True
        else:
            out.append(line)
    if not hit:
        raise KeyError(f"{key} not found in {path}")
    open(path, "w").write("".join(out))


def recompile(vdir):
    es = os.path.join(vdir, "expert_src")
    subprocess.run([PYEXE, "generate_solver_via_acados.py"], cwd=es,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


WORKER_SRC = r'''
import contextlib, importlib.util, io, json, os, sys, multiprocessing
import numpy as np

VDIR, STEPS, CASE, ENABLE_COLL = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
TASKS = json.loads(sys.argv[5])

def run(t):
    n_robots, seed = t
    sys.path.insert(0, VDIR); sys.path.insert(0, os.path.join(VDIR, "sim_engine"))
    spec = importlib.util.spec_from_file_location(
        "dyn_sim", os.path.join(VDIR, "sim_engine", "headless_sim_engine.py"))
    m = importlib.util.module_from_spec(spec)
    for k in list(sys.modules):
        if any(s in k for s in ["dyn_sim","headless_sim_engine","swarm_dynamics",
                                "light_source_model","obstacle_model","collision_response","expert_src"]):
            del sys.modules[k]
    spec.loader.exec_module(m)
    # Spawn diversity: the engines take their initial-layout RNG from the module
    # global SPAWN_SEED (default 42), NOT from np.random -- np.random.seed alone
    # varies only in-run noise (stochastic family) and NOTHING deterministic.
    # The official campaign worker does exactly this (auto_benchmarking:165).
    m.SPAWN_SEED = seed
    np.random.seed(seed)
    feas = []
    with contextlib.redirect_stdout(io.StringIO()):
        sim = m.HeadlessSim(n_robots=n_robots, behavior_mode=m.SwarmBehavior(CASE),
                            enable_obstacles=False, enable_collision=ENABLE_COLL)
        for _ in range(STEPS):
            sim.step(); feas.append(list(sim.solver_statuses))
    fa = np.array(feas)
    ts = [x for r in sim.solve_times for x in r]
    return dict(n=n_robots, seed=seed, coll=int(sim.total_collisions),
                feas=float(100.0*np.sum(fa==0)/fa.size) if fa.size else 0.0,
                solve_us=float(np.mean(ts)) if ts else 0.0)

if __name__ == "__main__":
    OUT = sys.argv[6]
    with multiprocessing.Pool(int(os.environ.get("SWEEP_WORKERS", "8"))) as pool:
        res = pool.map(run, [tuple(t) for t in TASKS])
    with open(OUT, "w") as fh:
        json.dump(res, fh)
'''


def evaluate(vdir, ns, seeds, workers, family, tag):
    """Run one compiled config in a fresh subprocess; return per-run dicts.

    The worker script and its result file are UNIQUE PER VARIANT: the four
    variant sweeps run as concurrent Slurm jobs from this same directory, and a
    shared scratch path would let one job truncate another's worker mid-exec.
    Results come back through a FILE, not stdout -- acados prints from C on fd 1,
    which contextlib.redirect_stdout cannot capture and which would corrupt a
    stdout-delimited payload.
    """
    import json
    wpath = os.path.join(HERE, f"_sweep_worker_{tag}.py")
    rpath = os.path.join(HERE, f"_sweep_result_{tag}.json")
    with open(wpath, "w") as f:
        f.write(WORKER_SRC)
    if os.path.exists(rpath):
        os.remove(rpath)
    tasks = [[n, s] for n in ns for s in range(seeds)]
    env = dict(os.environ, SWEEP_WORKERS=str(workers), MPLBACKEND="Agg")
    enable_coll = "1"
    r = subprocess.run([PYEXE, wpath, vdir, str(STEPS), "phototaxis", enable_coll,
                        json.dumps(tasks), rpath],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(rpath):
        raise RuntimeError(
            f"eval subprocess failed (rc={r.returncode})\n"
            f"--- stderr tail ---\n{r.stderr[-3000:]}\n"
            f"--- stdout tail ---\n{r.stdout[-1500:]}")
    with open(rpath) as fh:
        return json.load(fh)


def agg(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    n = len(a)
    return (float(a.mean()) if n else 0.0,
            float(1.96 * a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)


G_LO, G_HI = 0.2, 0.95
LF_LO, LF_HI = 0.1, 0.95


def _write_rows(outcsv, fieldnames, variant, gam, lf, d_safe, ns, rows):
    with open(outcsv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for n in ns:
            rs = [r for r in rows if r["n"] == n]
            cm, cc = agg(rs, "coll")
            fm, _ = agg(rs, "feas")
            sm, _ = agg(rs, "solve_us")
            w.writerow(dict(variant=variant, gamma=gam, l_frac=lf,
                            lookahead_dist=round(lf * d_safe, 5),
                            d_handle=round(max(d_safe - lf * d_safe, 0.005), 5),
                            n_robots=n, n_seeds=len(rs), coll_mean=round(cm, 2),
                            coll_ci95=round(cc, 2), feas_mean=round(fm, 3),
                            solve_us_mean=round(sm, 1)))


def run_optuna(args, spec, vdir, ycfg, fam, d_safe):
    """TPE search over (gamma, L/d). Warm-started from the screening CSV so the
    64 grid evaluations already paid for act as completed trials; bounds extend
    past the old grid edges (the screening optima all sat ON a boundary).
    Objective = total collisions over SCREEN_NS at 5 seeds -- identical to the
    screening protocol, so warm-start values and new trials are commensurable.
    Study persists in sqlite -> a resubmitted job resumes the same study.
    """
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.trial import TrialState, create_trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ns = SCREEN_NS[fam]
    seeds = args.seeds
    outdir = os.path.join(HERE, "sweep_results")
    os.makedirs(outdir, exist_ok=True)
    outcsv = os.path.join(outdir, f"sweep_{args.variant}_optuna.csv")
    fieldnames = ["variant", "gamma", "l_frac", "lookahead_dist", "d_handle", "n_robots",
                  "n_seeds", "coll_mean", "coll_ci95", "feas_mean", "solve_us_mean"]
    if not os.path.isfile(outcsv):
        with open(outcsv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    db = os.path.join(outdir, f"optuna_{args.variant}.db")
    study = optuna.create_study(
        study_name=f"gammaL_{args.variant}", storage=f"sqlite:///{db}",
        load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=0))

    dists = {"gamma": FloatDistribution(G_LO, G_HI),
             "l_frac": FloatDistribution(LF_LO, LF_HI)}

    if not study.trials:
        screen = os.path.join(outdir, f"sweep_{args.variant}_screen.csv")
        by_cfg = {}
        if os.path.isfile(screen):
            for r in csv.DictReader(open(screen)):
                k = (float(r["gamma"]), float(r["l_frac"]))
                by_cfg.setdefault(k, 0.0)
                by_cfg[k] += float(r["coll_mean"])
        for (g, lf), tot in sorted(by_cfg.items()):
            study.add_trial(create_trial(params={"gamma": g, "l_frac": lf},
                                         distributions=dists, value=tot,
                                         state=TrialState.COMPLETE))
        print(f"[optuna {args.variant}] warm-started with {len(by_cfg)} screening trials",
              flush=True)
        if by_cfg:
            bg, blf = min(by_cfg, key=by_cfg.get)
            for lf_probe in ({0.15, 0.2} if blf <= 0.35 else {0.93, LF_HI}):
                study.enqueue_trial({"gamma": bg, "l_frac": round(lf_probe, 3)})

    n_done = sum(t.state == TrialState.COMPLETE for t in study.trials)

    def objective(trial):
        g = round(trial.suggest_float("gamma", G_LO, G_HI), 4)
        lf = round(trial.suggest_float("l_frac", LF_LO, LF_HI), 4)
        for k in spec["gamma_keys"]:
            yaml_set(ycfg, k, g)
        set_lookahead(ycfg, spec, round(lf * d_safe, 5))
        t0 = time.time()
        recompile(vdir)
        rows = evaluate(vdir, ns, seeds, args.workers, fam, args.variant)
        _write_rows(outcsv, fieldnames, args.variant, g, lf, d_safe, ns, rows)
        total = 0.0
        for n in ns:
            cm, _ = agg([r for r in rows if r["n"] == n], "coll")
            total += cm
        print(f"  [trial {trial.number}] gamma={g} L/d={lf} -> total={total:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return total

    study.optimize(objective, n_trials=args.optuna)
    best = study.best_trial
    print(f"[optuna {args.variant}] BEST gamma={best.params['gamma']:.4f} "
          f"L/d={best.params['l_frac']:.4f}  total={best.value:.0f} "
          f"({n_done} warm + {len(study.trials)-n_done} live trials)", flush=True)

    if args.verify_best:
        g = round(best.params["gamma"], 4)
        lf = round(best.params["l_frac"], 4)
        print(f"[optuna {args.variant}] verifying best at FULL protocol ...", flush=True)
        for k in spec["gamma_keys"]:
            yaml_set(ycfg, k, g)
        set_lookahead(ycfg, spec, round(lf * d_safe, 5))
        recompile(vdir)
        rows = evaluate(vdir, FULL_NS[fam], 10, args.workers, fam, args.variant)
        vcsv = os.path.join(outdir, f"sweep_{args.variant}_verify.csv")
        if not os.path.isfile(vcsv):
            with open(vcsv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        _write_rows(vcsv, fieldnames, args.variant, g, lf, d_safe, FULL_NS[fam], rows)
        print(f"[optuna {args.variant}] best verified -> {vcsv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--verify", nargs=2, type=float, metavar=("GAMMA", "LFRAC"),
                    help="evaluate ONE config at the full protocol instead of the grid")
    ap.add_argument("--optuna", type=int, default=None, metavar="N_TRIALS",
                    help="run Optuna TPE over (gamma, L/d), warm-started from the "
                         "screening CSV; bounds gamma in [0.2,0.95], L/d in [0.1,0.95]")
    ap.add_argument("--verify-best", action="store_true",
                    help="after --optuna: evaluate the study best at the FULL protocol")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    vdir = os.path.join(SWARM, spec["path"])
    ycfg = os.path.join(vdir, "expert_src", "config", "solver_ellipse_mpc.yaml")
    fam = spec["family"]
    d_safe = yaml_get(ycfg, "d_safe")

    orig = {}
    try:
        head = subprocess.run(["git", "show", f"HEAD:{os.path.relpath(ycfg, REPO)}"],
                              cwd=REPO, capture_output=True, text=True, check=True).stdout
        for k in spec["gamma_keys"] + lookahead_keys(spec):
            m = re.search(rf"^\s+{re.escape(k)}:\s*([0-9.eE+-]+)", head, re.M)
            if not m:
                raise KeyError(k)
            orig[k] = float(m.group(1))
    except Exception as e:
        print(f"[sweep] WARNING: could not read HEAD config ({e}); "
              f"falling back to working tree", flush=True)
        orig = {k: yaml_get(ycfg, k) for k in spec["gamma_keys"] + lookahead_keys(spec)}

    def _bail(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _bail)
    print(f"[sweep {args.variant}] {spec['path']}")
    print(f"[sweep {args.variant}] d_safe={d_safe}  original config: {orig}", flush=True)

    if args.optuna:
        try:
            run_optuna(args, spec, vdir, ycfg, fam, d_safe)
        finally:
            print(f"[sweep {args.variant}] restoring original YAML + solver ...", flush=True)
            for k, v in orig.items():
                yaml_set(ycfg, k, v)
            recompile(vdir)
            print(f"[sweep {args.variant}] restored {orig}", flush=True)
        return

    if args.verify:
        gam, lf = args.verify
        grid = [(gam, lf)]
        ns, seeds, tag = FULL_NS[fam], 10, "verify"
    else:
        grid = list(itertools.product(GAMMAS, L_FRACS))
        ns, seeds, tag = SCREEN_NS[fam], args.seeds, "screen"

    outdir = os.path.join(HERE, "sweep_results")
    os.makedirs(outdir, exist_ok=True)
    outcsv = os.path.join(outdir, f"sweep_{args.variant}_{tag}.csv")
    fieldnames = ["variant", "gamma", "l_frac", "lookahead_dist", "d_handle", "n_robots",
                  "n_seeds", "coll_mean", "coll_ci95", "feas_mean", "solve_us_mean"]
    seen = set()
    if os.path.isfile(outcsv):
        with open(outcsv, newline="") as f:
            for r in csv.DictReader(f):
                seen.add((float(r["gamma"]), float(r["l_frac"])))
        print(f"[sweep {args.variant}] resuming: {len(seen)} configs already done", flush=True)
    else:
        with open(outcsv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    t_start = time.time()
    try:
        for i, (gam, lf) in enumerate(grid, 1):
            if (gam, lf) in seen:
                continue
            L = round(lf * d_safe, 5)
            for k in spec["gamma_keys"]:
                yaml_set(ycfg, k, gam)
            set_lookahead(ycfg, spec, L)
            t0 = time.time()
            recompile(vdir)
            rows = evaluate(vdir, ns, seeds, args.workers, fam, args.variant)
            with open(outcsv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                for n in ns:
                    rs = [r for r in rows if r["n"] == n]
                    cm, cc = agg(rs, "coll")
                    fm, _ = agg(rs, "feas")
                    sm, _ = agg(rs, "solve_us")
                    w.writerow(dict(variant=args.variant, gamma=gam, l_frac=lf,
                                    lookahead_dist=L, d_handle=round(max(d_safe - L, 0.005), 5),
                                    n_robots=n, n_seeds=len(rs), coll_mean=round(cm, 2),
                                    coll_ci95=round(cc, 2), feas_mean=round(fm, 3),
                                    solve_us_mean=round(sm, 1)))
            tot = sum(r["coll"] for r in rows) / max(1, seeds)
            print(f"  [{i:2d}/{len(grid)}] gamma={gam:<4} L={L:<7} (L/d={lf}) "
                  f"D_handle={max(d_safe-L,0.005):.3f}  sum_coll/seed={tot:9.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        print(f"[sweep {args.variant}] restoring original YAML + solver ...", flush=True)
        for k, v in orig.items():
            yaml_set(ycfg, k, v)
        recompile(vdir)
        print(f"[sweep {args.variant}] restored {orig}", flush=True)

    print(f"[sweep {args.variant}] done in {time.time()-t_start:.0f}s -> {outcsv}")


if __name__ == "__main__":
    main()
