#!/usr/bin/env python3
"""Compares the two contact-resolution schemes side by side."""

import os
import sys
import csv
import time
import argparse
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_V07_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SIM = os.path.join(_V07_ROOT, "sim_engine")
for _p in (_V07_ROOT, _SIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collision_response as _cr1
import collision_response2 as _cr2
from collision_response import (
    _ellipse_overlap_fast, _ellipse_wall_overlaps,
)
import headless_sim_engine as E
from swarm_dynamics import SwarmBehavior

VARIANTS = {
    "v1_teleport":   _cr1.CollisionHandler,
    "v2_pure_steer": _cr2.CollisionHandler,
}
VARIANT_LABEL = {
    "v1_teleport":   "v1 (teleport)",
    "v2_pure_steer": "v2 (pure steer)",
}

DEFAULT_SEEDS   = 10
DEFAULT_NS      = [10, 25, 50, 75]
DEFAULT_STEPS   = E.MAX_STEPS
STEADY_FRACTION = 0.5
PREFILTER_M     = 0.07

RESULTS_ROOT = os.path.join(_HERE, "results")

_CMP_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "..", "comparison", "comparison_code", "deterministic"))
if _CMP_DIR not in sys.path:
    sys.path.insert(0, _CMP_DIR)
try:
    from comparison_config import (
        configure_publication_style as _house_style,
        apply_premium_plot_style,
    )
    _HAVE_HOUSE = True
except Exception:
    _HAVE_HOUSE = False

OKABE_ITO = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "grey":   "#999999",
}
VCOLOR = {"v1_teleport": OKABE_ITO["red"], "v2_pure_steer": OKABE_ITO["blue"]}
VMARK  = {"v1_teleport": "s", "v2_pure_steer": "o"}


def configure_style():
    if _HAVE_HOUSE:
        _house_style()
        return
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "lines.linewidth": 1.6, "axes.linewidth": 0.8,
        "grid.linewidth": 0.4, "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _style_axis(ax, xlabel, ylabel, is_log=False):
    if _HAVE_HOUSE:
        apply_premium_plot_style(ax, "", xlabel, ylabel, is_log=is_log, show_legend=False)
    else:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if is_log:
            ax.set_yscale("log")
        ax.grid(True, which="major", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))


def _simulate(variant, n_robots, seed, steps):
    """Run a single headless phototaxis sim with `variant`'s handler swapped in.

    Returns a dict of per-run metrics.  Metrics tagged "pre-resolution" are captured
    in a wrapper around collision.step (i.e. after the MPC has moved the robots but
    before the handler separates/steers them) so the two handlers are compared on the
    state their full pipeline actually produced, not on v1's post-teleport positions.
    """
    E.CollisionHandler = VARIANTS[variant]
    E.SPAWN_SEED       = int(seed)

    sim = E.HeadlessSim(
        n_robots=int(n_robots),
        behavior_mode=SwarmBehavior.PHOTOTAXIS,
        enable_obstacles=True,
        enable_collision=True,
    )
    obstacles = (list(sim.obstacle_manager.active_obstacles)
                 if sim.obstacle_manager is not None else [])

    n = int(n_robots)
    acc = {
        "rr_frames": 0, "ro_frames": 0, "wall_frames": 0,
        "pen_rr_sum": 0.0, "pen_rr_max": 0.0, "pen_rr_cnt": 0,
        "pen_ro_sum": 0.0, "pen_ro_max": 0.0, "pen_ro_cnt": 0,
        "min_center": float("inf"),
        "tele_total": 0.0, "tele_max": 0.0,
    }
    iu, ju = np.triu_indices(n, k=1) if n > 1 else (np.array([], int), np.array([], int))

    real_step = sim.collision.step

    def wrapped(*a, **k):
        pos = np.array([r.pos for r in sim.robots])
        if n > 1:
            dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
            dpair = dmat[iu, ju]
            acc["min_center"] = min(acc["min_center"], float(dpair.min()))
            for k_idx in np.where(dpair < PREFILTER_M)[0]:
                i, j = int(iu[k_idx]), int(ju[k_idx])
                ri, rj = sim.robots[i], sim.robots[j]
                ov, pen, _ = _ellipse_overlap_fast(
                    ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                    rj.pos, float(rj.theta), rj.bot_width * 0.5, rj.bot_height * 0.5)
                if ov and pen > 0.0:
                    acc["rr_frames"] += 1
                    acc["pen_rr_sum"] += pen
                    acc["pen_rr_cnt"] += 1
                    acc["pen_rr_max"] = max(acc["pen_rr_max"], pen)
        for ri in sim.robots:
            for ob in obstacles:
                ov, pen, _ = _ellipse_overlap_fast(
                    ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                    ob.pos, 0.0, ob.radius, ob.radius)
                if ov and pen > 0.0:
                    acc["ro_frames"] += 1
                    acc["pen_ro_sum"] += pen
                    acc["pen_ro_cnt"] += 1
                    acc["pen_ro_max"] = max(acc["pen_ro_max"], pen)
            acc["wall_frames"] += len(_ellipse_wall_overlaps(
                ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5, E.WALL_HALF))

        before = [r.pos.copy() for r in sim.robots]
        real_step(*a, **k)
        for b, r in zip(before, sim.robots):
            d = float(np.linalg.norm(r.pos - b))
            acc["tele_total"] += d
            acc["tele_max"] = max(acc["tele_max"], d)

    sim.collision.step = wrapped

    fails = 0
    for _ in range(steps):
        sim.step()
        fails += sum(1 for s in sim.solver_statuses if s != 0)

    start = int(steps * STEADY_FRACTION)
    total_ctrl = n * steps
    intens = [np.mean(sim.intensity_log[i][start:]) for i in range(n) if sim.intensity_log[i]]
    speeds = [t for i in range(n) for t in sim.v_log[i]]
    solves = [t for i in range(n) for t in sim.solve_times[i] if t <= 10000.0]

    return {
        "variant": variant,
        "n_robots": n,
        "seed": int(seed),
        "overlap_frames_rr": int(acc["rr_frames"]),
        "overlap_frames_ro": int(acc["ro_frames"]),
        "wall_frames": int(acc["wall_frames"]),
        "max_pen_rr": float(acc["pen_rr_max"]),
        "mean_pen_rr": float(acc["pen_rr_sum"] / acc["pen_rr_cnt"]) if acc["pen_rr_cnt"] else 0.0,
        "max_pen_ro": float(acc["pen_ro_max"]),
        "mean_pen_ro": float(acc["pen_ro_sum"] / acc["pen_ro_cnt"]) if acc["pen_ro_cnt"] else 0.0,
        "min_center_dist": float(acc["min_center"]) if np.isfinite(acc["min_center"]) else 0.0,
        "collision_events": int(sim.total_collisions),
        "teleport_total": float(acc["tele_total"]),
        "teleport_max": float(acc["tele_max"]),
        "mean_intensity_ss": float(np.mean(intens)) if intens else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "fail_rate": float(100.0 * fails / total_ctrl) if total_ctrl else 0.0,
        "mean_solve_us": float(np.mean(solves)) if solves else 0.0,
    }


def _worker(task):
    try:
        return {"ok": True, "task": task, "res": _simulate(**task["kw"])}
    except Exception as exc:
        import traceback
        return {"ok": False, "task": task, "err": f"{exc}\n{traceback.format_exc()}"}


def _run_pool(tasks, workers):
    if workers <= 1:
        return [_worker(t) for t in tasks]
    with mp.Pool(processes=workers) as pool:
        return pool.map(_worker, tasks)


def _agg(values):
    a = np.asarray(values, dtype=float)
    n = a.size
    mean = float(a.mean()) if n else 0.0
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci95


SUMMARY_KEYS = [
    "overlap_frames_rr", "overlap_frames_ro", "wall_frames",
    "max_pen_rr", "mean_pen_rr", "max_pen_ro", "mean_pen_ro",
    "min_center_dist", "collision_events",
    "teleport_total", "teleport_max",
    "mean_intensity_ss", "mean_speed", "fail_rate", "mean_solve_us",
]
BETTER_DIR = {
    "overlap_frames_rr": -1, "overlap_frames_ro": -1, "wall_frames": -1,
    "max_pen_rr": -1, "mean_pen_rr": -1, "max_pen_ro": -1, "mean_pen_ro": -1,
    "min_center_dist": +1, "collision_events": -1,
    "teleport_total": -1, "teleport_max": -1,
    "mean_intensity_ss": +1, "fail_rate": -1,
}


def run_comparison(seeds, ns, steps, workers):
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    tasks = []
    for variant in VARIANTS:
        for n in ns:
            for seed in range(seeds):
                tasks.append({
                    "variant": variant, "n": n, "seed": seed,
                    "kw": dict(variant=variant, n_robots=n, seed=seed, steps=steps),
                })
    print(f"[compare] {len(tasks)} runs "
          f"({len(VARIANTS)} variants x {len(ns)} N x {seeds} seeds, {steps} steps each)")
    t0 = time.time()
    results = _run_pool(tasks, workers)
    print(f"[compare] simulations done in {(time.time() - t0) / 60.0:.1f} min")

    raw_rows = []
    grouped = defaultdict(list)
    n_fail = 0
    for r in results:
        if not r["ok"]:
            n_fail += 1
            print("  ! run failed:", r["err"].splitlines()[-1])
            continue
        raw_rows.append(r["res"])
        grouped[(r["task"]["variant"], r["task"]["n"])].append(r["res"])
    if n_fail:
        print(f"  ! {n_fail} runs failed")
    raw_rows.sort(key=lambda x: (x["variant"], x["n_robots"], x["seed"]))
    _write_csv(os.path.join(RESULTS_ROOT, "comparison_raw.csv"), raw_rows)

    summary_rows = []
    for variant in VARIANTS:
        for n in ns:
            rs = grouped[(variant, n)]
            if not rs:
                continue
            rec = {"variant": variant, "n_robots": n, "n_seeds": len(rs)}
            for key in SUMMARY_KEYS:
                m, _s, ci = _agg([x[key] for x in rs])
                rec[key + "_mean"] = m
                rec[key + "_ci"] = ci
            summary_rows.append(rec)
    _write_csv(os.path.join(RESULTS_ROOT, "comparison_summary.csv"), summary_rows)

    make_plots(summary_rows, ns, RESULTS_ROOT)
    verdict(summary_rows, ns, RESULTS_ROOT)
    return summary_rows


def _series(summary_rows, variant, key):
    rs = sorted([r for r in summary_rows if r["variant"] == variant],
                key=lambda r: r["n_robots"])
    ns = [r["n_robots"] for r in rs]
    m = np.array([r[key + "_mean"] for r in rs])
    ci = np.array([r[key + "_ci"] for r in rs])
    return ns, m, ci


def _plot_metric_vs_n(summary_rows, key, ylabel, fname, out_dir, scale=1.0,
                      logy=False, ci_floor=0.0):
    configure_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for variant in VARIANTS:
        ns, m, ci = _series(summary_rows, variant, key)
        if not ns:
            continue
        m = m * scale
        ci = ci * scale
        c, mk = VCOLOR[variant], VMARK[variant]
        ax.plot(ns, m, marker=mk, color=c, label=VARIANT_LABEL[variant], zorder=3)
        ax.fill_between(ns, np.maximum(m - ci, ci_floor), m + ci,
                        color=c, alpha=0.18, linewidth=0, zorder=2)
    _style_axis(ax, r"Number of robots $N_b$", ylabel, is_log=logy)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out_dir, fname)
    plt.close(fig)


def _plot_scorecard(summary_rows, n_ref, out_dir):
    """Grouped bar chart of v1 vs v2 at a representative density (normalized per metric)."""
    keys = [
        ("overlap_frames_rr", "Overlap RR"),
        ("overlap_frames_ro", "Overlap RO"),
        ("max_pen_rr", "Max pen RR"),
        ("teleport_total", "Teleport"),
        ("fail_rate", "Fail %"),
        ("mean_intensity_ss", "Intensity"),
    ]
    by = {(r["variant"], r["n_robots"]): r for r in summary_rows}
    if not all((v, n_ref) in by for v in VARIANTS):
        return
    configure_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(keys))
    w = 0.38
    for vi, variant in enumerate(VARIANTS):
        vals = []
        for key, _ in keys:
            v1 = by[("v1_teleport", n_ref)][key + "_mean"]
            v2 = by[("v2_pure_steer", n_ref)][key + "_mean"]
            denom = max(abs(v1), abs(v2), 1e-12)
            vals.append(by[(variant, n_ref)][key + "_mean"] / denom)
        ax.bar(x + (vi - 0.5) * w, vals, w, color=VCOLOR[variant],
               label=VARIANT_LABEL[variant], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys], rotation=20, ha="right")
    _style_axis(ax, "", f"Normalized value (N={n_ref})")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "scorecard")
    plt.close(fig)


def make_plots(summary_rows, ns, out_dir):
    _plot_metric_vs_n(summary_rows, "overlap_frames_rr",
                      "Robot-robot overlap-pair-frames", "overlap_rr_vs_n", out_dir,
                      logy=True, ci_floor=0.2)
    _plot_metric_vs_n(summary_rows, "overlap_frames_ro",
                      "Robot-obstacle overlap-frames", "overlap_ro_vs_n", out_dir,
                      logy=True, ci_floor=0.2)
    _plot_metric_vs_n(summary_rows, "max_pen_rr",
                      "Max robot-robot penetration (cm)", "max_pen_rr_vs_n", out_dir,
                      scale=100.0)
    _plot_metric_vs_n(summary_rows, "max_pen_ro",
                      "Max robot-obstacle penetration (cm)", "max_pen_ro_vs_n", out_dir,
                      scale=100.0)
    _plot_metric_vs_n(summary_rows, "teleport_total",
                      "Total injected displacement (cm)", "teleport_total_vs_n", out_dir,
                      scale=100.0)
    _plot_metric_vs_n(summary_rows, "min_center_dist",
                      "Min robot-robot centre distance (cm)", "min_center_dist_vs_n", out_dir,
                      scale=100.0)
    _plot_metric_vs_n(summary_rows, "mean_intensity_ss",
                      "Steady-state mean light intensity", "intensity_vs_n", out_dir)
    _plot_metric_vs_n(summary_rows, "fail_rate",
                      "Solver failure rate (%)", "fail_rate_vs_n", out_dir, ci_floor=0.0)
    n_ref = ns[len(ns) // 2] if ns else None
    if n_ref is not None:
        _plot_scorecard(summary_rows, n_ref, out_dir)
    print(f"[compare] figures -> {out_dir}")


def _pct(v1, v2):
    """Percent change of v2 relative to v1 (negative = v2 lower)."""
    if abs(v1) < 1e-12:
        return 0.0 if abs(v2) < 1e-12 else float("inf")
    return 100.0 * (v2 - v1) / abs(v1)


def verdict(summary_rows, ns, out_dir):
    by = {(r["variant"], r["n_robots"]): r for r in summary_rows}
    lines = []
    lines.append("=" * 78)
    lines.append("VERDICT - collision_response.py (v1) vs collision_response2.py (v2)")
    lines.append("Scenario: PHOTOTAXIS + 2 static obstacles | pre-resolution metrics")
    lines.append("=" * 78)

    pooled = defaultdict(lambda: defaultdict(list))
    for r in summary_rows:
        for key in SUMMARY_KEYS:
            pooled[r["variant"]][key].append(r[key + "_mean"])

    def pooled_mean(variant, key):
        vals = pooled[variant][key]
        return float(np.mean(vals)) if vals else 0.0

    cats = [
        ("SAFETY (robot-robot overlap-frames)", "overlap_frames_rr"),
        ("SAFETY (robot-obstacle overlap-frames)", "overlap_frames_ro"),
        ("SAFETY (max robot-robot penetration)", "max_pen_rr"),
        ("SAFETY (min robot-robot centre dist)", "min_center_dist"),
        ("REALISM (total injected displacement)", "teleport_total"),
        ("LIVENESS (steady-state light intensity)", "mean_intensity_ss"),
        ("LIVENESS (solver failure rate)", "fail_rate"),
    ]
    lines.append(f"\nPooled across N={ns}:")
    lines.append(f"{'category':<44}{'v1':>12}{'v2':>12}{'winner':>10}")
    wins = {"v1_teleport": 0, "v2_pure_steer": 0}
    for label, key in cats:
        v1 = pooled_mean("v1_teleport", key)
        v2 = pooled_mean("v2_pure_steer", key)
        d = BETTER_DIR.get(key, -1)
        if abs(v1 - v2) < 1e-9:
            win = "tie"
        elif (d > 0 and v2 > v1) or (d < 0 and v2 < v1):
            win = "v2"; wins["v2_pure_steer"] += 1
        else:
            win = "v1"; wins["v1_teleport"] += 1
        lines.append(f"{label:<44}{v1:>12.4g}{v2:>12.4g}{win:>10}")

    lines.append("\nPer-N robot-robot overlap-frames (mean ± 95% CI):")
    lines.append(f"{'N':>5}{'v1':>20}{'v2':>20}{'Δ% (v2 vs v1)':>16}")
    for n in ns:
        if ("v1_teleport", n) not in by or ("v2_pure_steer", n) not in by:
            continue
        a = by[("v1_teleport", n)]; b = by[("v2_pure_steer", n)]
        v1 = a["overlap_frames_rr_mean"]; v2 = b["overlap_frames_rr_mean"]
        lines.append(f"{n:>5}"
                     f"{v1:>13.1f}±{a['overlap_frames_rr_ci']:<5.1f}"
                     f"{v2:>13.1f}±{b['overlap_frames_rr_ci']:<5.1f}"
                     f"{_pct(v1, v2):>15.1f}%")

    tele_v1 = pooled_mean("v1_teleport", "teleport_total")
    tele_v2 = pooled_mean("v2_pure_steer", "teleport_total")
    ov_v1 = pooled_mean("v1_teleport", "overlap_frames_rr")
    ov_v2 = pooled_mean("v2_pure_steer", "overlap_frames_rr")
    overall = max(wins, key=wins.get)
    lines.append("\n" + "-" * 78)
    lines.append(f"Category wins:  v1={wins['v1_teleport']}  v2={wins['v2_pure_steer']}")
    lines.append(f"v2 injects {tele_v2*100:.2f} cm vs v1 {tele_v1*100:.2f} cm of teleport "
                 f"(pooled total/run); v2 is the physically-honest one.")
    lines.append(f"v2 robot-robot overlap-frames vs v1: {_pct(ov_v1, ov_v2):+.1f}% "
                 f"(negative = v2 safer on real contacts).")
    lines.append(f"OVERALL (by category count): {VARIANT_LABEL[overall]}")
    lines.append("=" * 78)

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, "verdict.txt"), "w") as f:
        f.write(text + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def _read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def replot_from_csv(ns):
    path = os.path.join(RESULTS_ROOT, "comparison_summary.csv")
    if not os.path.isfile(path):
        print(f"[plots-only] no summary CSV at {path}; run the sweep first.")
        return
    rows = []
    for r in _read_csv(path):
        r["n_robots"] = int(r["n_robots"])
        rows.append(r)
    sweep_ns = ns or sorted({r["n_robots"] for r in rows})
    make_plots(rows, sweep_ns, RESULTS_ROOT)
    verdict(rows, sweep_ns, RESULTS_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS,
                    help="robot counts to sweep")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help="simulation steps per run (600 = 60 s at dt=0.1)")
    ap.add_argument("--workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--plots-only", action="store_true",
                    help="regenerate figures + verdict from existing summary CSV")
    args = ap.parse_args()

    if args.plots_only:
        replot_from_csv(args.ns)
        return

    print(f"seeds={args.seeds}  ns={args.ns}  steps={args.steps}  "
          f"workers={args.workers}  out={RESULTS_ROOT}")
    run_comparison(args.seeds, args.ns, args.steps, args.workers)
    print("\nDONE")


if __name__ == "__main__":
    main()
