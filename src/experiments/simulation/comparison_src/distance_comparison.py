#!/usr/bin/env python3
"""Measures how close robots come to each other and to obstacles."""
import os
import sys
import io
import csv
import time
import argparse
import contextlib
import importlib.util
import multiprocessing
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from comparison_config import (COLORS, DISPLAY_NAMES, configure_publication_style,
                               apply_font_target, line_style)
import comparison_config as cfg


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

SAFETY_LEVEL_FOLDERS = [
    ("src/deterministic_bristlebot_swarm/01_hard_bf", "Hard BF"),
    ("src/deterministic_bristlebot_swarm/02_hard_dt_cbf", "Hard Dt-CBF"),
    ("src/deterministic_bristlebot_swarm/03_slacked_bf", "Slack BF"),
    ("src/deterministic_bristlebot_swarm/04_slacked_dt_cbf", "Slack Dt-CBF"),
    ("src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead", "LA Slack Dt-CBF"),
    ("src/deterministic_bristlebot_swarm/06_slacked_dt_hocbf", "Slack Dt-HOCBF"),
]
VARIANTS = [lab for _, lab in SAFETY_LEVEL_FOLDERS]
DIST_NS = [10, 25, 50, 75, 100, 125, 137]
D_SAFE_CM = 7.0
ROBOT_A_CM, ROBOT_B_CM = 3.0, 1.5
OBS_RADIUS_CM = 3.0
RR_KEY = "min_inter_robot_dist_m"
STEPS = 600
CASE_NAME = os.environ.get("DIST_CASE", "phototaxis")
RESULTS_DIR = os.environ.get("DIST_RESULTS_DIR",
                             os.path.join(SCRIPT_DIR, "results_distance"))
_SHARED = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "results"))
_SHARED_CSV = os.path.join(_SHARED, "csv")
_SHARED_FIG = os.path.join(_SHARED, "figures")
os.makedirs(_SHARED_CSV, exist_ok=True)
os.makedirs(_SHARED_FIG, exist_ok=True)


def _worker(args):
    folder, label, n_robots, seed = args
    folder_path = os.path.join(ROOT_DIR, folder)
    sim_engine_dir = os.path.join(folder_path, "sim_engine")
    sim_path = os.path.join(sim_engine_dir, "headless_sim_engine.py")
    orig = list(sys.path)
    sys.path.insert(0, folder_path)
    sys.path.insert(0, sim_engine_dir)
    try:
        for key in list(sys.modules.keys()):
            if any(n in key for n in ["dynamic_sim_module", "headless_sim_engine",
                                      "swarm_dynamics", "light_source_model",
                                      "obstacle_model", "collision_response", "expert_src"]):
                del sys.modules[key]
        spec = importlib.util.spec_from_file_location("dynamic_sim_module", sim_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SPAWN_SEED = seed

        _devnull = os.open(os.devnull, os.O_WRONLY)
        _saved1, _saved2 = os.dup(1), os.dup(2)
        os.dup2(_devnull, 1)
        os.dup2(_devnull, 2)
        try:
            sim = module.HeadlessSim(
                n_robots=n_robots,
                behavior_mode=module.SwarmBehavior(CASE_NAME),
                enable_obstacles=True,
                enable_collision=True,
            )
            mgr = getattr(sim, "obstacle_manager", None)
            obstacles = list(mgr.active_obstacles) if mgr else []
            rr_per_step = []
            ro_per_step = []
            for _ in range(STEPS):
                sim.step()
                rp = np.array([r.pos for r in sim.robots])
                if len(rp) > 1:
                    dmat = np.linalg.norm(rp[:, None, :] - rp[None, :, :], axis=2)
                    np.fill_diagonal(dmat, np.inf)
                    rr_per_step.append(float(dmat.min(axis=1).mean()))
                if obstacles:
                    step_min = min(
                        float((np.linalg.norm(rp - ob.pos, axis=1) - ob.radius).min())
                        for ob in obstacles
                    )
                    ro_per_step.append(step_min)
        finally:
            os.dup2(_saved1, 1)
            os.dup2(_saved2, 2)
            os.close(_devnull)
            os.close(_saved1)
            os.close(_saved2)
        start = STEPS // 2
        rr_ss = rr_per_step[start:] if len(rr_per_step) > start else rr_per_step
        min_rr = float(np.mean(rr_ss)) if rr_ss else float("nan")
        ro_ss = ro_per_step[start:] if len(ro_per_step) > start else ro_per_step
        min_ro = float(np.mean(ro_ss)) if ro_ss else float("nan")
        sys.path = orig
        return {"ok": True, "label": label, "n_robots": n_robots, "seed": seed,
                "min_rr": min_rr, "min_ro": min_ro}
    except Exception as exc:
        sys.path = orig
        return {"ok": False, "label": label, "n_robots": n_robots, "seed": seed, "err": str(exc)}


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    n = a.size
    mean = float(a.mean()) if n else float("nan")
    ci = float(1.96 * a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, ci


def run(seeds, workers):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    partial_path = os.path.join(RESULTS_DIR, "_partial_distances.csv")

    done = set()
    if os.path.isfile(partial_path):
        with open(partial_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["label"], int(r["n_robots"]), int(r["seed"])))
        print(f"[distance] resuming: {len(done)} runs already cached")

    all_tasks = [(folder, lab, n, s)
                 for folder, lab in SAFETY_LEVEL_FOLDERS
                 for n in DIST_NS for s in range(seeds)]
    todo = [t for t in all_tasks if (t[1], t[2], t[3]) not in done]
    n_expected = len(SAFETY_LEVEL_FOLDERS) * len(DIST_NS) * seeds
    print(f"[distance] {len(todo)}/{n_expected} runs to do "
          f"({len(VARIANTS)}x{len(DIST_NS)}x{seeds}), workers={workers}, obstacles ON")

    write_header = not os.path.isfile(partial_path)
    pf = open(partial_path, "a", newline="")
    pw = csv.writer(pf)
    if write_header:
        pw.writerow(["label", "n_robots", "seed", "ok", "min_rr", "min_ro"]); pf.flush()

    t0 = time.time()
    done_now = 0
    if todo:
        with multiprocessing.Pool(processes=workers) as pool:
            for r in pool.imap_unordered(_worker, todo):
                pw.writerow([r["label"], r["n_robots"], r["seed"], int(r.get("ok", False)),
                             r.get("min_rr", ""), r.get("min_ro", "")])
                pf.flush()
                done_now += 1
                if done_now % len(VARIANTS) == 0 or done_now == len(todo):
                    print(f"  [{done_now:4d}/{len(todo)}] (+{len(done)} cached) "
                          f"{time.time()-t0:.0f}s", flush=True)
    pf.close()
    print(f"[distance] compute done in {time.time()-t0:.0f}s")

    grouped = defaultdict(list)
    n_ok = 0
    with open(partial_path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["ok"]):
                grouped[(r["label"], int(r["n_robots"]))].append(r)
                n_ok += 1
    if n_ok < n_expected:
        print(f"[distance] PARTIAL: {n_ok}/{n_expected} ok runs -- re-launch to finish; "
              f"figure below is provisional.")

    def _vals(rs, key):
        return [float(x[key]) for x in rs if x[key] not in ("", "nan")]

    rows = []
    for lab in VARIANTS:
        for n in DIST_NS:
            rs = grouped.get((lab, n), [])
            rr_m, rr_ci = _agg(_vals(rs, "min_rr"))
            ro_m, ro_ci = _agg(_vals(rs, "min_ro"))
            rows.append({"safety_level": lab, "n_robots": n, "n_seeds": len(rs),
                         "min_inter_robot_dist_m": rr_m, "min_inter_robot_dist_m_ci95": rr_ci,
                         "min_robot_obstacle_dist_m": ro_m, "min_robot_obstacle_dist_m_ci95": ro_ci})
    csv_path = os.path.join(_SHARED_CSV, "phototaxis_distances_vs_density.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {csv_path}")
    plot(rows)


def plot(rows, figsize=(7.2, 3.9), box_aspect=1, legend_frame=False, out_base=None,
         font_ref_width=6.997, rect_bottom=0.10):
    scale = apply_font_target(font_ref_width,
                              col_width_in=cfg.CAS_SC_LINEWIDTH_IN)
    configure_publication_style()
    DISP = DISPLAY_NAMES
    plot_variants = list(VARIANTS)
    db = defaultdict(dict)
    for r in rows:
        db[r["safety_level"]][int(r["n_robots"])] = r
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")
    panels = [("min_inter_robot_dist_m", r"$d_\mathrm{rr}$ (cm)", "a"),
              ("min_robot_obstacle_dist_m", r"$d_\mathrm{ro}$ (cm)", "b")]
    for ax, (key, ylab, tg) in zip(axes, panels):
        ax.axvline(137, color="#111111", ls="--", lw=1.1, alpha=0.85, zorder=1.5)
        ref_cm = D_SAFE_CM if key == RR_KEY else D_SAFE_CM - OBS_RADIUS_CM
        ax.axhline(ref_cm, color="#444444", ls="--", lw=1.0, alpha=0.85, zorder=1)
        for lab in plot_variants:
            ns = sorted(db[lab].keys())
            if not ns:
                continue
            c = COLORS.get(lab, "#333333")
            m = np.array([db[lab][n][key] * 100 for n in ns])
            ci = np.array([db[lab][n][key + "_ci95"] * 100 for n in ns])
            ax.plot(ns, m, marker="o", color=c, label=DISP.get(lab, lab), lw=1.8,
                    markersize=4, zorder=3, ls=line_style(lab))
            ax.fill_between(ns, m - ci, m + ci, color=c, alpha=0.18, linewidth=0, zorder=2)
        ax.set_xlim(5, 143)
        lo_cm, hi_cm = ((2 * ROBOT_B_CM, 2 * ROBOT_A_CM) if key == RR_KEY
                        else (ROBOT_B_CM, ROBOT_A_CM))
        ax.axhspan(lo_cm, hi_cm, color="#999999", alpha=0.13, zorder=0)
        ax.set_xticks([10, 25, 50, 75, 100, 137])
        ax.set_xlabel(r"Swarm size $N_b$", labelpad=4, color="#222222")
        ax.set_ylabel(ylab, labelpad=4, color="#222222")
        ax.grid(True, axis="y", which="major", linestyle=":", alpha=0.4, color="#bbbbbb")
        for _side in ("top", "right", "bottom", "left"):
            ax.spines[_side].set_visible(True)
            ax.spines[_side].set_color("#888888")
        ax.tick_params(colors="#333333", labelsize=8 * scale)
        if box_aspect:
            ax.set_box_aspect(box_aspect)
        ax.text(-0.16, 1.02, rf"$\mathbf{{({tg})}}$", transform=ax.transAxes,
                fontsize=9 * scale, va="bottom", ha="right", color="#111111")
        ax.annotate(r"$D_{\mathrm{safe}}$" if key == RR_KEY
                    else r"$D_{\mathrm{safe,obs}}\!-\!r_m$",
                    xy=(103, ref_cm), xytext=(0, -4),
                    textcoords="offset points", fontsize=7 * scale, color="#444444",
                    va="top", ha="right")
    _ylim = axes[0].get_ylim()
    for _ax in axes:
        _ax.set_ylim(*_ylim)
        _ax.set_yticks([0, 2, 4, 6, 8, 10])
    _y_nmax = 0.5 * ((D_SAFE_CM - _ylim[0]) / (_ylim[1] - _ylim[0]) + 1.0)
    for _ax in axes:
        _ax.text(133, _y_nmax, r"$N_{\max}=137$", transform=_ax.get_xaxis_transform(),
                 rotation=90, ha="right", va="center", fontsize=6.5 * scale,
                 color="#111111", alpha=0.9)
    _LEG_NCOL = (len(plot_variants) if len(plot_variants) <= cfg.MAX_SINGLE_ROW_LEGEND
                 else -(-len(plot_variants) // 2))
    # Legend-only dash pattern. The curves keep the compact dashes from LINESTYLES,
    # but at legend scale a 2 pt gap closes up beside the marker and the entry reads
    # solid, hiding the only cue that separates the dashed variants. Widening the gap
    # here leaves every plotted line untouched.
    def _legend_ls(lab):
        return "-" if line_style(lab) == "-" else (0, (4, 4))

    handles = [Line2D([0], [0], color=COLORS.get(l, "#333333"), marker="o", lw=1.8,
                      ls=_legend_ls(l), label=DISP.get(l, l))
               for l in plot_variants]
    fig.tight_layout(rect=[0, rect_bottom, 1, 1.0])
    if legend_frame:
        leg_kw = dict(loc="lower center", ncol=_LEG_NCOL, bbox_to_anchor=(0.5, 0.01),
                      fontsize=cfg.FONTSIZE_LEGEND, markerscale=0.7,
                      handlelength=2.2, handletextpad=0.3, columnspacing=0.6,
                      frameon=False)
    else:
        leg_kw = dict(loc="lower center", ncol=_LEG_NCOL, bbox_to_anchor=(0.5, 0.01),
                      fontsize=cfg.FONTSIZE_LEGEND, markerscale=0.7,
                      handlelength=2.2, handletextpad=0.3, columnspacing=0.6,
                      frameon=False)
    fig.legend(handles=handles, **leg_kw)
    base = out_base if out_base else os.path.join(_SHARED_FIG, "safety_margins")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {base}.png / .pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        # Read back from where the full run writes it (_SHARED_CSV, line ~191).
        # RESULTS_DIR holds only the in-progress checkpoint, so --plot-only could
        # never find a completed campaign's CSV.
        csv_path = os.path.join(_SHARED_CSV, "phototaxis_distances_vs_density.csv")
        rows = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({k: (float(v) if k not in ("safety_level",) else v) for k, v in r.items()})
        plot(rows)
    else:
        run(args.seeds, args.workers)
