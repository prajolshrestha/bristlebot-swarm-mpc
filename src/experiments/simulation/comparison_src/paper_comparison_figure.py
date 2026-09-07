#!/usr/bin/env python3
"""Draws the safety-formulation comparison figure."""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comparison_config import (
    COLORS, configure_publication_style, apply_premium_plot_style,
    add_figure_legend, add_subfigure_label, save_publication_figure,
    apply_font_target, display_name, line_style, CAS_SC_LINEWIDTH_IN,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["Hard BF", "Hard Dt-CBF", "Slack BF",
            "Slack Dt-CBF", "LA Slack Dt-CBF", "Slack Dt-HOCBF"]

_FS = 1.0

FONT_REF_W = 13.008
LEGEND_KW = {"handlelength": 0.9, "handletextpad": 0.25, "columnspacing": 0.45}



def _read_metrics(csv_path):
    """Return {label: {n: {col: float}}} from a metrics_vs_density CSV."""
    db = {v: {} for v in VARIANTS}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = row["safety_level"]
            if label not in db:
                continue
            n = int(row["n_robots"])
            db[label][n] = {k: (float(v) if _isnum(v) else v) for k, v in row.items()}
    return db


def _isnum(s):
    try:
        float(s); return True
    except (TypeError, ValueError):
        return False


def _series(db, label, mean_key, ci_key):
    ns = sorted(db[label].keys())
    m = np.array([db[label][n][mean_key] for n in ns])
    ci_keys = (ci_key,) if isinstance(ci_key, str) else tuple(ci_key)

    def _ci_for(n):
        for k in ci_keys:
            if k in db[label][n]:
                return db[label][n][k]
        return 0.0

    ci = np.array([_ci_for(n) for n in ns])
    return np.array(ns), m, ci


def _panel(ax, db, mean_key, ci_key, ylabel, is_log=False, ci_floor=None, scale=1.0):
    for label in VARIANTS:
        if not db.get(label):
            continue
        ns, m, ci = _series(db, label, mean_key, ci_key)
        m = m * scale
        ci = ci * scale
        c = COLORS[label]
        disp = display_name(label)
        lo = m - ci
        hi = m + ci
        m_plot = m
        if ci_floor is not None:
            lo = np.maximum(lo, ci_floor)
            if is_log:
                m_plot = np.maximum(m, ci_floor)
                hi = np.maximum(hi, ci_floor)
        ax.plot(ns, m_plot, marker="o", color=c, label=disp, lw=2.0, markersize=5,
                ls=line_style(label))
        ax.fill_between(ns, lo, hi, color=c, alpha=0.18, linewidth=0)
    apply_premium_plot_style(ax, "", r"Swarm size $N_b$", ylabel,
                             is_log=is_log, show_legend=False)
    ax.axvline(137, color="#111111", ls="--", lw=1.1, alpha=0.85, zorder=1.5)
    ax.text(133, 0.30, r"$N_{\max}=137$", transform=ax.get_xaxis_transform(),
            rotation=90, ha="right", va="center", fontsize=6.5 * _FS, color="#111111", alpha=0.9)
    ax.set_xlim(5, 143)
    ax.set_xticks([10, 25, 50, 75, 100, 137])
    if is_log and ci_floor is not None:
        ax.set_ylim(bottom=ci_floor * 0.8)


def build_figure(csv_path, out_basename, log_a=True, rect_bottom=0.145):
    global _FS
    _FS = apply_font_target(FONT_REF_W, col_width_in=CAS_SC_LINEWIDTH_IN,
                            ratio_overrides={"legend": 1.0})
    configure_publication_style()
    db = _read_metrics(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor="white", squeeze=False)
    # Panel (a) is solver feasibility and panel (b) the collision count, matching the
    # order the manuscript discusses them in: feasibility first, then collisions.
    _panel(axes[0, 0], db, "feasibility_rate_pct",
           ("feasibility_ci95", "feasibility_rate_pct_ci95"),
           "Feasibility (%)")
    _panel(axes[0, 1], db, "collisions_events_mean", "collisions_events_ci95",
           "Collision count", is_log=log_a, ci_floor=0.2 if log_a else None)
    _panel(axes[0, 2], db, "mean_solve_time_us", "mean_solve_time_us_ci95",
           "Solve time (ms)", scale=1e-3)
    if log_a:
        axes[0, 1].set_yticks([1, 10, 100, 1000, 10000, 100000])
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_yticks([0, 25, 50, 75, 100])
    axes[0, 2].set_ylim(0, 4.2)
    axes[0, 2].set_yticks([0, 1, 2, 3, 4])
    for j in range(3):
        add_subfigure_label(axes[0, j], j, x=-0.10)

    fig.tight_layout(rect=[0, rect_bottom, 1, 0.98])
    add_figure_legend(fig, axes[0, 0], y=0.02, x=0.5, single_row=True,
                      kw_overrides=LEGEND_KW)
    save_publication_figure(fig, out_basename, is_final=True, log_label="comparison")
    plt.close(fig)
    print(f"  wrote {out_basename}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="phototaxis")
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--res-dir", default=None,
                    help="results dir override (e.g. results_parallel_wo_obstacles_600s)")
    args = ap.parse_args()
    if args.res_dir:
        res_dir = args.res_dir if os.path.isabs(args.res_dir) \
            else os.path.join(SCRIPT_DIR, args.res_dir)
    else:
        res_dir = os.path.join(
            SCRIPT_DIR,
            "results_parallel_with_obstacles" if args.obstacles else "results_parallel_wo_obstacles",
        )
    shared = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "csv"))
    csv_path = os.path.join(shared, f"{args.case}_metrics_vs_density.csv")
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(res_dir, "csv_data", f"{args.case}_metrics_vs_density.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    fig_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "figures"))
    os.makedirs(fig_dir, exist_ok=True)
    out_base = os.path.join(fig_dir, "safety_formulation_comparison")
    build_figure(csv_path, out_base)


if __name__ == "__main__":
    main()
