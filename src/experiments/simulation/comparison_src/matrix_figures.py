#!/usr/bin/env python3
"""Draws the collision figures and writes the matrix summary."""
import os
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comparison_config import (
    COLORS, display_name, line_style, configure_publication_style,
    apply_premium_plot_style, apply_font_target, add_figure_legend,
    add_subfigure_label, save_publication_figure, CAS_SC_LINEWIDTH_IN,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MATRIX_OUT_DIR",
                     os.path.join(SCRIPT_DIR, "..", "results", "csv", "matrix"))
PAPER_FIGS = os.path.join(SCRIPT_DIR, "..", "results", "figures")
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "results", "figures")

STEPS = 9000
WIN = 3000
COLL_FLOOR = 0.2
MIN_SEEDS = int(os.environ.get("MATRIX_MIN_SEEDS", "10"))
_SHORT = []

VARIANTS = ["01", "02", "03", "04", "05", "06"]
VARIANT_LABEL = {"01": "Hard BF", "02": "Hard Dt-CBF", "03": "Slack BF",
                 "04": "Slack Dt-CBF", "05": "LA Slack Dt-CBF",
                 "06": "Slack Dt-HOCBF"}
NS = [10, 25, 50, 75, 100, 125, 137]
OBS = [0, 2, 12]
OBS_LABEL = {0: "Free space", 2: "Few obstacles (2)", 12: "Cluttered (12)"}

BEHAVIORS = ["phototaxis", "phototaxis_orbital"]
BEH_LABEL = {"phototaxis": "Phototaxis", "phototaxis_orbital": "Orbital",
             "phototaxis_orbital_contracting": "Orbital (contracting)",
             "no_light": "No light"}
BEH_COLOR = {"phototaxis": "#D55E00", "phototaxis_orbital": "#0072B2",
             "phototaxis_orbital_contracting": "#009E73", "no_light": "#CC79A7"}
BEH_MARK = {"phototaxis": "o", "phototaxis_orbital": "^",
            "phototaxis_orbital_contracting": "D", "no_light": "s"}

OBS_COLOR = {0: "#6BAED6", 2: "#2171B5", 12: "#08306B"}
OBS_MARK = {0: "o", 2: "^", 12: "s"}

LO_B, HI_B, N_MAX = 25, 75, 137
XTICKS = [10, 25, 50, 75, 100, 137]
XTICKS_NARROW = [10, 50, 100, 137]
XLIM = (5, 143)


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def load_seeds(variant, beh, n, obs, cols=None):
    """All seed runs for one grid point -> [{column: ndarray}]. Missing points
    return [] so a partially finished campaign still plots what exists."""
    runs = []
    pat = os.path.join(RES, f"v{variant}_{beh}_N{n}_obs{obs}_reacon_seed*_steps{STEPS}.csv")
    for p in sorted(glob.glob(pat)):
        rows = list(csv.DictReader(open(p, newline="")))
        if not rows:
            continue
        keys = cols or rows[0].keys()
        runs.append({k: np.array([_f(r[k]) for r in rows]) for k in keys})
    return runs


def _mean_ci(vals):
    vals = np.asarray(vals, float)
    m = float(np.nanmean(vals))
    if len(vals) < 2:
        return m, 0.0
    return m, 1.96 * float(np.nanstd(vals, ddof=1)) / np.sqrt(len(vals))


def final_collisions(variant, beh, n, obs):
    """Distinct contacts over the whole run = last cum_collisions sample."""
    runs = load_seeds(variant, beh, n, obs, cols=["cum_collisions"])
    if len(runs) < MIN_SEEDS:
        if runs:
            _SHORT.append((variant, beh, n, obs, len(runs)))
        return None
    return _mean_ci([r["cum_collisions"][-1] for r in runs])


def steady_metric(variant, beh, n, obs, key):
    """Seed mean/CI of a metric averaged over the steady-state window."""
    runs = load_seeds(variant, beh, n, obs, cols=[key])
    if len(runs) < MIN_SEEDS:
        if runs:
            _SHORT.append((variant, beh, n, obs, len(runs)))
        return None
    return _mean_ci([np.nanmean(r[key][-WIN:]) for r in runs])


def _series(fn):
    """fn(n) -> (mean, ci) or None; skips grid points with no data yet."""
    xs, ms, cis = [], [], []
    for n in NS:
        got = fn(n)
        if got is None:
            continue
        xs.append(n)
        ms.append(got[0])
        cis.append(got[1])
    return xs, np.array(ms), np.array(cis)


def _decorate(ax, scale, j, ylabel, is_log=False, narrow=False):
    ax.axvline(N_MAX, color="#111111", ls="--", lw=1.1, alpha=0.85, zorder=1.5)
    if not narrow:
        ax.text(133, 0.30, r"$N_{\max}=137$", transform=ax.get_xaxis_transform(),
                rotation=90, ha="right", va="center", fontsize=6.5 * scale,
                color="#111111", alpha=0.9)
    apply_premium_plot_style(ax, "", r"Swarm size $N_b$", ylabel,
                            is_log=is_log, show_legend=False)
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS_NARROW if narrow else XTICKS)
    add_subfigure_label(ax, j, x=-0.06)


def fig_collisions(variant="05", copy_to_paper=False):
    """Fig 6: behaviours x obstacle environments for the deployed variant."""
    scale = apply_font_target(12.98, col_width_in=CAS_SC_LINEWIDTH_IN,
                              ratio_overrides={"legend": 1.0})
    configure_publication_style()
    behs = BEHAVIORS + ["no_light"]

    data = {}
    for beh in behs:
        per_obs = {}
        for o in OBS:
            xs, m, ci = _series(lambda n, b=beh, o=o: final_collisions(variant, b, n, o))
            if xs:
                per_obs[o] = (xs, m, ci)
        if per_obs:
            data[beh] = per_obs
    if not data:
        print(f"[collisions] no data yet in {RES}")
        return

    y_peak = COLL_FLOOR
    for per_obs in data.values():
        for xs, m, ci in per_obs.values():
            y_peak = max(y_peak, float(np.nanmax(np.maximum(m + ci, COLL_FLOOR))))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor="white", sharey=True)
    for j, o in enumerate(OBS):
        ax = axes[j]
        _decorate(ax, scale, j, "Collision count" if j == 0 else "", is_log=True)
        for beh, per_obs in data.items():
            if o not in per_obs:
                continue
            xs, m, ci = per_obs[o]
            ax.plot(xs, np.maximum(m, COLL_FLOOR), marker=BEH_MARK[beh],
                    color=BEH_COLOR[beh], lw=2.0, markersize=5,
                    label=BEH_LABEL[beh], zorder=3)
            ax.fill_between(xs, np.maximum(m - ci, COLL_FLOOR),
                            np.maximum(m + ci, COLL_FLOOR), color=BEH_COLOR[beh],
                            alpha=0.18, linewidth=0, zorder=2)
        ax.set_ylim(bottom=COLL_FLOOR * 0.8, top=y_peak * 3.5)
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
    fig.tight_layout(rect=[0, 0.145, 1, 0.97])
    add_figure_legend(fig, axes[0], y=0.02, axes_grid=axes, single_row=True)
    _save(fig, "behavior_collisions", copy_to_paper, "behaviors-3env-900s")


def fig_collisions_by_behavior(variant="05", copy_to_paper=False):
    """Fig 6 transposed: one panel per behaviour with the three obstacle
    environments overlaid, so obstacle density is compared within a behaviour
    rather than across panels. Obstacle count is ordinal, so the three
    environments take a single-hue light-to-dark ramp plus distinct markers."""
    scale = apply_font_target(12.98, col_width_in=CAS_SC_LINEWIDTH_IN,
                              ratio_overrides={"legend": 1.0})
    configure_publication_style()
    behs = BEHAVIORS + ["no_light"]

    data = {}
    for beh in behs:
        per_obs = {}
        for o in OBS:
            xs, m, ci = _series(lambda n, b=beh, o=o: final_collisions(variant, b, n, o))
            if xs:
                per_obs[o] = (xs, m, ci)
        if per_obs:
            data[beh] = per_obs
    if not data:
        print(f"[collisions2] no data yet in {RES}")
        return

    y_peak = COLL_FLOOR
    for per_obs in data.values():
        for xs, m, ci in per_obs.values():
            y_peak = max(y_peak, float(np.nanmax(np.maximum(m + ci, COLL_FLOOR))))

    drawn = [b for b in behs if b in data]
    fig, axes = plt.subplots(1, len(drawn), figsize=(13.5, 4.0), facecolor="white",
                             sharey=True, squeeze=False)
    axes = axes[0]
    for j, beh in enumerate(drawn):
        ax = axes[j]
        _decorate(ax, scale, j, "Collision count" if j == 0 else "", is_log=True)
        ax.set_title(BEH_LABEL[beh], fontsize=8 * scale)
        for o in OBS:
            if o not in data[beh]:
                continue
            xs, m, ci = data[beh][o]
            ax.plot(xs, np.maximum(m, COLL_FLOOR), marker=OBS_MARK[o],
                    color=OBS_COLOR[o], lw=2.0, markersize=5,
                    label=OBS_LABEL[o], zorder=3)
            ax.fill_between(xs, np.maximum(m - ci, COLL_FLOOR),
                            np.maximum(m + ci, COLL_FLOOR), color=OBS_COLOR[o],
                            alpha=0.18, linewidth=0, zorder=2)
        ax.set_ylim(bottom=COLL_FLOOR * 0.8, top=y_peak * 3.5)
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
    fig.tight_layout(rect=[0, 0.145, 1, 0.97])
    add_figure_legend(fig, axes[0], y=0.02, axes_grid=axes, single_row=True)
    _save(fig, "behavior_collisions_by_behavior", copy_to_paper,
          "behaviors-by-obstacle-900s")


def fig_coordination(variant="05", obs=0, copy_to_paper=False):
    """Fig 7: steady-state order parameters for the deployed variant."""
    metrics = [("coherence", r"Global polarization $\Phi$"),
               ("local_polarization", r"Local polarization $\Phi_{\rm loc}$"),
               ("local_nematic", r"Local nematic order $S_{\rm loc}$"),
               ("global_nematic", r"Global nematic order $S_{\rm glob}$")]
    fig_w = 18.0
    scale = apply_font_target(17.1)
    configure_publication_style()

    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_w, 4.0),
                             facecolor="white", sharey=True)
    drew = False
    for j, (ax, (key, _lbl)) in enumerate(zip(axes, metrics)):
        _decorate(ax, scale, j, "Order parameter" if j == 0 else "", narrow=True)
        for beh in BEHAVIORS:
            xs, m, ci = _series(lambda n, b=beh: steady_metric(variant, b, n, obs, key))
            if not xs:
                continue
            drew = True
            ax.plot(xs, m, marker=BEH_MARK[beh], color=BEH_COLOR[beh], lw=2.0,
                    markersize=5, label=BEH_LABEL[beh], zorder=3)
            ax.fill_between(xs, m - ci, m + ci, color=BEH_COLOR[beh], alpha=0.18,
                            linewidth=0, zorder=2)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    if not drew:
        print(f"[coordination] no data yet in {RES}")
        plt.close(fig)
        return
    fig.tight_layout(rect=[0.02, 0.145, 1, 0.97])
    add_figure_legend(fig, axes[0], y=0.02, axes_grid=axes, single_row=True)
    _save(fig, "coordination_order", copy_to_paper, "coordination-order-900s")


def fig_methods(metric="cum_collisions", copy_to_paper=False):
    """Cross-variant comparison: behaviour rows x obstacle columns, one curve
    per formulation. `metric` is either cum_collisions (whole-run contacts) or
    any per-step column (steady-state window mean)."""
    is_coll = (metric == "cum_collisions")
    scale = apply_font_target(13.5)
    configure_publication_style()

    fig, axes = plt.subplots(len(BEHAVIORS), len(OBS), figsize=(13.5, 11.0),
                             facecolor="white", sharex=True,
                             sharey=not is_coll, squeeze=False)
    drew = False
    for i, beh in enumerate(BEHAVIORS):
        for j, o in enumerate(OBS):
            ax = axes[i][j]
            for v in VARIANTS:
                lab = VARIANT_LABEL[v]
                if is_coll:
                    xs, m, ci = _series(lambda n, v=v, b=beh, o=o: final_collisions(v, b, n, o))
                else:
                    xs, m, ci = _series(lambda n, v=v, b=beh, o=o: steady_metric(v, b, n, o, metric))
                if not xs:
                    continue
                drew = True
                c = COLORS[lab]
                lo = np.maximum(m - ci, COLL_FLOOR) if is_coll else m - ci
                hi = np.maximum(m + ci, COLL_FLOOR) if is_coll else m + ci
                ax.plot(xs, np.maximum(m, COLL_FLOOR) if is_coll else m,
                        marker="o", color=c, lw=1.8, markersize=4,
                        ls=line_style(lab), label=display_name(lab), zorder=3)
                ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0, zorder=2)
            ylab = ("Collision count" if is_coll else metric.replace("_", " ")) if j == 0 else ""
            _decorate(ax, scale, i * len(OBS) + j, ylab, is_log=is_coll)
            if i == 0:
                ax.set_title(OBS_LABEL[o], fontsize=8 * scale)
            if j == len(OBS) - 1:
                ax.text(1.02, 0.5, BEH_LABEL[beh], transform=ax.transAxes,
                        rotation=270, va="center", ha="left", fontsize=7.5 * scale)
    if not drew:
        print(f"[methods] no data yet in {RES}")
        plt.close(fig)
        return
    fig.tight_layout(rect=[0, 0.075, 1, 0.97])
    add_figure_legend(fig, axes[0][0], y=0.02, axes_grid=axes[0], single_row=True)
    _save(fig, f"methods_900s_{metric}", copy_to_paper and metric == "cum_collisions",
          f"methods-8variant-{metric}")


def summary_csv():
    """Flat per-(variant, behaviour, N, obstacles) table: the numbers the paper
    text and tables quote, so no value is transcribed by hand."""
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, "matrix900_summary.csv")
    keys = ["min_dist_m", "avg_min_sep_m", "mean_dist_m", "coherence",
            "local_polarization", "local_nematic", "global_nematic",
            "comm_neighbor_mean_dist_m", "safety_violations", "mean_path_len_m"]
    n_rows = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "label", "behavior", "n_robots", "obstacles", "seeds",
                    "collisions_mean", "collisions_ci95"]
                   + [f"{k}_{s}" for k in keys for s in ("mean", "ci95")])
        for v in VARIANTS:
            for beh in BEHAVIORS + ["no_light"]:
                for o in OBS:
                    for n in NS:
                        runs = load_seeds(v, beh, n, o)
                        if not runs:
                            continue
                        cm, cc = _mean_ci([r["cum_collisions"][-1] for r in runs])
                        row = [v, VARIANT_LABEL[v], beh, n, o, len(runs),
                               f"{cm:.2f}", f"{cc:.2f}"]
                        for k in keys:
                            m, ci = _mean_ci([np.nanmean(r[k][-WIN:]) for r in runs])
                            row += [f"{m:.5f}", f"{ci:.5f}"]
                        w.writerow(row)
                        n_rows += 1
    print(f"wrote {out} ({n_rows} rows)")
    return out


def _save(fig, basename, copy_to_paper, log_label):
    os.makedirs(FIG_DIR, exist_ok=True)
    save_publication_figure(fig, os.path.join(FIG_DIR, basename),
                            is_final=True, log_label=log_label)
    if copy_to_paper:
        os.makedirs(PAPER_FIGS, exist_ok=True)
        import shutil
        for ext in ("png", "pdf"):
            src = os.path.join(FIG_DIR, f"{basename}.{ext}")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(PAPER_FIGS, f"{basename}.{ext}"))
        print(f"  -> copied {basename}.(png|pdf) into {PAPER_FIGS}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collisions", action="store_true")
    ap.add_argument("--collisions2", action="store_true")
    ap.add_argument("--coordination", action="store_true")
    ap.add_argument("--methods", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", default="05", help="deployed variant for figs 6/7")
    ap.add_argument("--copy-to-paper", action="store_true",
                    help="also refresh paper1_v2/figs/ with the final PDFs/PNGs")
    a = ap.parse_args()
    if not any((a.collisions, a.collisions2, a.coordination, a.methods,
                a.summary, a.all)):
        a.all = True
    if a.all or a.collisions:
        fig_collisions(a.variant, a.copy_to_paper)
    if a.all or a.collisions2:
        fig_collisions_by_behavior(a.variant, a.copy_to_paper)
    if a.all or a.coordination:
        fig_coordination(a.variant, 0, a.copy_to_paper)
    if a.all or a.methods:
        for metric in ("cum_collisions", "min_dist_m", "local_nematic", "coherence"):
            fig_methods(metric, a.copy_to_paper)
    if a.all or a.summary:
        summary_csv()
    if _SHORT:
        uniq = sorted(set(_SHORT))
        print(f"\n[incomplete] {len(uniq)} grid point(s) had <{MIN_SEEDS} seeds and were "
              f"OMITTED from the figures:")
        for v, b, n, o, k in uniq[:12]:
            print(f"    v{v} {b} N={n} obs={o}: {k}/{MIN_SEEDS} seeds")
        if len(uniq) > 12:
            print(f"    ... and {len(uniq) - 12} more")
        print("  -> these figures are NOT final; rerun once the campaign completes.")


if __name__ == "__main__":
    main()
