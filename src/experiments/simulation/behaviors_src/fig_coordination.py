#!/usr/bin/env python3
"""Draws the coordination order parameters against swarm size."""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import steady_state_matrix_plot as MP
CMP_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "comparison_src"))
if CMP_DIR not in sys.path:
    sys.path.insert(0, CMP_DIR)

_CMP_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "comparison_src"))
if _CMP_SRC not in sys.path:
    sys.path.insert(0, _CMP_SRC)
from comparison_config import (
    configure_publication_style, apply_premium_plot_style, apply_font_target,
    add_figure_legend, add_subfigure_label, save_publication_figure,
    CAS_SC_LINEWIDTH_IN,
)

BEH_COLOR = {"phototaxis": "#ff7f0e", "phototaxis_orbital": "#1f77b4",
             "phototaxis_orbital_contracting": "#2ca02c",
             "no_light": "#7f7f7f"}
BEH_MARK = {"phototaxis": "o", "phototaxis_orbital": "s",
            "phototaxis_orbital_contracting": "^", "no_light": "D"}
BEH_LABEL = {"phototaxis": "Phototaxis", "phototaxis_orbital": "Orbital",
             "no_light": "No light",
             "phototaxis_orbital_contracting": "Orbital (contracting)"}

N_MAX = 137
XTICKS = [10, 50, 100, 137]
XTICKS_MINOR = [25, 75, 125]
XLIM = (5, 143)

TAG_X, TAG_Y = -0.115, 1.03
YTICKS = [0.0, 0.25, 0.50, 0.75, 1.0]

NS = [10, 25, 50, 75, 100, 125, 137]
OBS = 0
WIN = 3000
YLABEL = "Order parameter"
METRICS = [("coherence", None), ("local_polarization", None),
           ("global_nematic", None), ("local_nematic", None)]

BEHAVIORS_SHOWN = ("phototaxis", "phototaxis_orbital", "no_light")
FIG_W = 18.0
FONT_REF_W = 17.07
FIG_H = 5.4
PAPER_FIGS = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "figures"))


def series(beh, key):
    xs, means, cis = [], [], []
    for n in NS:
        runs = MP.load_seeds("05", beh, n, OBS)
        if not runs:
            continue
        vals = np.array([np.nanmean(r[key][-WIN:]) for r in runs])
        xs.append(n)
        means.append(float(vals.mean()))
        cis.append(1.96 * float(vals.std(ddof=1)) / np.sqrt(len(vals)))
    return xs, np.array(means), np.array(cis)


def _require_data():
    """Fail loudly if a plotted behavior has no runs.

    Both failure modes this guards against produced a figure that LOOKED fine:
    a stale RES path made every series empty, and a campaign that never ran
    no_light silently dropped that curve. matplotlib draws empty axes without
    complaint, so the only signal was identical file sizes across obstacle
    counts. Check before drawing instead.
    """
    missing = []
    for beh in BEHAVIORS_SHOWN:
        found = sum(len(MP.load_seeds("05", beh, n, OBS)) for n in NS)
        if found == 0:
            missing.append(beh)
    if missing:
        raise SystemExit(
            f"no runs for {missing} at obstacles={OBS} in {MP.RES}\n"
            f"run the matrix campaign first: "
            f"cd src/experiments/simulation && sbatch jobs/cluster/run_matrix.sh 05")


def main():
    _require_data()
    scale = apply_font_target(FONT_REF_W, col_width_in=CAS_SC_LINEWIDTH_IN,
                              ratio_overrides={"legend": 1.0})
    configure_publication_style()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(FIG_W, FIG_H), facecolor="white", sharey=True)
    for j, (ax, (key, _)) in enumerate(zip(axes, METRICS)):
        ax.axvline(N_MAX, color="#111111", ls="--", lw=1.1, alpha=0.85, zorder=1.5)
        for beh in BEHAVIORS_SHOWN:
            x, m, ci = series(beh, key)
            if not x:
                continue
            ax.plot(x, m, marker=BEH_MARK[beh], color=BEH_COLOR[beh], lw=2.0,
                    markersize=5, label=BEH_LABEL[beh], zorder=3)
            ax.fill_between(x, m - ci, m + ci, color=BEH_COLOR[beh], alpha=0.18,
                            linewidth=0, zorder=2)
        apply_premium_plot_style(ax, "", r"Swarm size $N_b$",
                                 YLABEL if j == 0 else "",
                                 is_log=False, show_legend=False)
        ax.set_xlim(*XLIM)
        ax.set_xticks(XTICKS)
        ax.set_xticks(XTICKS_MINOR, minor=True)
        ax.set_ylim(0, 1.02)
        ax.set_yticks(YTICKS)
        add_subfigure_label(ax, j, x=TAG_X, y=TAG_Y)
    fig.tight_layout(rect=[0.012, 0.125, 1, 0.97])
    add_figure_legend(fig, axes[0], y=0.02, axes_grid=axes, single_row=True)
    os.makedirs(PAPER_FIGS, exist_ok=True)
    base = "coordination_order" if OBS == 0 else f"coordination_order_{OBS}obs"
    save_publication_figure(fig, os.path.join(PAPER_FIGS, base),
                            is_final=True, log_label=f"coordination-order-obs{OBS}")
    plt.close(fig)
    print(f"saved {base} for obs={OBS} (scale={scale:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacles", type=int, default=0, choices=[0, 2, 12],
                    help="obstacle count; 0 is the main-text figure, 2 and 12 the appendix ones")
    OBS = ap.parse_args().obstacles
    main()
