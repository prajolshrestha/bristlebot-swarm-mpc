"""Arranges the behavior snapshots into one sheet per environment."""
import os
import sys
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import plot_behaviors_for_paper as pb

_CMP_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "comparison_src"))
if _CMP_SRC not in sys.path:
    sys.path.insert(0, _CMP_SRC)
import comparison_config as cfg
from comparison_config import (add_figure_legend, add_subfigure_label,
                               configure_publication_style, apply_font_target)
from collect_behavior_obstacle_snapshots import _ckpt_path, SNAPSHOT_TIMES_SEC, SPAN_TAG

OUT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "results", "figures"))
N = 50
BEHAVIORS = [
    ("phototaxis", "Phototaxis"),
    ("phototaxis_no_reynolds", "No Reynolds"),
    ("phototaxis_orbital", "Phototaxis orbital"),
    ("no_light", "No light"),
]
ENVIRONMENTS = [(0, "free"), (2, "2obs"), (12, "12obs")]

FONT_REF_WIDTH = 13.2
FIG_H = 12.6
RECT_BOTTOM = 0.055


def render_env(count, tag):
    steps = tuple(int(round(t / pb.DT)) for t in SNAPSHOT_TIMES_SEC)
    rows, missing = [], []
    for beh, label in BEHAVIORS:
        cp = _ckpt_path(beh, count, N)
        if os.path.exists(cp):
            with open(cp, "rb") as f:
                rows.append((beh, label, pickle.load(f)))
        else:
            missing.append((beh, cp))
    if not rows:
        return None
    if missing:
        names = ", ".join(b for b, _ in missing)
        raise SystemExit(
            f"missing snapshot checkpoint(s) for obstacles={count}: {names}\n"
            + "\n".join(f"    {p}" for _, p in missing)
            + "\nRendering would silently drop those rows while the filename still "
              "claims every behavior. Collect them first:\n"
              "    cd src/experiments/simulation && sbatch jobs/cluster/run_snapshots.sh")

    n_rows, n_cols = len(rows), len(steps)
    fig_w, _ = pb._snapshot_grid_figsize(n_rows, n_cols)
    apply_font_target(FONT_REF_WIDTH)
    pb.FONTSIZE_AXIS = cfg.FONTSIZE_AXIS
    pb.FONTSIZE_PANEL_TITLE = cfg.FONTSIZE_PANEL_TITLE
    pb.FONTSIZE_TICK = cfg.FONTSIZE_AXIS * 0.62
    cfg.FONTSIZE_LEGEND = cfg.FONTSIZE_AXIS
    configure_publication_style()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, FIG_H),
                             facecolor="white", squeeze=False)
    for ri, (beh, label, snaps) in enumerate(rows):
        for col, snap in enumerate(snaps):
            ax = axes[ri, col]
            lab = pb._grid_panel_axis_labels(row=ri, col=col, n_rows=n_rows,
                                             steps=steps, step_idx=col)
            if col == 0:
                lab["row_ylabel"] = label
            pb._style_arena_panel(ax, **lab)
            if beh == "no_light":
                snap = {**snap, "heatmap": None}
            pb._draw_snapshot_content(ax, snap)
            ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
            ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
            if col == 0:
                add_subfigure_label(ax, ri, x=-0.24)

    handles = pb._snapshot_legend_handles(show_obstacles=(count > 0), show_light=True)
    fig.tight_layout(rect=[0.17, RECT_BOTTOM, 1, 0.98])
    add_figure_legend(fig, handles=handles, labels=[h.get_label() for h in handles],
                      x=pb._subplot_block_center_x(axes), y=0.012, single_row=True)

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f"behavior_snapshots_{tag}")
    pb._export_figure(fig, base, ("png", "pdf"), is_final=True,
                      log_label=f"{tag} 4beh N{N} {n_rows}x{n_cols}")
    plt.close(fig)
    return base


def main():
    for count, tag in ENVIRONMENTS:
        base = render_env(count, tag)
        print(f"  wrote {os.path.basename(base)}.png/.pdf" if base else f"  [skip] {tag}", flush=True)
    print(f"[grouped] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
