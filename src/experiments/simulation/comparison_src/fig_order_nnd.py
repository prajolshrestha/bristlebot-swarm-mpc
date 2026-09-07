"""Local polar order and nearest-neighbour distance, one column per behavior."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matrix_figures as M
from comparison_config import (configure_publication_style, apply_font_target,
                               add_figure_legend, add_subfigure_label,
                               apply_premium_plot_style, CAS_SC_LINEWIDTH_IN)

VARIANT = "05"
D_SAFE_CM = 7.0
BEHS = ["phototaxis", "phototaxis_orbital", "no_light"]
OBS_COLOR = {0: "#0072B2", 2: "#E69F00", 12: "#D55E00"}
OBS_MARK = {0: "o", 2: "s", 12: "^"}
ROWS = [("local_polarization", r"$\Phi_{\rm loc}$", 1.0),
        ("avg_min_sep_m", r"NND (cm)", 100.0)]


def build(by_behavior=True, basename=None, log_label=None):
    """Two rows (order, NND) by three columns.

    by_behavior=True  columns are behaviors, one curve per obstacle count.
    by_behavior=False columns are obstacle counts, one curve per behavior.
    """
    basename = basename or ("order_nnd_by_behavior" if by_behavior
                            else "order_nnd_by_obstacle")
    log_label = log_label or basename.replace("_", "-")
    apply_font_target(12.98, col_width_in=CAS_SC_LINEWIDTH_IN,
                      ratio_overrides={"legend": 1.0})
    configure_publication_style()

    cols = BEHS if by_behavior else M.OBS
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), facecolor="white", sharex=True)
    k = 0
    for i, (key, ylab, mult) in enumerate(ROWS):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            ax.axvline(M.N_MAX, color="#111111", ls="--", lw=1.1, alpha=0.85, zorder=1.5)
            if i == 1:
                ax.axhline(D_SAFE_CM, color="#555555", ls=":", lw=1.2, alpha=0.9,
                           zorder=1.5)
            for sname in (M.OBS if by_behavior else BEHS):
                beh, o = (c, sname) if by_behavior else (sname, c)
                xs, m, ci = M._series(
                    lambda n, b=beh, oo=o: M.steady_metric(VARIANT, b, n, oo, key))
                if not xs:
                    continue
                m, ci = m * mult, ci * mult
                col = OBS_COLOR[o] if by_behavior else M.BEH_COLOR[beh]
                mk = OBS_MARK[o] if by_behavior else M.BEH_MARK[beh]
                lab = M.OBS_LABEL[o] if by_behavior else M.BEH_LABEL[beh]
                ax.plot(xs, m, marker=mk, color=col, lw=2.0, markersize=5,
                        label=lab, zorder=3)
                ax.fill_between(xs, m - ci, m + ci, color=col, alpha=0.18,
                                linewidth=0, zorder=2)
            apply_premium_plot_style(ax, "", r"Swarm size $N_b$" if i == 1 else "",
                                     ylab if j == 0 else "", show_legend=False)
            ax.set_xlim(*M.XLIM)
            ax.set_xticks(M.XTICKS)
            if i == 0:
                ax.set_ylim(0, 1.02)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            else:
                ax.set_ylim(4, 21)
                ax.set_yticks([5, 10, 15, 20])
            add_subfigure_label(ax, k, x=-0.06)
            k += 1

    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    add_figure_legend(fig, axes[0][0], y=0.015, axes_grid=axes.ravel(), single_row=True)
    M._save(fig, basename, False, log_label)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true",
                    help="also build the by-obstacle sibling (not used by v4.4)")
    a = ap.parse_args()
    build(by_behavior=True)
    if a.both:
        build(by_behavior=False)
