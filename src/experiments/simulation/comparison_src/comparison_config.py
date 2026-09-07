#!/usr/bin/env python3
"""Shared settings and plotting style for every comparison script."""

import numpy as np

from occupancy_config import N_LIST

SIM_TIME_SEC = 60.0

DT = 0.1

STEPS = int(SIM_TIME_SEC / DT)

STEADY_STATE_TIME_SEC = 30.0
STEADY_STATE_STEPS = int(STEADY_STATE_TIME_SEC / DT)

SEEDS = list(range(10))

CASES = [
    {
        "name": "phototaxis",
        "display_name": "Phototaxis",
        "suffix": "phototaxis"
    },
    {
        "name": "phototaxis_orbital_contracting",
        "display_name": "Phototaxis Orbital Contracting",
        "suffix": "phototaxis_orbital_contracting"
    }
]

VARIANT_LABELS = (
    "Hard BF",
    "Hard Dt-CBF",
    "Slack BF",
    "Slack Dt-CBF",
    "LA Slack Dt-CBF",
    "Slack Dt-HOCBF",
)

COLORS = {
    "Hard BF": "#D55E00",
    "Hard Dt-CBF": "#0072B2",
    "Slack BF": "#E69F00",
    "Slack Dt-CBF": "#009E73",
    "LA Slack Dt-CBF": "#009E73",
    "Slack Dt-HOCBF": "#CC79A7",
}

DISPLAY_NAMES = {}


def display_name(label):
    return DISPLAY_NAMES.get(label, label)


LINESTYLES = {
    "Slack Dt-CBF": (0, (5, 2)),
    "Slack Dt-HOCBF": (0, (5, 2)),
}


def line_style(label):
    return LINESTYLES.get(label, "-")

PLOT_DPI = 300
FIGSIZE_SINGLE = (7, 5)
FIGSIZE_DUAL = (14, 5)
FONTSIZE_AXIS = 9
FONTSIZE_PANEL_TITLE = 9.5
FONTSIZE_TICK = 8
FONTSIZE_LEGEND = 8
FONTSIZE_SUBFIGURE = 9
FONTSIZE_SUPTITLE = 11

CAPTION_PT = 9.0
DOC_COL_WIDTH_IN = 6.7
CAS_SC_LINEWIDTH_IN = 6.48
_FONT_RATIOS = {
    "tick": 8.0 / 9.0,
    "title": 9.5 / 9.0,
    "legend": 8.0 / 9.0,
    "subfig": 9.0 / 9.0,
    "suptitle": 11.0 / 9.0,
}


def apply_font_target(fig_width_in, caption_pt=CAPTION_PT, col_width_in=DOC_COL_WIDTH_IN,
                      ratio_overrides=None):
    """Rescale the FONTSIZE_* globals so axis labels render at ``caption_pt`` in
    the compiled PDF, and return the applied scale for figure-local annotation
    sizes. A figure ``fig_width_in`` wide is drawn to ``col_width_in`` in the
    document, so fonts are pre-multiplied by ``fig_width_in / col_width_in``.
    Call before configure_publication_style() and before creating the axes.

    ``ratio_overrides`` replaces individual _FONT_RATIOS entries for this call
    only, e.g. ``{"legend": 1.0}`` to set the legend at the axis-label size
    instead of 8/9 of it. Use it per figure rather than editing _FONT_RATIOS,
    which every figure shares."""
    global FONTSIZE_AXIS, FONTSIZE_TICK, FONTSIZE_PANEL_TITLE
    global FONTSIZE_LEGEND, FONTSIZE_SUBFIGURE, FONTSIZE_SUPTITLE
    ratios = dict(_FONT_RATIOS)
    if ratio_overrides:
        ratios.update(ratio_overrides)
    scale = fig_width_in / col_width_in
    FONTSIZE_AXIS = caption_pt * scale
    FONTSIZE_TICK = FONTSIZE_AXIS * ratios["tick"]
    FONTSIZE_PANEL_TITLE = FONTSIZE_AXIS * ratios["title"]
    FONTSIZE_LEGEND = FONTSIZE_AXIS * ratios["legend"]
    FONTSIZE_SUBFIGURE = FONTSIZE_AXIS * ratios["subfig"]
    FONTSIZE_SUPTITLE = FONTSIZE_AXIS * ratios["suptitle"]
    return scale

CASE_ROW_LABELS = {
    "phototaxis": "Phototaxis",
    "phototaxis_orbital_contracting": "Phototaxis Orbital",
}


def configure_publication_style():
    """ICRA / Science Robotics–friendly matplotlib defaults (serif, vector-ready)."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": FONTSIZE_AXIS,
            "axes.labelsize": FONTSIZE_AXIS,
            "axes.titlesize": FONTSIZE_PANEL_TITLE,
            "xtick.labelsize": FONTSIZE_TICK,
            "ytick.labelsize": FONTSIZE_TICK,
            "legend.fontsize": FONTSIZE_LEGEND,
            "figure.dpi": PLOT_DPI,
            "savefig.dpi": PLOT_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 1.6,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _legend_kwargs():
    return {
        "fontsize": FONTSIZE_LEGEND,
        "frameon": False,
        "handlelength": 2.2,
        "handletextpad": 0.45,
        "columnspacing": 1.4,
        "borderpad": 0.35,
    }


def apply_premium_plot_style(
    ax,
    title,
    xlabel,
    ylabel,
    legend_loc="upper right",
    is_log=False,
    show_legend=True,
    legend_outside=True,
):
    """
    Applies publication-grade, premium styling to a Matplotlib axes object.
    Includes custom grid layouts, modern typography sizes, and smooth styling.

    By default the legend is placed outside the axes (right margin) so it does
    not overlap curves. For multi-panel figures, pass show_legend=False on each
    axis and call add_figure_legend() once on the figure.
    """
    ax.set_title(title, fontsize=FONTSIZE_PANEL_TITLE, fontweight="bold", pad=8, color="#111111")
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS, labelpad=4, color="#222222")
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS, labelpad=4, color="#222222")

    if is_log:
        ax.set_yscale("log")

    ax.grid(True, which="major", linestyle=":", alpha=0.45, color="#bbbbbb")
    for _side in ("top", "right", "bottom", "left"):
        ax.spines[_side].set_visible(True)
        ax.spines[_side].set_color("#888888")

    handles, labels = ax.get_legend_handles_labels()
    if show_legend and handles:
        legend_style = _legend_kwargs()
        if legend_outside:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                **legend_style,
            )
        else:
            ax.legend(loc=legend_loc, **legend_style)

    ax.tick_params(colors="#333333", labelsize=FONTSIZE_TICK)


def apply_premium_3d_plot_style(ax, title, xlabel, ylabel, zlabel):
    """Publication styling for 3-D axes (matches ``apply_premium_plot_style``)."""
    ax.set_title(title, fontsize=FONTSIZE_PANEL_TITLE, fontweight="bold", pad=10, color="#111111")
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS, labelpad=8, color="#222222")
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS, labelpad=8, color="#222222")
    ax.set_zlabel(zlabel, fontsize=FONTSIZE_AXIS, labelpad=8, color="#222222")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#cccccc")
    ax.yaxis.pane.set_edgecolor("#cccccc")
    ax.zaxis.pane.set_edgecolor("#cccccc")
    ax.grid(True, linestyle=":", alpha=0.35, color="#bbbbbb")
    ax.tick_params(colors="#333333", labelsize=FONTSIZE_TICK, pad=2)
    ax.view_init(elev=24, azim=-58)


def add_subfigure_label(ax, index, x=-0.12, y=1.03):
    """
    IEEE-style subfigure tag (a), (b), … at the top-left of a panel.
    Uses mathtext so the label matches LaTeX math-mode typography in the figure.
    """
    tag = chr(ord("a") + index)
    ax.text(
        x,
        y,
        rf"$\mathbf{{({tag})}}$",
        transform=ax.transAxes,
        fontsize=FONTSIZE_SUBFIGURE,
        va="bottom",
        ha="right",
        color="#111111",
    )


def figure_axes_center_x(fig, axes_grid=None) -> float:
    """Horizontal center of a subplot grid (or all axes on ``fig``) in figure coordinates."""
    if axes_grid is not None:
        arr = np.atleast_2d(axes_grid)
        ax_lo = arr[0, 0].get_position()
        ax_hi = arr[-1, -1].get_position()
        return 0.5 * (ax_lo.x0 + ax_hi.x1)
    fig_axes = fig.get_axes()
    if not fig_axes:
        return 0.5
    positions = [a.get_position() for a in fig_axes]
    return 0.5 * (min(p.x0 for p in positions) + max(p.x1 for p in positions))


MAX_SINGLE_ROW_LEGEND = 6


def add_figure_legend(
    fig,
    ax=None,
    handles=None,
    labels=None,
    ncol=None,
    y=0.02,
    x=None,
    axes_grid=None,
    single_row=True,
    kw_overrides=None,
):
    """Shared figure legend below multi-panel plots (avoids per-axis overlap).

    ``kw_overrides`` replaces individual _legend_kwargs() entries for this call,
    e.g. tighter ``columnspacing``/``handlelength`` to narrow a single-row legend.
    That matters for more than looks: a legend wider than the panel row is what
    makes savefig(bbox_inches="tight") widen the canvas, which then scales the
    whole figure -- and its fonts -- down in the document. Narrowing the legend
    lets the font target be calibrated to the saved width."""
    if handles is None or labels is None:
        if ax is None:
            return
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if single_row:
        ncol = len(labels) if len(labels) <= MAX_SINGLE_ROW_LEGEND \
            else -(-len(labels) // 2)
    else:
        ncol = ncol or len(labels)
    anchor_x = x if x is not None else figure_axes_center_x(fig, axes_grid)
    leg_kw = _legend_kwargs()
    if kw_overrides:
        leg_kw.update(kw_overrides)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(anchor_x, y),
        ncol=ncol,
        **leg_kw,
    )


def save_publication_figure(fig, path_without_ext, is_final=False, log_label=None):
    """Save PNG (300 DPI) and vector PDF for a publication-ready figure."""
    png_path = f"{path_without_ext}.png"
    pdf_path = f"{path_without_ext}.pdf"
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    if is_final and log_label:
        import os

        print(f"  ├── Saved: {os.path.basename(png_path)} ({log_label})")
        print(f"  ├── Saved: {os.path.basename(pdf_path)} (vector)")
