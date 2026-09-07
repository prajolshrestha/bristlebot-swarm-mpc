"""Legend entries shared by the demo clips and the montage footer strip."""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROBOT_COLORS   = ["#0072B2"]
VIOL_COLOR     = "#CC00CC"
OBSTACLE_COLOR = "#FF0000"
LIGHT_FACE     = "#ffb300"
LIGHT_EDGE     = "#e65100"


def video_legend_handles(show_obstacles, show_light=True, dash=(4, 3), lw=2.0):
    """Legend handles for one clip, or for the union shown under a montage.

    dash and lw scale the communication-link swatch: a small legend needs a
    finer pattern or the dashes merge into a solid line.
    """
    h = [Line2D([0], [0], color="black", lw=1.5, label="World boundary")]
    if show_light:
        h.append(Patch(facecolor=LIGHT_FACE, edgecolor=LIGHT_EDGE,
                       linewidth=0.6, alpha=0.75, label="Light"))
    h.append(Patch(facecolor=ROBOT_COLORS[0], edgecolor="#222222",
                   linewidth=0.6, label="Robot"))
    if show_obstacles:
        h.append(Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR,
                       linewidth=0.8, label="Static obstacle"))
    h.append(Patch(facecolor=ROBOT_COLORS[0], edgecolor=VIOL_COLOR,
                   linewidth=1.4, label="Active collision"))
    h.append(Line2D([0], [0], color=ROBOT_COLORS[0], lw=lw, alpha=0.7,
                    linestyle=(0, dash), label="Communication link"))
    return h


def save_legend_strip(path, width_px, height_px, show_obstacles, speed_label, dpi=100):
    """Render the montage footer band: one shared legend plus the playback-speed note."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"]})
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), facecolor="white")
    handles = video_legend_handles(show_obstacles, show_light=True)
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.62), ncol=3,
               frameon=False, fontsize=32, handlelength=2.0, handletextpad=0.9,
               columnspacing=3.0, labelspacing=0.7)
    fig.text(0.5, 0.06, speed_label, ha="center", va="bottom",
             fontsize=27, color="#555555", style="italic")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
