"""Renders the doorway simulation."""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})

from video_legend import ROBOT_COLORS, VIOL_COLOR, video_legend_handles

WALL_HALF = 0.45
ROBOT_A, ROBOT_B = 0.03, 0.015
D_SAFE = 0.07
DOOR_HALF = 0.055

TITLES = {
    "phototaxis":  "Phototaxis",
    "orbital":     "Phototactic Orbital",
    "poc":         "Phototaxis Orbital Contracting",
    "no_reynolds": "Phototaxis Without Coupling",
    "no_light":    "Cohesion Without a Stimulus",
    "no_stimulus": "Cohesion Without a Stimulus",
}
ENV_LABELS = {"free": "Obstacle-free", "sparse": "Sparse (2 obstacles)",
              "cluttered": "Cluttered (12 obstacles)"}
ENV_ORDER = ["free", "sparse", "cluttered"]

CMAP_LIGHT = mcolors.LinearSegmentedColormap.from_list(
    "lamp_white",
    ["#ffffff", "#fffde7", "#fff176", "#ffb300", "#e65100", "#b71c1c"])


def draw_frame(ax, d, s, env_label, show_light=True, show_env_label=True,
               show_axes=False, show_door=True):
    ax.clear()
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5) if show_axes else ax.set_ylim(-0.52, 0.5)
    if show_axes:
        # Same axis furniture and sizes as the random-initial-position clips.
        ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
        ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
        ax.set_xlabel("$x$ [m]", color="#333333", fontsize=22)
        ax.set_ylabel("$y$ [m]", color="#333333", fontsize=22)
        ax.tick_params(colors="#333333", labelsize=20)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")

    na = int(d["active"][s])
    pos = d["pos"][s]
    theta = d["theta"][s]

    if show_light and d["heat"].size:
        ax.imshow(d["heat"][s].astype(np.float32), origin="lower",
                  extent=d["heat_extent"], cmap=CMAP_LIGHT, vmin=0, vmax=1.0,
                  interpolation="bilinear", zorder=1, alpha=0.85)
        sp = d["src_pos"][s]
        if np.isfinite(sp).all():
            amp = float(d["src_amp"])
            for r_scale, a in [(0.06, 0.20), (0.03, 0.40), (0.01, 0.80)]:
                ax.add_patch(Circle(sp, r_scale * amp, color="#b45309",
                                    alpha=a, fill=False, linewidth=1.5, zorder=2))
            ax.plot(*sp, "o", color="#b45309", markersize=5, alpha=0.95, zorder=3)

    ax.add_patch(Rectangle((-WALL_HALF, -WALL_HALF), 2 * WALL_HALF, 2 * WALL_HALF,
                           linewidth=1.5, edgecolor="black", facecolor="none",
                           zorder=4))
    # Robots are instantiated 9 cm inside the boundary, not driven in from
    # outside: the OCP bounds position to +/-0.40, so a robot at the wall line
    # would be outside its own state bounds. Drawing an opening would imply a
    # traversal that never happens, so it is off by default here.
    wall = str(d["door_wall"]) if "door_wall" in d else "bottom"
    dpos = d["door_pos"] if "door_pos" in d else np.array([0.0, -WALL_HALF])
    dh = float(d["door_half"]) if "door_half" in d else DOOR_HALF
    if not show_door:
        pass
    elif wall == "left":
        cy = float(dpos[1])
        ax.plot([-WALL_HALF, -WALL_HALF], [cy - dh, cy + dh],
                color="white", lw=3.0, zorder=5, solid_capstyle="butt")
        for yd in (cy - dh, cy + dh):
            ax.plot([-WALL_HALF - 0.018, -WALL_HALF + 0.018], [yd, yd],
                    color="black", lw=2.0, zorder=6)
    else:
        cx = float(dpos[0])
        ax.plot([cx - dh, cx + dh], [-WALL_HALF, -WALL_HALF],
                color="white", lw=3.0, zorder=5, solid_capstyle="butt")
        for xd in (cx - dh, cx + dh):
            ax.plot([xd, xd], [-WALL_HALF - 0.018, -WALL_HALF + 0.018],
                    color="black", lw=2.0, zorder=6)

    for ob in d["obstacles"]:
        ax.add_patch(Circle((ob[0], ob[1]), ob[2], facecolor="#FF0000",
                            edgecolor="#FF0000", linewidth=1.5, alpha=0.85,
                            zorder=7))

    if na == 0:
        return

    apos = pos[:na]
    ath = theta[:na]

    viol = np.zeros(na, dtype=bool)
    if na >= 2:
        diffs = apos[:, None, :] - apos[None, :, :]
        dm = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dm, np.inf)
        viol = (dm < D_SAFE).any(axis=1)

    segs, cols = [], []
    for i in range(na):
        lo = max(int(d["spawn_step"][i]), s - 60)
        if s - lo < 2:
            continue
        h = d["pos"][lo:s + 1, i]
        ss = np.stack([h[:-1], h[1:]], axis=1)
        rgba = np.zeros((len(ss), 4))
        rgba[:, 2] = 1.0
        rgba[:, 3] = 0.05 + 0.55 * (np.arange(len(ss)) / max(len(h), 1))
        segs.extend(ss)
        cols.extend(rgba)
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8,
                                         zorder=5))

    colliding = d["colliding"][s]
    for i in range(na):
        edge = VIOL_COLOR if viol[i] else "#222222"
        ax.add_patch(Ellipse(xy=apos[i], width=ROBOT_A * 2, height=ROBOT_B * 2,
                             angle=np.rad2deg(ath[i]), facecolor=ROBOT_COLORS[0],
                             edgecolor=edge, linewidth=1.2 if viol[i] else 0.6,
                             alpha=0.92, zorder=6))
        if colliding[i]:
            ax.add_patch(Ellipse(xy=apos[i], width=ROBOT_A * 2 * 1.6,
                                 height=ROBOT_B * 2 * 1.6,
                                 angle=np.rad2deg(ath[i]), facecolor="none",
                                 edgecolor="white", linewidth=1.6, alpha=0.9,
                                 zorder=9))

    for i in range(na):
        tip = apos[i] + 0.015 * np.array([np.cos(ath[i]), np.sin(ath[i])])
        ax.annotate("", xy=tip, xytext=apos[i], zorder=7,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2,
                                    alpha=0.9))

    for i in range(na):
        lp = apos[i] - (ROBOT_A * 0.5) * np.array([np.cos(ath[i]),
                                                   np.sin(ath[i])])
        ax.text(lp[0], lp[1], str(i + 1), color="white", fontsize=7,
                ha="center", va="center", fontweight="bold", zorder=8)

    t_now = s * float(d["dt"])
    md = float(d["min_dist"][s]) * 100
    md_str = f"{md:.1f} cm" if np.isfinite(md) else "-"
    hud = (f"$t$ = {t_now:.1f} s   $N$ = {na}   min dist = {md_str}   "
           f"colliding: {int(colliding[:na].sum())} "
           f"(total: {int(d['total_coll'][s])})")
    ax.set_title(f"{env_label}\n{hud}" if show_env_label else hud,
                 fontsize=9 if show_env_label else 19, color="#111111", pad=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True)
    ap.add_argument("--rundir", default=os.path.join(_HERE, "door_runs"))
    ap.add_argument("--outdir", default=os.path.join(
        _HERE, "..", "supplimentary_videos", "video"))
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--env", choices=ENV_ORDER, default=None,
                    help="render this environment alone instead of all three "
                         "side by side")
    ap.add_argument("--size", type=float, default=12.0,
                    help="figure side in inches; the frame is size*dpi px square")
    ap.add_argument("--dpi", type=int, default=100,
                    help="pass --size 10 --dpi 120 to match the N50 clips exactly: "
                         "same 1200 px frame, but a point renders 20 percent larger")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--hide-door", action="store_true",
                    help="draw an unbroken wall; robots are spawned inside "
                         "the arena, so an opening implies a traversal that "
                         "does not happen")
    ap.add_argument("--nmax", type=int, default=137)
    ap.add_argument("--suffix", default="",
                    help="appended to the output file name, e.g. _vmin001")
    ap.add_argument("--label", default="",
                    help="small annotation shown bottom-left, e.g. "
                         "'v_min = 0.01 m/s'")
    args = ap.parse_args()

    envs = [args.env] if args.env else ENV_ORDER
    data = {}
    for env in envs:
        f = os.path.join(args.rundir, f"{args.behavior}_{env}.npz")
        data[env] = dict(np.load(f, allow_pickle=False))
    n_steps = min(int(d["active"].shape[0]) for d in data.values())
    frames = range(0, n_steps, args.stride)
    speed = args.stride * float(data[envs[0]]["dt"]) * args.fps
    sim_s = n_steps * float(data[envs[0]]["dt"])

    if args.env:
        fig, ax_one = plt.subplots(1, 1, figsize=(args.size, args.size),
                                   facecolor="white")
        axes = [ax_one]
        fig.suptitle(f"{TITLES[args.behavior]} - {ENV_LABELS[args.env]}",
                     fontsize=30, color="#111111", y=0.96)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4), facecolor="white")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08,
                            wspace=0.04)
        fig.suptitle(
            f"Distributed Predictive Flocking: {TITLES[args.behavior]} - "
            f"robots enter one by one through a door whenever it is clear",
            fontsize=14, color="#111111", y=0.97)
    if args.label:
        fig.text(0.01, 0.025, args.label, ha="left", va="bottom",
                 color="#333333", fontsize=10, alpha=0.85)

    show_light = args.behavior not in ("no_light", "no_stimulus")



    def animate(k):
        s = k * args.stride
        for ax, env in zip(axes, envs):
            draw_frame(ax, data[env], s, ENV_LABELS[env], show_light,
                       show_env_label=not args.env, show_axes=bool(args.env),
                       show_door=not args.hide_door)
        if args.env:
            # Same call viz_engine makes for its single-panel figure. The axis
            # furniture is identical frame to frame, so once is enough.
            if k == 0:
                fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
                # tight_layout allocates only the margin each side needs, so the
                # wide y-label pushes a square axes right of centre. Recentre it.
                bb = axes[0].get_position()
                axes[0].set_position([0.5 - bb.width / 2.0, bb.y0,
                                      bb.width, bb.height])
        return []

    os.makedirs(args.outdir, exist_ok=True)
    if args.env:
        out = os.path.join(args.outdir,
                           f"spawn_{args.behavior}_{args.env}_N{args.nmax}"
                           f"_{int(round(sim_s))}s_{speed:.0f}x{args.suffix}.mp4")
    else:
        out = os.path.join(args.outdir,
                           f"spawn_{args.behavior}_3env{args.suffix}.mp4")
    writer = animation.FFMpegWriter(
        fps=args.fps, codec="libx264",
        extra_args=["-crf", str(args.crf), "-preset", "medium",
                    "-pix_fmt", "yuv420p"])
    anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                   interval=1000 / args.fps)
    anim.save(out, writer=writer, dpi=args.dpi)
    print(f"saved {out} ({len(frames)} frames, {len(frames)/args.fps:.0f}s, "
          f"{speed:.0f}x)")


if __name__ == "__main__":
    main()
