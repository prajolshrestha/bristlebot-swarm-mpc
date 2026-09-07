"""Renders the doorway simulation as a single showcase frame."""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt
from matplotlib import animation

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from spawn_render import draw_frame, ENV_LABELS, TITLES

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to the door-spawn .npz")
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--stride", type=int, default=2,
                    help="control steps per rendered frame (2 -> 6x real time)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--size", type=float, default=9.0,
                    help="square figure side in inches (9 x 120dpi = 1080 px)")
    ap.add_argument("--env", default="free", help="env label for the panel")
    ap.add_argument("--behavior", default="phototaxis", help="title behavior key")
    ap.add_argument("--label", default="", help="annotation, bottom-left")
    args = ap.parse_args()

    d = dict(np.load(args.run, allow_pickle=False))
    n_steps = int(d["active"].shape[0])
    frames = range(0, n_steps, args.stride)
    speed = args.stride * float(d["dt"]) * args.fps
    nmax = int(d["nmax"])

    fig, ax = plt.subplots(figsize=(args.size, args.size), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
    fig.suptitle(
        f"Distributed Predictive Flocking: {TITLES.get(args.behavior, args.behavior)} - "
        f"up to {nmax} robots enter one by one through a door",
        fontsize=15, color="#111111", y=0.975)
    fig.text(0.98, 0.015, f"{speed:.0f}x real-time", ha="right", va="bottom",
             color="#333333", fontsize=11, alpha=0.85)
    if args.label:
        fig.text(0.02, 0.015, args.label, ha="left", va="bottom",
                 color="#333333", fontsize=11, alpha=0.85)

    show_light = args.behavior != "no_light"

    def animate(k):
        draw_frame(ax, d, k * args.stride, ENV_LABELS.get(args.env, args.env),
                   show_light)
        return []

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=args.fps, codec="libx264",
        extra_args=["-crf", "24", "-preset", "medium", "-pix_fmt", "yuv420p"])
    anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                   interval=1000 / args.fps)
    anim.save(args.out, writer=writer, dpi=args.dpi)
    print(f"saved {args.out} ({len(frames)} frames, "
          f"{len(frames)/args.fps:.0f}s video, {speed:.0f}x, "
          f"{n_steps*float(d['dt']):.0f}s simulated, N_max={nmax})")


if __name__ == "__main__":
    main()
