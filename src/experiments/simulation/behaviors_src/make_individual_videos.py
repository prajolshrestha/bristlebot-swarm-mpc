"""Build one submission video per behavior and arena, at the Elsevier recommended frame size."""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
VIDEO_DIR = os.path.join(REPO, "src", "experiments", "simulation", "results", "videos")

sys.path.insert(0, os.path.join(
    REPO, "src", "deterministic_bristlebot_swarm",
    "05_slacked_dt_cbf_with_lookahead", "sim_engine"))
from video_legend import video_legend_handles

FRAME_W, FRAME_H = 492, 276
CONTENT = FRAME_H
LEGEND_W = FRAME_W - CONTENT

BEHAVIORS = [
    ("phototaxis",  "(a)", "Phototaxis"),
    ("no_reynolds", "(b)", "Phototaxis without coupling"),
    ("orbital",     "(c)", "Phototactic orbital"),
    ("no_stimulus", "(d)", "Cohesion without a stimulus"),
]

ENVIRONMENTS = [
    ("free",      "Obstacle-free arena",  "no static obstacles"),
    ("sparse",    "Sparse arena",         "2 static obstacles"),
    ("cluttered", "Cluttered arena",      "12 static obstacles"),
]

DESCRIPTIONS = {
    "phototaxis":  ("The swarm stays cohesive and heading-aligned while tracking a light "
                    "source that circles the arena every 10 s."),
    "no_reynolds": ("The Reynolds coupling costs and the avoidance constraints are both "
                    "disabled, so the robots crowd the light peak and collide."),
    "orbital":     ("The light is held stationary and the swarm co-rotates along a level "
                    "set of its intensity field."),
    "no_stimulus": ("No light is present; the Reynolds costs alone hold the group "
                    "together."),
}


def _fig(w_px, h_px, dpi=100):
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"]})
    return plt.figure(figsize=(w_px / dpi, h_px / dpi), facecolor="white")


def title_card(path, n, letter, name, arena, obstacles, show_s, duration_s, speedup):
    """The opening frame: which video this is, what it shows, and the legend."""
    fig = _fig(FRAME_W, FRAME_H)

    def line(y, s, fs, weight="normal", color="#111111", style="normal"):
        fig.text(0.5, y, s, ha="center", va="center", fontsize=fs,
                 fontweight=weight, color=color, style=style)

    line(0.855, f"Supplementary Video {n}", 8.5, "bold", "#555555")
    line(0.645, f"{letter} {name}", 16, "bold")
    line(0.450, f"{arena}, {obstacles}", 9.5, color="#333333")
    line(0.310, "50 robots, 0.8 x 0.8 m, LA Slack Dt-CBF controller", 8, color="#333333")
    line(0.115, f"First {show_s:g} s of a {duration_s:g} s run, played at "
                f"{speedup:g}x real time", 7.5, color="#666666", style="italic")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def legend_panel(path, show_obstacles):
    """The right-hand strip, so the frame's landscape shape is not wasted on padding."""
    fig = _fig(LEGEND_W, FRAME_H)
    handles = video_legend_handles(show_obstacles, show_light=True,
                                   dash=(1.6, 1.2), lw=1.6)
    fig.legend(handles=handles, loc="center", ncol=1, frameon=False, fontsize=6,
               handlelength=2.8, handletextpad=0.5, labelspacing=1.1)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def build(ff, n, beh, env, outdir, duration_s, show_s, speedup, card_s, bitrate_k):
    bkey, letter, bname = beh
    ekey, arena, obstacles = env
    src = os.path.join(VIDEO_DIR, f"demo_{bkey}_{ekey}_N50_{duration_s:g}s_{speedup:g}x.mp4")
    if not os.path.isfile(src) or os.path.getsize(src) == 0:
        print(f"  {n:02d}: missing {os.path.basename(src)}")
        return None

    stem = f"supp_video_{n:02d}_{bkey}_{ekey}"
    out = os.path.join(outdir, stem + ".mp4")
    card = os.path.join(outdir, f"_card_{n:02d}.png")
    strip = os.path.join(outdir, f"_leg_{n:02d}.png")
    show_obs = ekey != "free"
    title_card(card, n, letter, bname, arena, obstacles, show_s, duration_s, speedup)
    legend_panel(strip, show_obs)

    play_s = show_s / speedup
    cmd = [ff, "-v", "error", "-y",
           "-loop", "1", "-t", str(card_s), "-framerate", "25", "-i", card,
           "-t", str(play_s), "-i", src,
           "-loop", "1", "-framerate", "25", "-i", strip,
           "-filter_complex",
           f"[1:v]scale={CONTENT}:{CONTENT}:flags=lanczos[c];"
           f"color=white:{FRAME_W}x{FRAME_H}:d={play_s}:r=25[bg];"
           "[bg][c]overlay=0:0:shortest=1[b1];"
           f"[b1][2:v]overlay={CONTENT}:0:shortest=1,format=yuv420p[body];"
           "[0:v]format=yuv420p[card];"
           "[card][body]concat=n=2:v=1:a=0[out]",
           "-map", "[out]", "-r", "25",
           "-c:v", "libx264", "-profile:v", "high", "-level", "3.1", "-preset", "slow",
           "-b:v", f"{bitrate_k}k", "-maxrate", f"{int(bitrate_k*1.2)}k",
           "-bufsize", f"{bitrate_k*2}k",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    os.remove(card); os.remove(strip)

    png = os.path.join(outdir, stem + "_still.png")
    subprocess.run([ff, "-v", "error", "-y", "-ss", str(play_s * 0.5),
                    "-i", src, "-vframes", "1", png], check=True)
    from PIL import Image
    Image.open(png).convert("RGB").save(os.path.join(outdir, stem + "_still.pdf"),
                                        "PDF", resolution=150.0)

    r = subprocess.run([ff, "-i", out], capture_output=True, text=True).stderr
    dur = 0.0
    for ln in r.splitlines():
        if "Duration:" in ln:
            hh, mm, ss = ln.split("Duration:")[1].split(",")[0].strip().split(":")
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
    mb = os.path.getsize(out) / 1e6
    info = dict(n=n, stem=stem, file=os.path.basename(out), letter=letter, name=bname,
                arena=arena, obstacles=obstacles, env=ekey, behavior=bkey,
                duration_s=dur, size_mb=mb, kbps=mb * 8000 / max(dur, 1e-9),
                desc=DESCRIPTIONS[bkey])
    print(f"  {n:02d}: {info['file']:44s} {FRAME_W}x{FRAME_H} "
          f"{dur:5.1f}s {mb:5.2f} MB {info['kbps']:6.0f} kbps")
    return info


def write_manifest(infos, outdir, duration_s, show_s, speedup):
    lines = ["# Supplementary videos", "",
             f"Twelve videos: four collective behaviors in each of three obstacle "
             f"environments. Every video runs the identical deployed controller "
             f"(LA Slack Dt-CBF); only the task objective and the arena differ. Each opens "
             f"with a title frame carrying the legend, then shows the first {show_s:g} s of "
             f"a {duration_s:g} s simulation at {speedup:g}x real time.", "",
             f"Frame {FRAME_W}x{FRAME_H}, H.264/MP4, 25 fps. Panel letters (a)-(d) match "
             f"the panels of Fig. 5 in the manuscript.", "",
             "| # | File | Behavior | Arena | Duration | Size |",
             "|---|---|---|---|---|---|"]
    for i in infos:
        lines.append(f"| {i['n']} | `{i['file']}` | {i['letter']} {i['name']} | "
                     f"{i['arena']} | {i['duration_s']:.1f} s | {i['size_mb']:.2f} MB |")
    lines += ["", "## Captions", ""]
    for i in infos:
        lines += [f"### Supplementary Video {i['n']} - {i['file']}", "",
                  f"- Still: `{i['stem']}_still.png`, `{i['stem']}_still.pdf`",
                  f"- {FRAME_W}x{FRAME_H}, 25 fps, {i['duration_s']:.1f} s, "
                  f"{i['size_mb']:.2f} MB, {i['kbps']:.0f} kbps, H.264 in MP4", "",
                  f"{i['letter']} {i['name']}, {i['arena'].lower()} with "
                  f"{i['obstacles']}. {i['desc']} Fifty robots in a 0.8 x 0.8 m arena under "
                  f"the deployed LA Slack Dt-CBF controller. The clip shows the first "
                  f"{show_s:g} s of a {duration_s:g} s run, played at {speedup:g}x real "
                  f"time.", ""]
    total = sum(i["size_mb"] for i in infos)
    lines += ["## Package", "",
              f"- {len(infos)} videos, {total:.2f} MB total; largest "
              f"{max(i['size_mb'] for i in infos):.2f} MB",
              "- Elsevier: 150 MB max per file, 5 min max, 15 fps min, 260 kbps min "
              "(750 preferred), recommended frame 492x276", ""]
    p = os.path.join(outdir, "SUPPLEMENTARY.md")
    open(p, "w").write("\n".join(lines))
    print(f"  manifest: {p}  ({total:.2f} MB total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--show-seconds", type=float, default=180)
    ap.add_argument("--speed", type=float, default=5)
    ap.add_argument("--card-seconds", type=float, default=2)
    ap.add_argument("--bitrate-kbps", type=int, default=750)
    ap.add_argument("--outdir", default=os.path.join(VIDEO_DIR, "individual_180s"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    infos, n = [], 0
    for env in ENVIRONMENTS:
        for beh in BEHAVIORS:
            n += 1
            infos.append(build(ff, n, beh, env, a.outdir, a.duration, a.show_seconds,
                               a.speed, a.card_seconds, a.bitrate_kbps))
    if not all(infos):
        sys.exit(1)
    write_manifest(infos, a.outdir, a.duration, a.show_seconds, a.speed)


if __name__ == "__main__":
    main()
