"""Build one-row montages at the Elsevier recommended frame size, legend along the bottom."""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg
from PIL import Image

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
VIDEO_DIR = os.path.join(REPO, "src", "experiments", "simulation", "results", "videos")

sys.path.insert(0, os.path.join(
    REPO, "src", "deterministic_bristlebot_swarm",
    "05_slacked_dt_cbf_with_lookahead", "sim_engine"))
from video_legend import video_legend_handles

FRAME_W, FRAME_H = 492, 276
PANEL = FRAME_W // 4
LABEL_H = 26
LEGEND_H = FRAME_H - LABEL_H - PANEL

CROP_SIDE, CROP_X, CROP_Y = 830, 187, 218

PANELS = [
    ("phototaxis",  "(a)", "Phototaxis"),
    ("no_reynolds", "(b)", "Phototaxis without coupling"),
    ("orbital",     "(c)", "Phototactic orbital"),
    ("no_stimulus", "(d)", "Cohesion without a stimulus"),
]

ENVIRONMENTS = {
    "free":      dict(n=1, arena="Obstacle-free arena",  obstacles="no static obstacles"),
    "sparse":    dict(n=2, arena="Sparse arena",         obstacles="2 static obstacles"),
    "cluttered": dict(n=3, arena="Cluttered arena",      obstacles="12 static obstacles"),
}


def _fig(w_px, h_px, dpi=100):
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"]})
    return plt.figure(figsize=(w_px / dpi, h_px / dpi), facecolor="white")


def title_card(path, env, show_s, duration_s, speedup):
    meta = ENVIRONMENTS[env]
    fig = _fig(FRAME_W, FRAME_H)

    def line(y, s, fs, weight="normal", color="#111111", style="normal"):
        fig.text(0.5, y, s, ha="center", va="center", fontsize=fs,
                 fontweight=weight, color=color, style=style)

    line(0.93, f"Supplementary Video {meta['n']}", 8, "bold", "#555555")
    line(0.795, meta["arena"], 15, "bold")
    line(0.675, f"50 robots, 0.8 x 0.8 m, {meta['obstacles']}", 8.5, color="#333333")
    line(0.585, "All panels run the identical LA Slack Dt-CBF controller;", 7.5,
         color="#333333")
    line(0.505, "only the task objective differs.", 7.5, color="#333333")

    y = 0.375
    for letter, name in [(p[1], p[2]) for p in PANELS]:
        fig.text(0.30, y, letter, ha="right", va="center", fontsize=7.5, fontweight="bold")
        fig.text(0.325, y, name, ha="left", va="center", fontsize=7.5)
        y -= 0.072

    line(0.055, f"First {show_s:g} s of a {duration_s:g} s run, played at "
                f"{speedup:g}x real time", 7, color="#666666", style="italic")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def label_band(path):
    """The (a)-(d) strip sitting directly above the four panels."""
    fig = _fig(FRAME_W, LABEL_H)
    for i, (_, letter, _) in enumerate(PANELS):
        fig.text((i + 0.5) / 4, 0.42, letter, ha="center", va="center",
                 fontsize=9, fontweight="bold", color="#111111")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def legend_band(path, show_obstacles, speed_label):
    fig = _fig(FRAME_W, LEGEND_H)
    handles = video_legend_handles(show_obstacles, show_light=True,
                                   dash=(1.6, 1.2), lw=1.6)
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.60), ncol=3,
               frameon=False, fontsize=6.5, handlelength=2.6, handletextpad=0.5,
               columnspacing=1.4, labelspacing=0.7)
    fig.text(0.5, 0.10, speed_label, ha="center", va="center", fontsize=6,
             color="#666666", style="italic")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def hires_still(ff, clips, path, at_s):
    """A full-resolution version of the same row, for the required video still."""
    tiles = []
    for c in clips:
        tmp = path + f".{len(tiles)}.png"
        subprocess.run([ff, "-v", "error", "-y", "-ss", str(at_s), "-i", c,
                        "-vframes", "1", tmp], check=True)
        tiles.append(Image.open(tmp).convert("RGB")
                     .crop((CROP_X, CROP_Y, CROP_X + CROP_SIDE, CROP_Y + CROP_SIDE)))
        os.remove(tmp)
    band = 90
    out = Image.new("RGB", (CROP_SIDE * 4, CROP_SIDE + band), "white")
    for i, t in enumerate(tiles):
        out.paste(t, (i * CROP_SIDE, band))
    lab = path + ".lab.png"
    fig = _fig(CROP_SIDE * 4, band, dpi=100)
    for i, (_, letter, name) in enumerate(PANELS):
        fig.text((i + 0.5) / 4, 0.42, f"{letter} {name}", ha="center", va="center",
                 fontsize=34, fontweight="bold", color="#111111")
    fig.savefig(lab, dpi=100, facecolor="white")
    plt.close(fig)
    out.paste(Image.open(lab).convert("RGB").resize((CROP_SIDE * 4, band)), (0, 0))
    os.remove(lab)
    out.save(path)
    out.save(path[:-4] + ".pdf", "PDF", resolution=150.0)
    return out.size


def build(ff, env, outdir, duration_s, show_s, speedup, card_s, bitrate_k):
    meta = ENVIRONMENTS[env]
    clips = [os.path.join(VIDEO_DIR,
                          f"demo_{b}_{env}_N50_{duration_s:g}s_{speedup:g}x.mp4")
             for b, _, _ in PANELS]
    missing = [c for c in clips if not os.path.isfile(c) or os.path.getsize(c) == 0]
    if missing:
        for m in missing:
            print(f"  {env}: missing {os.path.basename(m)}")
        return None

    play_s = show_s / speedup
    stem = f"supp_video_{meta['n']}_{env}_row"
    out = os.path.join(outdir, stem + ".mp4")
    card = os.path.join(outdir, f"_card_{env}.png")
    labs = os.path.join(outdir, f"_lab_{env}.png")
    legd = os.path.join(outdir, f"_leg_{env}.png")
    title_card(card, env, show_s, duration_s, speedup)
    label_band(labs)
    legend_band(legd, env != "free",
                f"{show_s:g} s of simulation at {speedup:g}x real time")

    cmd = [ff, "-v", "error", "-y",
           "-loop", "1", "-t", str(card_s), "-framerate", "25", "-i", card]
    for c in clips:
        cmd += ["-t", str(play_s), "-i", c]
    cmd += ["-loop", "1", "-framerate", "25", "-i", labs,
            "-loop", "1", "-framerate", "25", "-i", legd]

    crop = f"crop={CROP_SIDE}:{CROP_SIDE}:{CROP_X}:{CROP_Y},scale={PANEL}:{PANEL}:flags=lanczos"
    fc = "".join(f"[{i+1}:v]{crop}[p{i}];" for i in range(4))
    fc += "[p0][p1][p2][p3]hstack=inputs=4[row];"
    fc += f"color=white:{FRAME_W}x{FRAME_H}:d={play_s}:r=25[bg];"
    fc += f"[bg][row]overlay=0:{LABEL_H}:shortest=1[b1];"
    fc += "[b1][5:v]overlay=0:0:shortest=1[b2];"
    fc += f"[b2][6:v]overlay=0:{LABEL_H + PANEL}:shortest=1,format=yuv420p[body];"
    fc += "[0:v]format=yuv420p[card];[card][body]concat=n=2:v=1:a=0[out]"

    cmd += ["-filter_complex", fc, "-map", "[out]", "-r", "25",
            "-c:v", "libx264", "-profile:v", "high", "-level", "3.1", "-preset", "slow",
            "-b:v", f"{bitrate_k}k", "-maxrate", f"{int(bitrate_k*1.2)}k",
            "-bufsize", f"{bitrate_k*2}k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    for f in (card, labs, legd):
        os.remove(f)

    still = os.path.join(outdir, stem + "_still.png")
    sz = hires_still(ff, clips, still, play_s * 0.5)

    r = subprocess.run([ff, "-i", out], capture_output=True, text=True).stderr
    dur = 0.0
    for ln in r.splitlines():
        if "Duration:" in ln:
            hh, mm, ss = ln.split("Duration:")[1].split(",")[0].strip().split(":")
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
    mb = os.path.getsize(out) / 1e6
    print(f"  {env:10s} {stem}.mp4  {FRAME_W}x{FRAME_H}  {dur:5.1f}s  "
          f"{mb:5.2f} MB  {mb*8000/max(dur,1e-9):5.0f} kbps  still {sz[0]}x{sz[1]}")
    return dict(env=env, n=meta["n"], stem=stem, file=os.path.basename(out),
                arena=meta["arena"], obstacles=meta["obstacles"], duration_s=dur,
                size_mb=mb, kbps=mb * 8000 / max(dur, 1e-9), still=sz)


def write_manifest(infos, outdir, duration_s, show_s, speedup):
    lines = ["# Supplementary videos", "",
             "Three videos, one per obstacle environment. Each shows the four collective "
             "behaviors side by side in a single row, all running the identical deployed "
             "controller (LA Slack Dt-CBF) and differing only in the task objective. Panel "
             "letters (a)-(d) match the panels of Fig. 5 in the manuscript.", "",
             f"Frame {FRAME_W}x{FRAME_H}, H.264/MP4, 25 fps, ~750 kbps. Each opens with a "
             f"title frame, then shows the first {show_s:g} s of a {duration_s:g} s "
             f"simulation at {speedup:g}x real time, with the legend along the bottom.", ""]
    for i in infos:
        lines += [f"## Supplementary Video {i['n']} - {i['file']}", "",
                  f"- Still: `{i['stem']}_still.png`, `{i['stem']}_still.pdf` "
                  f"({i['still'][0]}x{i['still'][1]})",
                  f"- {FRAME_W}x{FRAME_H}, 25 fps, {i['duration_s']:.1f} s, "
                  f"{i['size_mb']:.2f} MB, {i['kbps']:.0f} kbps, H.264 in MP4", "",
                  "Caption:", "",
                  f"Collective behavior of 50 robots in a 0.8 x 0.8 m arena with "
                  f"{i['obstacles']}, under Distributed Predictive Flocking. All four "
                  f"panels run the identical deployed controller (LA Slack Dt-CBF) and "
                  f"differ only in the task objective: (a) phototaxis, (b) phototaxis "
                  f"without coupling, (c) phototactic orbital, (d) cohesion without a "
                  f"stimulus. The clip shows the first {show_s:g} s of a {duration_s:g} s "
                  f"run, played at {speedup:g}x real time.", ""]
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
    ap.add_argument("--envs", nargs="*", default=list(ENVIRONMENTS))
    ap.add_argument("--outdir", default=os.path.join(VIDEO_DIR, "montage_row_180s"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    infos = [build(ff, e, a.outdir, a.duration, a.show_seconds, a.speed,
                   a.card_seconds, a.bitrate_kbps) for e in a.envs]
    if not all(infos):
        sys.exit(1)
    write_manifest(infos, a.outdir, a.duration, a.show_seconds, a.speed)


if __name__ == "__main__":
    main()
