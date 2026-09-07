"""Compose the supplementary videos: a title card plus a 2x2 behavior grid per obstacle arena."""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "deterministic_bristlebot_swarm", "05_slacked_dt_cbf_with_lookahead", "sim_engine"))
from video_legend import save_legend_strip

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
VIDEO_DIR = os.path.join(REPO, "src", "experiments", "simulation", "results", "videos")
DEFAULT_CLIPS = os.path.join(VIDEO_DIR, "random_initial_position", "original")

PANELS = ["phototaxis", "no_reynolds", "orbital", "no_stimulus"]
LETTERS = ["(a)", "(b)", "(c)", "(d)"]


def _serif_bold():
    """Path to a bold serif TTF, for the drawtext panel letters."""
    import matplotlib
    return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts",
                        "ttf", "DejaVuSerif-Bold.ttf")


def letter_overlay(path, w, h):
    """Transparent PNG carrying (a)-(d) at each panel's title height.

    The spawn clips were rendered without panel letters and their recordings have
    been deleted, so the letters are composited here. drawtext is not available:
    the imageio_ffmpeg static build omits libfreetype, so this is drawn with PIL
    and layered in with the overlay filter instead.
    """
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(_serif_bold(), 46)
    # Bottom-left of each panel, not top-left: the two longer behavior names run
    # nearly the full panel width, so a top-left letter collides with the title.
    # This corner measures pure white in every panel.
    for k, lab in enumerate(LETTERS):
        d.text(((k % 2) * w + 34, (k // 2) * h + h - 108), lab, font=font,
               fill=(17, 17, 17, 255))
    im.save(path)

ENVIRONMENTS = {
    "free":      dict(n=1, obstacles="no static obstacles", arena="Obstacle-free arena",
                      ref="Arena and light source match Fig. 5 of the paper."),
    "sparse":    dict(n=2, obstacles="2 static obstacles", arena="Sparse arena",
                      ref="Obstacle layout is identical to Fig. A.1 of the paper."),
    "cluttered": dict(n=3, obstacles="12 static obstacles", arena="Cluttered arena",
                      ref="Obstacle layout is identical to Fig. A.2 of the paper."),
}

TITLE = ("Distributed Predictive Flocking for a Swarm of Miniature\n"
         "Vibration-Driven Robots in a Confined Arena")
AUTHORS = "Shrestha, Mohapatra, Novkoski, Oelhaf, Vandewalle, Smith, Maier"


def title_card(env, path, width_px, height_px, duration_s, speedup, dpi=100,
               n_robots=50, entry=""):
    meta = ENVIRONMENTS[env]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), facecolor="white")

    def line(y, s, fs, weight="normal", color="#111111"):
        fig.text(0.5, y, s, ha="center", va="center", fontsize=fs,
                 fontweight=weight, color=color)

    line(0.86, f"Supplementary Video {meta['n']}", 34, "bold", "#444444")
    head = meta["arena"] if env == "free" else f"{meta['arena']}: {meta['obstacles']}"
    line(0.72, head, 58, "bold")

    y = 0.555
    for s_ in (f"{n_robots} robots in a 0.8 x 0.8 m arena with {meta['obstacles']}.",
               "All four panels run the identical deployed controller",
               "(LA Slack Dt-CBF); only the task objective differs."):
        line(y, s_, 26, color="#222222"); y -= 0.042

    line(0.425, "Task in each panel:", 27, "bold", "#444444")

    y = 0.370
    for lab, txt in (("(a)", "Phototaxis"), ("(b)", "Phototaxis without coupling"),
                     ("(c)", "Phototactic orbital"), ("(d)", "Cohesion without a stimulus")):
        fig.text(0.38, y, lab, ha="right", va="center", fontsize=26, fontweight="bold")
        fig.text(0.41, y, txt, ha="left", va="center", fontsize=26)
        y -= 0.050

    line(0.100, f"{duration_s:g} s of simulation shown at {speedup:g}x real time.", 26,
         color="#222222")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


CAPTIONS = {
    "free": ("Collective behavior of 50 robots in an obstacle-free 0.8 x 0.8 m arena under "
             "Distributed Predictive Flocking. All four panels run the identical deployed "
             "controller (LA Slack Dt-CBF) and differ only in the task objective: "
             "(a) phototaxis, (b) phototaxis without coupling, (c) phototactic orbital, "
             "(d) cohesion without a stimulus. {shown}"),
    "sparse": ("The same four collective behaviors of 50 robots, in a 0.8 x 0.8 m arena "
               "containing 2 static obstacles. All four panels run the identical deployed "
               "controller (LA Slack Dt-CBF) and differ only in the task objective: "
               "(a) phototaxis, (b) phototaxis without coupling, (c) phototactic orbital, "
               "(d) cohesion without a stimulus. {shown}"),
    "cluttered": ("The same four collective behaviors of 50 robots, in a 0.8 x 0.8 m arena "
                  "containing 12 static obstacles. All four panels run the identical deployed "
                  "controller (LA Slack Dt-CBF) and differ only in the task objective: "
                  "(a) phototaxis, (b) phototaxis without coupling, (c) phototactic orbital, "
                  "(d) cohesion without a stimulus. {shown}"),
}


def probe(ff, path):
    """Container, codec, size, frame rate and duration of a finished file."""
    r = subprocess.run([ff, "-i", path], capture_output=True, text=True).stderr
    out = {"size_mb": os.path.getsize(path) / 1e6}
    for ln in r.splitlines():
        if "Duration:" in ln:
            hh, mm, ss = ln.split("Duration:")[1].split(",")[0].strip().split(":")
            out["duration_s"] = int(hh) * 3600 + int(mm) * 60 + float(ss)
        if "Stream" in ln and "Video:" in ln:
            out["codec"] = ln.split("Video:")[1].split(",")[0].strip().split()[0]
            for tok in ln.split(","):
                tok = tok.strip()
                if tok.endswith(" fps"):
                    out["fps"] = float(tok[:-4])
                if "x" in tok and tok.split()[0].replace("x", "").isdigit():
                    w, h = tok.split()[0].split("x")
                    out["width"], out["height"] = int(w), int(h)
    out["bitrate_mbps"] = out["size_mb"] * 8 / max(out.get("duration_s", 1), 1e-9)
    return out


def compose(env, duration_s, speedup, fps, card_s, outdir, out_width, crf,
            show_s=None, clip_dir=DEFAULT_CLIPS, prefix="demo", n_robots=50,
            entry=""):
    clips = [os.path.join(clip_dir,
                          f"{prefix}_{b}_{env}_N{n_robots}_{duration_s:g}s_{speedup:g}x.mp4")
             for b in PANELS]
    missing = [c for c in clips if not os.path.isfile(c) or os.path.getsize(c) == 0]
    if missing:
        print(f"  {env}: missing {len(missing)} clip(s)")
        for m in missing:
            print(f"    {os.path.basename(m)}")
        return None

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = os.path.join(outdir, f"_probe_{env}.png")
    subprocess.run([ff, "-v", "error", "-y", "-i", clips[0], "-vframes", "1", tmp], check=True)
    from PIL import Image
    w, h = Image.open(tmp).size
    os.remove(tmp)
    grid_w, grid_h = w * 2, h * 2

    shown = show_s if show_s else duration_s
    play_s = shown / speedup

    strip_h = 300
    full_h = grid_h + strip_h
    strip = os.path.join(outdir, f"_strip_{env}.png")
    save_legend_strip(strip, grid_w, strip_h, show_obstacles=(env != "free"),
                      speed_label=f"{shown:g} s of simulation at {speedup:g}x real time")

    card = os.path.join(outdir, f"_card_{env}.png")
    title_card(env, card, grid_w, full_h, shown, speedup,
               n_robots=n_robots, entry=entry)

    out_w = out_width - (out_width % 2)
    out_h = int(round(full_h * out_w / grid_w))
    out_h -= out_h % 2

    out = os.path.join(outdir, f"supp_video_{ENVIRONMENTS[env]['n']}_{env}.mp4")
    cmd = [ff, "-v", "error", "-y",
           "-loop", "1", "-t", str(card_s), "-framerate", str(fps), "-i", card]
    for c in clips:
        if show_s:
            cmd += ["-t", str(play_s)]
        cmd += ["-i", c]
    letters = os.path.join(outdir, f"_letters_{env}.png")
    letter_overlay(letters, w, h)
    cmd += ["-loop", "1", "-framerate", str(fps), "-i", strip]
    cmd += ["-loop", "1", "-framerate", str(fps), "-i", letters]
    cmd += ["-filter_complex",
            "[1:v][2:v]hstack[top];[3:v][4:v]hstack[bot];[top][bot]vstack[g0];"
            "[g0][6:v]overlay=0:0:shortest=1[g];"
            f"[g]pad={grid_w}:{full_h}:0:0:white[gp];"
            "[gp][5:v]overlay=0:%d:shortest=1[grid];" % grid_h +
            f"[0:v]scale={grid_w}:{full_h}[card];"
            "[card][grid]concat=n=2:v=1:a=0[cat];"
            f"[cat]scale={out_w}:{out_h}:flags=lanczos,format=yuv420p[out]",
            "-map", "[out]", "-r", str(fps), "-c:v", "libx264", "-crf", str(crf),
            "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    os.remove(card); os.remove(strip); os.remove(letters)

    still = out[:-4] + "_still.png"
    subprocess.run([ff, "-v", "error", "-y", "-ss", str(card_s + play_s * 0.35),
                    "-i", out, "-vframes", "1", still], check=True)

    info = probe(ff, out)
    info.update(env=env, file=os.path.basename(out), still=os.path.basename(still),
                caption=CAPTIONS[env].replace("50 robots", f"{n_robots} robots").format(shown=(
                    f"The clip shows the first {shown:g} s of a {duration_s:g} s run, "
                    f"played at {speedup:g}x real time." if show_s else
                    f"{duration_s:g} s of simulation, played at {speedup:g}x real time.")))
    print(f"  {env}: {info['file']}  {info.get('width')}x{info.get('height')}  "
          f"{info.get('fps')} fps  {info.get('duration_s', 0):.1f} s  "
          f"{info['size_mb']:.1f} MB  {info['bitrate_mbps']:.1f} Mbps")
    return info


def write_manifest(infos, outdir):
    """Per-file captions and measured specs, for pasting into the submission form."""
    lines = ["# Supplementary videos", "",
             "Generated by `behaviors_src/make_video_montages.py`. Each montage shows the",
             "four collective behaviors side by side in one obstacle environment, with the",
             "panel layout and obstacle positions matching the corresponding paper figure.",
             ""]
    for i in infos:
        lines += [f"## {i['file']}", "",
                  f"- Still: `{i['still']}`",
                  f"- {i.get('width')}x{i.get('height')}, {i.get('fps')} fps, "
                  f"{i.get('duration_s', 0):.1f} s, {i['size_mb']:.1f} MB, "
                  f"{i.get('codec')} in MP4",
                  "", "Caption:", "", i["caption"], ""]
    total = sum(i["size_mb"] for i in infos)
    lines += ["## Package", "",
              f"- {len(infos)} video(s), {total:.1f} MB total",
              "- Limits: 150 MB preferred per file, 5 min max, 15 fps min, 260 kbps min", ""]
    p = os.path.join(outdir, "SUPPLEMENTARY.md")
    open(p, "w").write("\n".join(lines))
    print(f"  manifest: {os.path.relpath(p)}  ({total:.1f} MB total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--speed", type=float, default=5)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--card-seconds", type=float, default=2)
    ap.add_argument("--envs", nargs="*", default=list(ENVIRONMENTS))
    ap.add_argument("--out-width", type=int, default=2400)
    ap.add_argument("--crf", type=int, default=34)
    ap.add_argument("--show-seconds", type=float, default=None,
                    help="window the full clips to this many simulated seconds")
    ap.add_argument("--clip-dir", default=DEFAULT_CLIPS)
    ap.add_argument("--prefix", default="demo")
    ap.add_argument("--n-robots", type=int, default=50)
    ap.add_argument("--entry", default="",
                    help="extra title-card line describing how robots enter")
    ap.add_argument("--outdir", default=os.path.join(VIDEO_DIR, "montage"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    made = [compose(e, a.duration, a.speed, a.fps, a.card_seconds, a.outdir,
                    a.out_width, a.crf, a.show_seconds, a.clip_dir,
                    a.prefix, a.n_robots, a.entry) for e in a.envs]
    if not all(made):
        sys.exit(1)
    write_manifest(made, a.outdir)


if __name__ == "__main__":
    main()
