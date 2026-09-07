"""Lay one behavior's three obstacle environments side by side in a single row."""

import argparse
import os
import subprocess

import imageio_ffmpeg

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
VIDEO_DIR = os.path.join(REPO, "src", "experiments", "simulation", "results", "videos")
ENVS = ["free", "sparse", "cluttered"]


def build(behavior, clip_dir, prefix, n_robots, duration_s, speedup, out_path,
          speed_mult, width, crf, fps):
    """hstack the three arenas, then speed up by speed_mult and scale for delivery."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    clips = [os.path.join(clip_dir,
                          f"{prefix}_{behavior}_{e}_N{n_robots}_{duration_s:g}s_{speedup:g}x.mp4")
             for e in ENVS]
    missing = [c for c in clips if not os.path.isfile(c)]
    if missing:
        for m in missing:
            print(f"  missing {os.path.basename(m)}")
        return None

    from PIL import Image
    probe = out_path + ".probe.png"
    subprocess.run([ff, "-v", "error", "-y", "-i", clips[0], "-vframes", "1", probe],
                   check=True)
    w, h = Image.open(probe).size
    os.remove(probe)

    out_w = width - (width % 2)
    out_h = int(round(h * out_w / (w * 3)))
    out_h -= out_h % 2

    cmd = [ff, "-v", "error", "-y"]
    for c in clips:
        cmd += ["-i", c]
    # setpts drops the presentation interval, so the clip plays speed_mult faster; the
    # source frames are kept, which is what keeps motion smooth rather than stuttering.
    cmd += ["-filter_complex",
            "[0:v][1:v][2:v]hstack=inputs=3[row];"
            f"[row]setpts=PTS/{speed_mult:g},"
            f"scale={out_w}:{out_h}:flags=lanczos,format=yuv420p[out]",
            "-map", "[out]", "-r", str(fps),
            "-c:v", "libx264", "-profile:v", "high", "-level", "4.0", "-preset", "slow",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)

    poster = os.path.splitext(out_path)[0] + ".jpg"
    r = subprocess.run([ff, "-i", out_path], capture_output=True, text=True).stderr
    dur = 0.0
    for ln in r.splitlines():
        if "Duration:" in ln:
            hh, mm, ss = ln.split("Duration:")[1].split(",")[0].strip().split(":")
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
    subprocess.run([ff, "-v", "error", "-y", "-ss", str(dur * 0.75), "-i", out_path,
                    "-vframes", "1", "-q:v", "3", poster], check=True)
    mb = os.path.getsize(out_path) / 1e6
    print(f"  {os.path.basename(out_path)}  {out_w}x{out_h}  {dur:.0f}s  "
          f"{mb:.1f} MB  {speedup*speed_mult:.0f}x real time")
    print(f"  {os.path.basename(poster)}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default="orbital")
    ap.add_argument("--clip-dir",
                    default=os.path.join(VIDEO_DIR, "spawn", "original"))
    ap.add_argument("--prefix", default="spawn")
    ap.add_argument("--n-robots", type=int, default=137)
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--speed", type=float, default=5, help="speedup already in the clips")
    ap.add_argument("--speed-mult", type=float, default=3,
                    help="further speed-up applied here; 5x clips x 3 = 15x")
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ok = build(a.behavior, a.clip_dir, a.prefix, a.n_robots, a.duration, a.speed,
               a.out, a.speed_mult, a.width, a.crf, a.fps)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
