"""Renders demonstration videos of each behavior."""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import viz_engine as V
H = V.H
SwarmBehavior = V.SwarmBehavior

V.plt.switch_backend("Agg")
V.plt.show = lambda *a, **k: None

BEHAVIORS = {
    "phototaxis":  SwarmBehavior.PHOTOTAXIS,
    "poc":         SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING,
    "no_reynolds": SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS,
    "no_stimulus": SwarmBehavior.NO_LIGHT,
    "orbital":     SwarmBehavior.PHOTOTAXIS_ORBITAL,
}

ENVIRONMENTS = {
    "free":      (0,  "Obstacle-free"),
    "sparse":    (2,  "Sparse (2 obstacles)"),
    "cluttered": (12, "Cluttered (12 obstacles)"),
}

PANEL_LABELS = {
    "phototaxis":  "(a) Phototaxis",
    "no_reynolds": "(b) Phototaxis without coupling",
    "orbital":     "(c) Phototactic orbital",
    "no_stimulus": "(d) Cohesion without a stimulus",
}

OBSTACLE_SEED = 0


def configure(behavior_key, env_key, duration_s, fps, dpi, n_robots=50, speed=1.0):
    """Set viz_engine module globals for one (behavior, env) clip.

    Renders `duration_s` seconds of robot motion. At speed=1 (real time) the
    controller advances at 10 Hz (dt=0.1) and INTERP = fps/10 interpolated frames
    are drawn per control step, so the saved video is exactly `duration_s` long.
    speed>1 draws fewer frames per control step, so the same robot motion plays
    back that many times faster (useful for long runs); the burned-in clock still
    shows true simulation time. Max speedup is fps*DT (one frame per control
    step); higher requests are capped there.
    """
    count, env_label = ENVIRONMENTS[env_key]

    steps   = max(1, round(duration_s / V.DT))
    interp  = max(1, round(fps * V.DT / max(speed, 1e-9)))
    eff_fps = fps if speed != 1.0 else int(round(interp / V.DT))

    V.N_ROBOTS      = int(n_robots)
    V.BEHAVIOR_MODE = BEHAVIORS[behavior_key]
    V.MAX_STEPS     = steps
    V.INTERP        = interp
    V.DRAW_INTERVAL = max(1, round(1000 / eff_fps))
    V.SAVE_DPI      = dpi
    V.SAVE_GIF      = False
    V.SAVE_VIDEO    = True

    if count <= 0:
        V.ENABLE_OBSTACLES = False
        V.OBSTACLE_DEFS    = []
    else:
        V.ENABLE_OBSTACLES = True
        V.OBSTACLE_DEFS    = H.build_obstacle_defs(count, seed=OBSTACLE_SEED)

    save_fps = int(1000 / V.DRAW_INTERVAL)
    speedup  = V.DT * save_fps / interp

    V.EXTRA_TITLE = f" - {env_label}"
    V.PANEL_TITLE = PANEL_LABELS.get(behavior_key)
    V.SHOW_VIDEO_LEGEND = V.PANEL_TITLE is None
    V.SPEED_LABEL = None if V.PANEL_TITLE is not None or abs(speedup - 1.0) < 0.05 \
        else f"{speedup:.0f}x speed"
    tag = "realtime" if abs(speedup - 1.0) < 0.05 else f"{speedup:.0f}x"
    V.VIDEO_NAME  = (f"demo_{behavior_key}_{env_key}_N{V.N_ROBOTS}"
                     f"_{int(round(duration_s))}s_{tag}.mp4")

    return dict(steps=steps, interp=interp, save_fps=save_fps,
                frames=steps * interp, speedup=speedup)


def render(behavior_key, env_key, duration_s, fps, dpi, n_robots=50, speed=1.0):
    info    = configure(behavior_key, env_key, duration_s, fps, dpi, n_robots, speed)
    vid_len = info["frames"] / info["save_fps"]
    print(f"\n=== {behavior_key} / {env_key}  "
          f"(N={V.N_ROBOTS}, {info['steps']} steps x{info['interp']} = "
          f"{info['frames']} frames, {info['save_fps']} fps, "
          f"{vid_len:.1f}s, {info['speedup']:.2f}x real-time, {dpi*10}px) ===")
    V.PhototaxisSim(show_right_panel=False).run()
    V.plt.close("all")
    print(f"  -> {os.path.join(V.VIDEO_DIR, V.VIDEO_NAME)}")


def main():
    ap = argparse.ArgumentParser(description="Generate variant-08 real-time demo MP4s.")
    ap.add_argument("--behavior", choices=list(BEHAVIORS), action="append",
                    help="restrict to behavior(s); repeatable. Default: all four.")
    ap.add_argument("--env", choices=list(ENVIRONMENTS), action="append",
                    help="restrict to environment(s); repeatable. Default: all three.")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds of real robot motion, real-time (default 60 -> "
                         "600 control steps at dt=0.1). Use e.g. 4 for a smoke test.")
    ap.add_argument("--fps", type=int, default=30,
                    help="video frame rate (default 30 -> 3 interpolated frames/step).")
    ap.add_argument("--dpi", type=int, default=120,
                    help="output dpi; 10in figure -> dpi*10 px square "
                         "(default 120 -> 1200px, slide-sized).")
    ap.add_argument("--n", type=int, default=50,
                    help="swarm size N_b (default 50).")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speedup (default 1 = real time). Capped at "
                         "fps*0.1 (one frame per control step), i.e. 3x at 30 fps.")
    args = ap.parse_args()

    behaviors = args.behavior or list(BEHAVIORS)
    envs      = args.env or list(ENVIRONMENTS)
    jobs      = [(b, e) for b in behaviors for e in envs]

    print(f"Rendering {len(jobs)} clip(s): behaviors={behaviors} envs={envs} "
          f"duration={args.duration}s fps={args.fps} dpi={args.dpi}")
    print(f"Output dir: {V.VIDEO_DIR}")
    for b, e in jobs:
        render(b, e, args.duration, args.fps, args.dpi, args.n, args.speed)
    print(f"\nDone. {len(jobs)} clip(s) in {V.VIDEO_DIR}")


if __name__ == "__main__":
    main()
