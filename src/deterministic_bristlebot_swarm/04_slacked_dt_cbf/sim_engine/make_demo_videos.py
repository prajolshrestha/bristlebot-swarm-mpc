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
    "no_light":    SwarmBehavior.NO_LIGHT,
    "orbital":     SwarmBehavior.PHOTOTAXIS_ORBITAL,
}

ENVIRONMENTS = {
    "free":      (0,  "Obstacle-free"),
    "sparse":    (2,  "Sparse (2 obstacles)"),
    "cluttered": (12, "Cluttered (12 obstacles)"),
}

OBSTACLE_SEED = 0


def configure(behavior_key, env_key, duration_s, fps, dpi):
    """Set viz_engine module globals for one real-time (behavior, env) clip.

    Renders `duration_s` seconds of robot motion at real-time speed: the
    controller advances at 10 Hz (dt=0.1) and INTERP = fps/10 interpolated frames
    are drawn per control step, so the saved video is exactly `duration_s` long
    at the requested fps.
    """
    count, env_label = ENVIRONMENTS[env_key]

    steps   = max(1, round(duration_s / V.DT))
    interp  = max(1, round(fps * V.DT))
    eff_fps = int(round(interp / V.DT))

    V.N_ROBOTS      = 50
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
    V.SPEED_LABEL = None
    V.VIDEO_NAME  = f"demo_{behavior_key}_{env_key}_N50_realtime.mp4"

    return dict(steps=steps, interp=interp, save_fps=save_fps,
                frames=steps * interp, speedup=speedup)


def render(behavior_key, env_key, duration_s, fps, dpi):
    info    = configure(behavior_key, env_key, duration_s, fps, dpi)
    vid_len = info["frames"] / info["save_fps"]
    print(f"\n=== {behavior_key} / {env_key}  "
          f"(N={V.N_ROBOTS}, {info['steps']} steps x{info['interp']} = "
          f"{info['frames']} frames, {info['save_fps']} fps, "
          f"{vid_len:.1f}s, {info['speedup']:.2f}x real-time, {dpi*10}px) ===")
    V.PhototaxisSim(show_right_panel=False).run()
    V.plt.close("all")
    print(f"  -> {os.path.join(V.VIDEO_DIR, V.VIDEO_NAME)}")


def main():
    ap = argparse.ArgumentParser(description="Generate variant-07 real-time demo MP4s.")
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
    args = ap.parse_args()

    behaviors = args.behavior or list(BEHAVIORS)
    envs      = args.env or list(ENVIRONMENTS)
    jobs      = [(b, e) for b in behaviors for e in envs]

    print(f"Rendering {len(jobs)} clip(s): behaviors={behaviors} envs={envs} "
          f"duration={args.duration}s fps={args.fps} dpi={args.dpi}")
    print(f"Output dir: {V.VIDEO_DIR}")
    for b, e in jobs:
        render(b, e, args.duration, args.fps, args.dpi)
    print(f"\nDone. {len(jobs)} clip(s) in {V.VIDEO_DIR}")


if __name__ == "__main__":
    main()
