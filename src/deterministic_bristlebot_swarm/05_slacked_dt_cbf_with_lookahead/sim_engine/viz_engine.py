"""The headless simulation with live animation on top."""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

_VIZ_BACKEND = matplotlib.get_backend()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import headless_sim_engine as H
from headless_sim_engine import HeadlessSim

try:
    plt.switch_backend(_VIZ_BACKEND)
except Exception:
    pass

from expert_src.controller import (
    D_SAFE, D_SAFE_OBS, MAX_CONSENSUS_ITER, MAX_NEIGHBOURS,
    W_SOFT, W_SOFT_OBS, W_SOFT_LIN, W_SOFT_OBS_LIN,
    Q_COHESION_DEFAULT, Q_ALIGN_DEFAULT,
)

SwarmBehavior    = H.SwarmBehavior
SourceMotionMode = H.SourceMotionMode
LightField       = H.LightField


DT          = H.DT
MPC_HORIZON = H.MPC_HORIZON
ARENA_HALF  = H.ARENA_HALF
WALL_HALF   = H.WALL_HALF
COMM_RADIUS = H.COMM_RADIUS
ROBOT_A     = H.ROBOT_A
ROBOT_B     = H.ROBOT_B
V_TARGET    = H.V_TARGET

N_ROBOTS    = 50

BEHAVIOR_MODE  = SwarmBehavior.PHOTOTAXIS

ORBIT_CENTER   = np.array([0.0, 0.0])
ORBIT_RADIUS   = 0.25

N_SOURCES      = 1
SOURCE_SIGMA   = 0.10
SOURCE_SPEED   = 0.15
SOURCE_SEED    = 7
SOURCE_MOTION_MODE = SourceMotionMode.STATIONARY
SOURCE_ROT_RADIUS  = 0.25
SOURCE_PULSING     = False

LOOKAHEAD_DIST = 0.07
EPSILON_GRAD   = 1e-6

Q_POS          = 60.0
Q_THETA        = 80.0
Q_VEL          = 1.5
Q_OM           = 1.5
R_U            = 0.01

Q_COHESION     = Q_COHESION_DEFAULT
Q_ALIGN        = Q_ALIGN_DEFAULT

SPAWN_SEED     = 42

OBSTACLE_RADIUS = 0.03

OBSTACLE_DEFS = [
    {"type": "static",  "pos": [ 0.15,  0.10], "active": True},
    {"type": "static",  "pos": [-0.15, -0.10], "active": True},
]
ENABLE_OBSTACLES = True

ENABLE_COLLISION       = True
COLLISION_STEER_DECAY  = 0.85
COLLISION_FLASH_FRAMES = 6


MAX_STEPS      = 600
HEATMAP_RES    = 20
DRAW_INTERVAL  = 10
SAVE_GIF       = False
GIF_NAME       = f"reynolds_swarm_{BEHAVIOR_MODE.value}_tcoupled_sync_all_soft_wsoft_{W_SOFT:g}.gif"
SAVE_VIDEO     = False
VIDEO_NAME     = f"reynolds_swarm_{BEHAVIOR_MODE.value}_tcoupled_sync_all_soft_wsoft_{W_SOFT:g}.mp4"

SPEED_LABEL    = None
EXTRA_TITLE    = ""
PANEL_TITLE    = None
SHOW_VIDEO_LEGEND = True


BEHAVIOR_DISPLAY = {
    "phototaxis": "Phototaxis",
    "phototaxis_orbital": "Phototactic Orbital",
    "phototaxis_orbital_contracting": "Phototactic Orbital (Contracting)",
    "phototaxis_no_reynolds": "Phototaxis Without Coupling",
    "no_light": "Cohesion Without a Stimulus",
}


from video_legend import (ROBOT_COLORS, VIOL_COLOR,
                          video_legend_handles as _video_legend_handles)


def behavior_title(mode):
    return BEHAVIOR_DISPLAY.get(mode.value, mode.value.replace("_", " ").title())


SAVE_DPI       = None
INTERP         = 1


SUPP_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supplimentary_videos")
GIF_DIR   = os.path.join(SUPP_DIR, "gif")
_REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
VIDEO_DIR = os.path.join(_REPO_ROOT, "src", "experiments", "simulation", "results",
                         "videos", "random_initial_position", "original")
SHOW_RIGHT_PANEL = False


BG_COLOR        = "white"
PANEL_COLOR     = "white"
GRID_COLOR      = "#dddddd"
TEXT_COLOR      = "#111111"
LABEL_COLOR     = "#333333"
SPINE_COLOR     = "#aaaaaa"
TITLE_COLOR     = "#111111"
BOARD_TITLE_COL = "#b45309"
MINDIST_COLOR   = "#cc2222"
SAFE_DASH_OK    = "#1565C0"

class PhototaxisSim(HeadlessSim):

    def __init__(self,
                 q_pos=Q_POS, q_theta=Q_THETA, q_vel=Q_VEL,
                 q_om=Q_OM,   r_u=R_U,
                 w_soft=W_SOFT, q_coh=Q_COHESION, q_ali=Q_ALIGN,
                 w_soft_obs=W_SOFT_OBS,
                 w_soft_lin=W_SOFT_LIN, w_soft_obs_lin=W_SOFT_OBS_LIN,
                 show_right_panel=SHOW_RIGHT_PANEL):

        if BEHAVIOR_MODE == SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS:
            w_soft, q_coh, q_ali, w_soft_obs = 0.0, 0.0, 0.0, 0.0
            w_soft_lin, w_soft_obs_lin = 0.0, 0.0

        H.SPAWN_SEED       = globals().get("SPAWN_SEED", 42)
        H.OBSTACLE_DEFS    = OBSTACLE_DEFS
        H.OBSTACLE_RADIUS  = OBSTACLE_RADIUS
        H.ENABLE_OBSTACLES = ENABLE_OBSTACLES

        super().__init__(
            n_robots=N_ROBOTS,
            behavior_mode=BEHAVIOR_MODE,
            q_coh=float(q_coh), q_ali=q_ali,
            w_soft=w_soft, w_soft_obs=w_soft_obs,
            w_soft_lin=w_soft_lin, w_soft_obs_lin=w_soft_obs_lin,
            max_consensus=MAX_CONSENSUS_ITER,
            enable_obstacles=ENABLE_OBSTACLES,
            enable_collision=ENABLE_COLLISION,
            steer_decay=COLLISION_STEER_DECAY,
            flash_frames=COLLISION_FLASH_FRAMES,
        )

        self.q_pos, self.q_theta, self.q_vel, self.q_om, self.r_u = (
            q_pos, q_theta, q_vel, q_om, r_u
        )
        for ctrl in self.controllers:
            ctrl.q_pos   = q_pos
            ctrl.q_theta = q_theta
            ctrl.q_vel   = q_vel
            ctrl.q_om    = q_om
            ctrl.r_u     = r_u

        self.w_soft, self.q_coh, self.q_ali, self.w_soft_obs = (
            w_soft, float(q_coh), q_ali, w_soft_obs
        )
        self.w_soft_lin, self.w_soft_obs_lin = w_soft_lin, w_soft_obs_lin

        self.field = self._build_field()

        self.show_right_panel = show_right_panel

        self.intensity_log = [[] for _ in range(self.n_robots)]
        self.w_log         = [[] for _ in range(self.n_robots)]

    @staticmethod
    def _build_field():
        """Construct the LightField exactly as the behavior engine does."""
        if BEHAVIOR_MODE == SwarmBehavior.NO_LIGHT:
            return None
        motion_mode = SOURCE_MOTION_MODE
        if BEHAVIOR_MODE in (
            SwarmBehavior.PHOTOTAXIS,
            SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS,
        ):
            motion_mode = SourceMotionMode.ROTATION
        elif BEHAVIOR_MODE in (
            SwarmBehavior.ORBIT,
            SwarmBehavior.PHOTOTAXIS_ORBITAL,
            SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING,
        ):
            motion_mode = SourceMotionMode.STATIONARY
        return LightField(
            motion_mode=motion_mode,
            rotation_radius=SOURCE_ROT_RADIUS,
            pulsing=SOURCE_PULSING,
        )

    def step(self):
        """Advance the headless simulation, then record the viz-only logs."""
        super().step()
        for i, robot in enumerate(self.robots):
            pos = self.histories[i][-1]
            intensity = self.field.intensity(pos) if self.field is not None else 0.0
            self.intensity_log[i].append(intensity)
            self.w_log[i].append(float(robot.ang_velocity))

    def run(self):
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "figure.dpi": 110,
            "savefig.dpi": 300,
        })

        cmap_light = mcolors.LinearSegmentedColormap.from_list(
            "lamp_white",
            ["#ffffff", "#fffde7", "#fff176", "#ffb300", "#e65100", "#b71c1c"],
        )

        if self.show_right_panel:
            fig = plt.figure(figsize=(16, 12), facecolor=BG_COLOR)
            fig.suptitle(
                PANEL_TITLE if PANEL_TITLE else
                f"Distributed Predictive Flocking: {behavior_title(BEHAVIOR_MODE)}{EXTRA_TITLE}",
                color=TITLE_COLOR, fontsize=(30 if PANEL_TITLE else 13), y=0.99,
            )
            gs = fig.add_gridspec(
                4, 3, left=0.04, right=0.97, top=0.95, bottom=0.04,
                wspace=0.30, hspace=0.50,
            )
            ax_world     = fig.add_subplot(gs[:, 0:2], facecolor=PANEL_COLOR)
            ax_board     = fig.add_subplot(gs[0, 2],   facecolor=PANEL_COLOR)
            ax_intensity = fig.add_subplot(gs[1, 2],   facecolor=PANEL_COLOR)
            ax_v         = fig.add_subplot(gs[2, 2],   facecolor=PANEL_COLOR)
            ax_w         = fig.add_subplot(gs[3, 2],   facecolor=PANEL_COLOR)
            axes_to_style = [ax_world, ax_board, ax_intensity, ax_v, ax_w]
        else:
            fig = plt.figure(figsize=(10, 10), facecolor=BG_COLOR)
            fig.suptitle(
                PANEL_TITLE if PANEL_TITLE else
                f"Distributed Predictive Flocking: {behavior_title(BEHAVIOR_MODE)}{EXTRA_TITLE}",
                color=TITLE_COLOR, fontsize=(30 if PANEL_TITLE else 13), y=0.96,
            )
            ax_world     = fig.add_subplot(1, 1, 1, facecolor=PANEL_COLOR)
            ax_board     = None
            ax_intensity = None
            ax_v         = None
            ax_w         = None
            axes_to_style = [ax_world]

        if SPEED_LABEL:
            fig.text(0.99, 0.012, SPEED_LABEL, ha="right", va="bottom",
                     color=LABEL_COLOR, fontsize=11, alpha=0.85)

        for ax in axes_to_style:
            ax.tick_params(colors=LABEL_COLOR, labelsize=(20 if PANEL_TITLE else 8))
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COLOR)
            ax.xaxis.label.set_color(LABEL_COLOR)
            ax.yaxis.label.set_color(LABEL_COLOR)

        def animate(frame):
            if frame >= MAX_STEPS * INTERP:
                return []

            q, r = divmod(frame, INTERP)
            if r == 0:
                if q == 0:
                    self._interp_prev = [(rb.pos.copy(), float(rb.theta))
                                         for rb in self.robots]
                self.step()
                self._interp_cur = [(rb.pos.copy(), float(rb.theta))
                                    for rb in self.robots]
            if INTERP > 1:
                alpha = r / INTERP
                for i, rb in enumerate(self.robots):
                    p0, t0 = self._interp_prev[i]
                    p1, t1 = self._interp_cur[i]
                    rb.pos = p0 + (p1 - p0) * alpha
                    dth = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
                    rb.theta = t0 + dth * alpha

            ax_world.clear()
            ax_world.set_facecolor(PANEL_COLOR)
            ax_world.set_aspect("equal")
            ax_world.set_xlabel("$x$ [m]", color=LABEL_COLOR,
                                 fontsize=(22 if PANEL_TITLE else 9))
            ax_world.set_ylabel("$y$ [m]", color=LABEL_COLOR,
                                 fontsize=(22 if PANEL_TITLE else 9))
            ax_world.tick_params(colors=LABEL_COLOR, labelsize=(20 if PANEL_TITLE else 8))
            for spine in ax_world.spines.values():
                spine.set_edgecolor(SPINE_COLOR)

            if self.field is not None:
                grid, extent = self.field.heatmap(HEATMAP_RES)
                ax_world.imshow(
                    grid, origin="lower", extent=extent,
                    cmap=cmap_light, vmin=0, vmax=1.0 * N_SOURCES,
                    interpolation="bilinear", zorder=1, alpha=0.85,
                )

                for src in self.field.sources:
                    for r_scale, a in [(0.06, 0.20), (0.03, 0.40), (0.01, 0.80)]:
                        ax_world.add_patch(plt.Circle(
                            src.pos, r_scale * src.amplitude,
                            color="#b45309", alpha=a, fill=False,
                            linewidth=1.5, zorder=2,
                        ))
                    ax_world.plot(*src.pos, "o", color="#b45309",
                                  markersize=6, alpha=0.95, zorder=3)

            ax_world.add_patch(plt.Rectangle(
                (-WALL_HALF, -WALL_HALF), 2 * WALL_HALF, 2 * WALL_HALF,
                linewidth=1.5, edgecolor="black", facecolor="none",
                linestyle="-", zorder=4,
            ))

            if self.obstacle_manager is not None:
                for obs in self.obstacle_manager.active_obstacles:
                    is_moving = hasattr(obs, 'vel') and np.linalg.norm(obs.vel) > 1e-6
                    face_col  = "#FF0000"
                    edge_col  = "#FF0000"
                    ax_world.add_patch(Circle(
                        obs.pos, obs.radius,
                        facecolor=face_col, edgecolor=edge_col,
                        linewidth=1.5, alpha=0.85, zorder=7,
                    ))
                    if is_moving:
                        ax_world.annotate(
                            "", xy=obs.pos + obs.vel * 0.5,
                            xytext=obs.pos,
                            arrowprops=dict(arrowstyle="->", color="#ff7043",
                                           lw=1.5, alpha=0.9),
                            zorder=8,
                        )

            positions = np.array([r.pos for r in self.robots])
            in_violation = set()
            for ii in range(N_ROBOTS):
                for jj in range(ii + 1, N_ROBOTS):
                    if np.linalg.norm(positions[ii] - positions[jj]) < D_SAFE:
                        in_violation.add(ii)
                        in_violation.add(jj)

            colliding = self.collision.active_set() if (ENABLE_COLLISION and self.collision) else set()

            for i, robot in enumerate(self.robots):
                color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                I_now = (self.field.intensity(robot.pos) / N_SOURCES) if self.field is not None else 0.0

                hist = np.array(self.histories[i][-60:])
                if len(hist) > 1:
                    segs = np.stack([hist[:-1], hist[1:]], axis=1)
                    rgba = np.zeros((len(segs), 4))
                    rgba[:, :3] = mcolors.to_rgb(color)
                    rgba[:, 3] = 0.05 + 0.55 * (np.arange(len(segs)) / len(hist))
                    ax_world.add_collection(LineCollection(
                        segs, colors=rgba, linewidths=1.0, zorder=5))

                traj = self._predicted_trajs[i]
                if traj is not None and len(traj) > 1:
                    ax_world.plot(
                        traj[:, 0], traj[:, 1],
                        color=color, lw=0.8, alpha=0.35,
                        linestyle=":", zorder=5,
                    )
                    ax_world.plot(
                        traj[-1, 0], traj[-1, 1],
                        "x", color=color, markersize=4, alpha=0.5, zorder=5,
                    )



                edge_col = VIOL_COLOR if i in in_violation else "#222222"
                # body_color = "black" if self.solver_statuses[i] != 0 else color
                body_color = color
                ax_world.add_patch(Ellipse(
                    xy=robot.pos, width=ROBOT_A * 2, height=ROBOT_B * 2,
                    angle=np.rad2deg(robot.theta),
                    facecolor=body_color, edgecolor=edge_col,
                    linewidth=1.4 if i in in_violation else 0.7,
                    alpha=0.92, zorder=6,
                ))

                if ENABLE_COLLISION and self.collision:
                    flash_alpha = self.collision.flash_alpha(i)
                    if flash_alpha > 0:
                        flash_ring = Ellipse(
                            xy=robot.pos,
                            width=(ROBOT_A * 2) * (1.0 + 0.6 * flash_alpha),
                            height=(ROBOT_B * 2) * (1.0 + 0.6 * flash_alpha),
                            angle=np.rad2deg(robot.theta),
                            facecolor="none", edgecolor="white",
                            linewidth=2.0 * flash_alpha, alpha=flash_alpha, zorder=9,
                        )
                        ax_world.add_patch(flash_ring)
                        accent_ring = Ellipse(
                            xy=robot.pos,
                            width=(ROBOT_A * 2) * (1.0 + 0.3 * flash_alpha),
                            height=(ROBOT_B * 2) * (1.0 + 0.3 * flash_alpha),
                            angle=np.rad2deg(robot.theta),
                            facecolor="none", edgecolor=color,
                            linewidth=1.5 * flash_alpha, alpha=0.7 * flash_alpha, zorder=8,
                        )
                        ax_world.add_patch(accent_ring)

                label_pos = robot.pos - (ROBOT_A * 0.5) * np.array(
                    [np.cos(robot.theta), np.sin(robot.theta)])
                ax_world.text(
                    label_pos[0], label_pos[1],
                    str(i + 1),
                    color="white", fontsize=7, ha="center", va="center",
                    fontweight="bold", zorder=8,
                )

                _, ref_theta, _ = self._goal_from_behavior_with_collision(i)
                arrow_end = robot.pos + 0.02 * np.array(
                    [np.cos(ref_theta), np.sin(ref_theta)])
                ax_world.annotate(
                    "", xy=arrow_end, xytext=robot.pos,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.8),
                    zorder=6,
                )

                orient_end = robot.pos + 0.015 * np.array(
                    [np.cos(robot.theta), np.sin(robot.theta)])
                ax_world.annotate(
                    "", xy=orient_end, xytext=robot.pos,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2, alpha=0.9),
                    zorder=7,
                )


                others = []
                for j in range(N_ROBOTS):
                    if j != i:
                        dist = np.linalg.norm(robot.pos - self.robots[j].pos)
                        if dist <= COMM_RADIUS:
                            others.append(j)

                others.sort(key=lambda j: np.linalg.norm(robot.pos - self.robots[j].pos))
                nearest = others[:MAX_NEIGHBOURS]

                for j in nearest:
                    nb_pos = self.robots[j].pos
                    ax_world.plot([robot.pos[0], nb_pos[0]],
                                  [robot.pos[1], nb_pos[1]],
                                  color=color,
                                  lw=(1.7 if PANEL_TITLE else 0.5),
                                  alpha=(0.6 if PANEL_TITLE else 0.4),
                                  linestyle=((0, (4, 3)) if PANEL_TITLE else "--"),
                                  zorder=4)

            _has_obs = (self.obstacle_manager is not None
                        and len(self.obstacle_manager.active_obstacles) > 0)
            if SHOW_VIDEO_LEGEND:
              ax_world.legend(
                handles=_video_legend_handles(_has_obs, show_light=bool(getattr(self.field, 'sources', None))),
                loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3,
                frameon=True, fancybox=False, edgecolor="#bbbbbb", facecolor="white",
                fontsize=9, handlelength=1.8, columnspacing=1.2, borderpad=0.5,
            )

            n_viol    = len(in_violation) // 2
            min_d_now = self.min_dist_log[-1] if self.min_dist_log else 0.0
            ax_world.set_xlim(-0.5, 0.5)
            ax_world.set_ylim(-0.5, 0.5)

            n_colliding = len(colliding)
            col_str = f"  |  {n_colliding} colliding (total: {self.total_collisions})" if ENABLE_COLLISION else ""

            ax_world.set_title(
                f"t = {self.current_time:.1f} s    min dist = {min_d_now*100:.1f} cm    "
                f"violations = {n_viol}{col_str}",
                color=LABEL_COLOR, fontsize=(19 if PANEL_TITLE else 9), pad=4,
            )


            if self.show_right_panel:
                ax_board.clear()
                ax_board.set_facecolor(PANEL_COLOR)
                ax_board.axis("off")
                if self.field is None:
                    ax_board.set_title("Light off",
                                       color=BOARD_TITLE_COL, fontsize=10)
                else:
                    ax_board.set_title("Light sensor (current)",
                                       color=BOARD_TITLE_COL, fontsize=10)

                    ranking = sorted(range(N_ROBOTS),
                                     key=lambda ii: -self.field.intensity(self.robots[ii].pos))
                    for rank, ii in enumerate(ranking):
                        color = ROBOT_COLORS[ii % len(ROBOT_COLORS)]
                        I_now = self.field.intensity(self.robots[ii].pos) / N_SOURCES
                        y     = 0.92 - rank * (0.88 / N_ROBOTS)
                        ax_board.barh(y, 0.70 * I_now, height=0.07, left=0.25,
                                      color=color, alpha=0.40,
                                      transform=ax_board.transAxes)
                        ax_board.text(0.02, y + 0.01,
                                      f"P{rank+1} {str(ii + 1)}",
                                      color=color, fontsize=8, fontweight="bold",
                                      transform=ax_board.transAxes)
                        ax_board.text(0.75, y + 0.01, f"{I_now*100:.1f}%",
                                      color=TEXT_COLOR, fontsize=8,
                                      transform=ax_board.transAxes, fontfamily="monospace")
                        for jj in range(int(I_now * 10)):
                            ax_board.text(0.26 + jj * 0.045, y + 0.005,
                                          "▌", color=color, fontsize=8, alpha=0.85,
                                          transform=ax_board.transAxes)

                ax_intensity.clear()
                ax_intensity.set_facecolor(PANEL_COLOR)
                ax_intensity.set_title("Sensor reading",
                                       color=TITLE_COLOR, fontsize=10)
                ax_intensity.set_xlabel("Time (s)", color=LABEL_COLOR, fontsize=9)
                ax_intensity.set_ylabel("Intensity",  color=LABEL_COLOR, fontsize=9)
                ax_intensity.tick_params(colors=LABEL_COLOR, labelsize=8)
                for spine in ax_intensity.spines.values():
                    spine.set_edgecolor(SPINE_COLOR)
                ax_intensity.spines["top"].set_visible(False)

                times = np.array(self.time_history[1:])
                for i in range(N_ROBOTS):
                    logs = np.array(self.intensity_log[i]) / N_SOURCES
                    if len(logs) > 1:
                        ax_intensity.plot(
                            times[-len(logs):], logs,
                            color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                            lw=1.2, alpha=0.85,
                            label=str(i + 1),
                        )

                ax2 = ax_intensity.twinx()
                ax2.set_facecolor(PANEL_COLOR)
                ax2.tick_params(colors=MINDIST_COLOR, labelsize=8)
                ax2.yaxis.label.set_color(MINDIST_COLOR)
                ax2.set_ylabel("Min dist (cm)", color=MINDIST_COLOR, fontsize=9)
                ax2.spines["top"].set_visible(False)
                md = np.array(self.min_dist_log) * 100
                if len(md) > 1:
                    ax2.plot(times[-len(md):], md,
                             color=MINDIST_COLOR, lw=1.5, alpha=0.9,
                             linestyle="--", label="min dist")
                    ax2.axhline(D_SAFE * 100, color="#cc0000",
                                lw=1.0, linestyle=":", alpha=0.7)
                ax2.set_ylim(0, max(10.0, md.max() * 1.1) if len(md) else 10.0)
                for spine in ax2.spines.values():
                    spine.set_edgecolor(SPINE_COLOR)

                ax_intensity.set_xlim(max(0, self.current_time - 20),
                                      self.current_time + 0.5)
                ax_intensity.set_ylim(0, 1.1)
                ax_intensity.legend(fontsize=8, facecolor=PANEL_COLOR,
                                    labelcolor=TEXT_COLOR, ncol=2, loc="upper left")
                ax_intensity.grid(True, color=GRID_COLOR, alpha=0.8)

                ax_v.clear()
                ax_v.set_facecolor(PANEL_COLOR)
                ax_v.set_title("Linear velocity $v$ [m/s]",
                               color=TITLE_COLOR, fontsize=10)
                ax_v.set_xlabel("Time (s)", color=LABEL_COLOR, fontsize=9)
                ax_v.set_ylabel("$v$ [m/s]", color=LABEL_COLOR, fontsize=9)
                ax_v.tick_params(colors=LABEL_COLOR, labelsize=8)
                for spine in ax_v.spines.values():
                    spine.set_edgecolor(SPINE_COLOR)
                ax_v.spines["top"].set_visible(False)
                ax_v.spines["right"].set_visible(False)
                for i in range(N_ROBOTS):
                    vlog = np.asarray(self.v_log[i])
                    if vlog.ndim == 2 and len(vlog) > 1:
                        vs = np.linalg.norm(vlog, axis=1)
                        ax_v.plot(times[-len(vs):], vs,
                                  color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                                  lw=1.1, alpha=0.85,
                                  label=str(i + 1))
                ax_v.axhline(V_TARGET, color="#888888", lw=1.0,
                             linestyle=":", alpha=0.6, label=f"v_ref={V_TARGET}")
                ax_v.set_xlim(max(0, self.current_time - 20), self.current_time + 0.5)
                ax_v.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR,
                            ncol=5, loc="upper left")
                ax_v.grid(True, color=GRID_COLOR, alpha=0.8)

                ax_w.clear()
                ax_w.set_facecolor(PANEL_COLOR)
                ax_w.set_title(r"Angular velocity $\omega$ [rad/s]",
                               color=TITLE_COLOR, fontsize=10)
                ax_w.set_xlabel("Time (s)", color=LABEL_COLOR, fontsize=9)
                ax_w.set_ylabel(r"$\omega$ [rad/s]", color=LABEL_COLOR, fontsize=9)
                ax_w.tick_params(colors=LABEL_COLOR, labelsize=8)
                for spine in ax_w.spines.values():
                    spine.set_edgecolor(SPINE_COLOR)
                ax_w.spines["top"].set_visible(False)
                ax_w.spines["right"].set_visible(False)
                for i in range(N_ROBOTS):
                    ws = np.array(self.w_log[i])
                    if len(ws) > 1:
                        ax_w.plot(times[-len(ws):], ws,
                                  color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                                  lw=1.1, alpha=0.85,
                                  label=str(i + 1))
                ax_w.axhline(0, color="#888888", lw=1.0, linestyle=":", alpha=0.5)
                ax_w.set_xlim(max(0, self.current_time - 20), self.current_time + 0.5)
                ax_w.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR,
                            ncol=5, loc="upper left")
                ax_w.grid(True, color=GRID_COLOR, alpha=0.8)

                fig.tight_layout(rect=[0, 0, 1, 0.97])
            else:
                fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

            if INTERP > 1:
                for i, rb in enumerate(self.robots):
                    rb.pos = self._interp_cur[i][0].copy()
                    rb.theta = self._interp_cur[i][1]
                if r == INTERP - 1:
                    self._interp_prev = self._interp_cur
            return []

        anim_obj = animation.FuncAnimation(
            fig, animate, frames=MAX_STEPS * INTERP,
            interval=DRAW_INTERVAL, blit=False,
        )

        _save_kwargs = {} if SAVE_DPI is None else {"dpi": SAVE_DPI}

        if SAVE_GIF:
            os.makedirs(GIF_DIR, exist_ok=True)
            _gif_path = os.path.join(GIF_DIR, GIF_NAME)
            print(f"Saving animation to {_gif_path} …")
            anim_obj.save(_gif_path, writer="pillow", fps=int(1000 / DRAW_INTERVAL),
                          **_save_kwargs)
            print("Save complete.")

        if SAVE_VIDEO:
            os.makedirs(VIDEO_DIR, exist_ok=True)
            _video_path = os.path.join(VIDEO_DIR, VIDEO_NAME)
            print(f"Saving animation to {_video_path} …")
            anim_obj.save(_video_path, writer="ffmpeg", fps=int(1000 / DRAW_INTERVAL),
                          extra_args=["-crf", "16", "-preset", "fast",
                                      "-pix_fmt", "yuv420p"],
                          **_save_kwargs)
            print("Save complete.")



        plt.show()
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("Phototaxis Distributed Predictive Flocking - Summary")
        print("=" * 60)
        print(f"  Duration:         {self.current_time:.1f} s  |  N={N_ROBOTS} robots")
        print(f"  D_safe:           {D_SAFE*100:.0f} cm  |  Weights: w={self.w_soft:g}, qc={self.q_coh:g}, qa={self.q_ali:g}")
        print(f"  Consensus iters:  {MAX_CONSENSUS_ITER} per timestep")
        print(f"  Sources:          {N_SOURCES} Gaussian (σ={SOURCE_SIGMA} m)")
        if self.min_dist_log:
            print(f"  Min separation (all time): {min(self.min_dist_log)*100:.2f} cm")
            n_viol = sum(1 for d in self.min_dist_log if d < D_SAFE)
            print(f"  Steps with violation:      {n_viol} / {len(self.min_dist_log)}")
            if ENABLE_COLLISION:
                print(f"  Total collision events:    {self.total_collisions}")
        print("-" * 60)
        for i in range(N_ROBOTS):
            avg_I  = (np.mean(self.intensity_log[i]) / N_SOURCES
                      if self.intensity_log[i] else 0.0)
            avg_st = np.mean(self.solve_times[i]) if self.solve_times[i] else 0.0
            print(f"  {str(i + 1)}: "
                  f"avg intensity {avg_I*100:.1f}%  |  "
                  f"avg solve {avg_st:.0f} μs  "
                  f"(×{MAX_CONSENSUS_ITER} = {avg_st*MAX_CONSENSUS_ITER:.0f} μs/step)")

            max_st = max(self.solve_times[i]) if self.solve_times[i] else 0
            print(f"  {str(i + 1)}: max solve {max_st:.0f} μs")
        print("=" * 60)


def main():
    print("=" * 60)
    print(f"Reynolds Swarm - Distributed Predictive Flocking")
    print(f"  Behavior Mode : {BEHAVIOR_MODE.value.upper()}")
    print(f"  N_ROBOTS={N_ROBOTS}  |  N_SOURCES={N_SOURCES}")
    print(f"  ARENA=±{ARENA_HALF}m  |  V={V_TARGET} m/s  |  D_safe={D_SAFE*100:.0f}cm")
    print(f"  Consensus iters per step: {MAX_CONSENSUS_ITER}")
    print("="*60)
    print("  BEHAVIOR MODES (change BEHAVIOR_MODE constant)")
    print("    PHOTOTAXIS_NO_REYNOLDS - phototaxis without cohesion/alignment/separation (current)")
    print("    NO_LIGHT - random walk with light off, Reynolds rules active")
    print("    PHOTOTAXIS_ORBITAL_CONTRACTING - stable orbit with inward spiral")
    print("    PHOTOTAXIS_ORBITAL - orbit the light source (rotated gradient)")
    print("    PHOTOTAXIS  - gradient ascent on light field")
    print("    ORBIT       - orbit a fixed center point (baseline)")
    print("=" * 60)

    PhototaxisSim().run()


if __name__ == "__main__":
    main()
