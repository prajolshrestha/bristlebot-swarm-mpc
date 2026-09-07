"""The original combined simulator and viewer, superseded by headless_sim_engine and viz_engine."""


import sys
import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from expert_src.Ellipse2dObj import EllipticalBot2D
from expert_src.controller import (
    CasadiAcadosEllipticalController, D_SAFE, D_SAFE_OBS, MAX_CONSENSUS_ITER,
    W_SOFT, W_SOFT_LIN, W_SOFT_OBS, W_SOFT_OBS_LIN,
    Q_COHESION_DEFAULT, Q_ALIGN_DEFAULT,
)
from swarm_dynamics import ReynoldsBehavior, SwarmBehavior
from light_source_model import LightField, SourceMotionMode
from obstacle_model import ObstacleManager, StaticObstacle, MovingObstacle
from collision_response import CollisionHandler, _ellipse_overlap_fast

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "expert_src", "config", "solver_ellipse_mpc.yaml")
with open(_CFG_PATH) as _cf:
    _CFG = yaml.safe_load(_cf)
_CFG_COMM = _CFG.get("communication", {})
_CFG_SOLVER = _CFG.get("solver_creation", {})
_CFG_BOUNDS = _CFG.get("model_bounds", {})

N_ROBOTS       = 25
DT             = float(_CFG_SOLVER.get("Ts", 0.1))
MPC_HORIZON    = int(_CFG_SOLVER.get("N", 20))
ARENA_HALF    = float(_CFG_BOUNDS.get("px_max", 0.4))
WALL_OUTSET   = 0.05
WALL_HALF     = ARENA_HALF + WALL_OUTSET
COMM_RADIUS   = float(_CFG_COMM.get("comm_radius", 0.15))

BEHAVIOR_MODE  = SwarmBehavior.PHOTOTAXIS_ORBITAL
ORBIT_CENTER   = np.array([0.0, 0.0])
ORBIT_RADIUS   = 0.25

N_SOURCES      = 1
SOURCE_SIGMA   = 0.10
SOURCE_SPEED   = 0.15
SOURCE_SEED    = 7
SOURCE_MOTION_MODE = SourceMotionMode.STATIONARY
SOURCE_ROT_RADIUS  = 0.25
SOURCE_PULSING     = False

V_TARGET       = 0.10
LOOKAHEAD_DIST = 0.07
EPSILON_GRAD   = 1e-6

Q_POS          = 60.0
Q_THETA        = 80.0
Q_VEL          = 1.5
Q_OM           = 1.5
R_U            = 0.01

Q_COHESION     = Q_COHESION_DEFAULT
Q_ALIGN        = Q_ALIGN_DEFAULT


OBSTACLE_RADIUS = 0.03

OBSTACLE_DEFS = [
    {"type": "static",  "pos": [ 0.15,  0.10], "active": True},
    {"type": "static",  "pos": [-0.15, -0.10], "active": True},
]
ENABLE_OBSTACLES = False

ENABLE_COLLISION       = True
COLLISION_STEER_DECAY  = 0.85
COLLISION_FLASH_FRAMES = 12

MAX_STEPS      = 200
HEATMAP_RES    = 20
DRAW_INTERVAL  = 10
SAVE_GIF       = False
GIF_NAME       = f"reynolds_swarm_{BEHAVIOR_MODE.value}_tcoupled_sync_all_soft_wsoft_{W_SOFT:g}.gif"
SAVE_VIDEO     = False
VIDEO_NAME     = f"reynolds_swarm_{BEHAVIOR_MODE.value}_tcoupled_sync_all_soft_wsoft_{W_SOFT:g}.mp4"
SUPP_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supplimentary_videos")
GIF_DIR   = os.path.join(SUPP_DIR, "gif")
VIDEO_DIR = os.path.join(SUPP_DIR, "video")
SHOW_RIGHT_PANEL = False

ROBOT_A = 0.03
ROBOT_A = 0.03
ROBOT_B = 0.015

ROBOT_COLORS = [
    "#E65100", "#1565C0", "#2E7D32", "#6A1B9A",
    "#AD1457", "#00695C", "#4527A0", "#558B2F",
    "#F9A825", "#C62828",
]
ROBOT_NAMES = [str(n) for n in range(1, N_ROBOTS + 1)]

BG_COLOR        = "white"
PANEL_COLOR     = "#f5f5f5"
GRID_COLOR      = "#dddddd"
TEXT_COLOR      = "#111111"
LABEL_COLOR     = "#333333"
SPINE_COLOR     = "#aaaaaa"
TITLE_COLOR     = "#111111"
BOARD_TITLE_COL = "#b45309"
VIOL_COLOR      = "#cc0000"
MINDIST_COLOR   = "#cc2222"
SAFE_DASH_OK    = "#1565C0"

class PhototaxisSim:

    def __init__(self,
                 q_pos=Q_POS, q_theta=Q_THETA, q_vel=Q_VEL,
                 q_om=Q_OM,   r_u=R_U,
                 w_soft=W_SOFT, q_coh=Q_COHESION, q_ali=Q_ALIGN,
                 w_soft_obs=W_SOFT_OBS,
                 w_soft_lin=W_SOFT_LIN, w_soft_obs_lin=W_SOFT_OBS_LIN,
                 show_right_panel=SHOW_RIGHT_PANEL):
        if BEHAVIOR_MODE == SwarmBehavior.NO_LIGHT:
            self.field = None
        else:
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

            self.field = LightField(
                motion_mode=motion_mode,
                rotation_radius=SOURCE_ROT_RADIUS,
                pulsing=SOURCE_PULSING,
            )

        self.q_pos      = q_pos
        self.q_theta    = q_theta
        self.q_vel      = q_vel
        self.q_om       = q_om
        self.r_u        = r_u
        if BEHAVIOR_MODE == SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS:
            w_soft, q_coh, q_ali, w_soft_obs = 0.0, 0.0, 0.0, 0.0
            w_soft_lin, w_soft_obs_lin = 0.0, 0.0
        self.w_soft     = w_soft
        self.q_coh      = float(q_coh)
        self.q_ali      = q_ali
        self.w_soft_obs = w_soft_obs
        self.w_soft_lin = w_soft_lin
        self.w_soft_obs_lin = w_soft_obs_lin
        self.show_right_panel = show_right_panel

        rng        = np.random.default_rng(globals().get("SPAWN_SEED", 42))

        self.behavior_engine = ReynoldsBehavior(
            arena_half   = ARENA_HALF,
            orbit_center = ORBIT_CENTER,
            orbit_radius = ORBIT_RADIUS,
            lookahead    = LOOKAHEAD_DIST,
            v_target     = V_TARGET,
        )

        self.robots:        list[EllipticalBot2D]                  = []
        self.controllers:   list[CasadiAcadosEllipticalController] = []
        self.histories:     list[list]                             = []
        self.intensity_log: list[list]   = [[] for _ in range(N_ROBOTS)]
        self.v_log:         list[list]   = [[] for _ in range(N_ROBOTS)]
        self.w_log:         list[list]   = [[] for _ in range(N_ROBOTS)]
        self.min_dist_log:  list[float]  = []

        self._predicted_trajs: list[np.ndarray | None] = [None] * N_ROBOTS

        self.solve_times: list[list] = [[] for _ in range(N_ROBOTS)]
        self.solver_statuses: list[int] = [0] * N_ROBOTS

        self.obstacle_manager = self._build_obstacle_manager()

        INIT_MIN_DIST = D_SAFE
        for i in range(N_ROBOTS):
            pos = None
            for _ in range(3000):
                cand = rng.uniform(-ARENA_HALF * 0.95, ARENA_HALF * 0.95, 2)
                if not all(np.linalg.norm(cand - r.pos) >= INIT_MIN_DIST
                           for r in self.robots):
                    continue
                if self.obstacle_manager is not None:
                    if any(np.linalg.norm(cand - obs.pos) < D_SAFE_OBS
                           for obs in self.obstacle_manager.active_obstacles):
                        continue
                pos = cand
                break
            if pos is None:
                pos = rng.uniform(-ARENA_HALF * 0.95, ARENA_HALF * 0.95, 2)

            theta = rng.uniform(-np.pi, np.pi)
            bot   = EllipticalBot2D(
                r=pos.copy(), theta_rad=theta,
                vmag=0.0, w=0.0,
                a=ROBOT_A, b=ROBOT_B, npoints=30,
            )
            self.robots.append(bot)
            self.histories.append([pos.copy()])

            ctrl = CasadiAcadosEllipticalController(
                dt=DT, N=MPC_HORIZON,
                use_noisy_observations=False,
                use_process_noise=False,
                process_noise_std=0.005,
                process_noise_std_th=0.005,
                w_soft=self.w_soft,
                q_cohesion=self.q_coh,
                q_align=self.q_ali,
                w_soft_obs=self.w_soft_obs,
                w_soft_lin=self.w_soft_lin,
                w_soft_obs_lin=self.w_soft_obs_lin,
            )
            ctrl.q_pos   = self.q_pos
            ctrl.q_theta = self.q_theta
            ctrl.q_vel   = self.q_vel
            ctrl.q_om    = self.q_om
            ctrl.r_u     = self.r_u
            if BEHAVIOR_MODE == SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS:
                ctrl.set_avoidance_enabled(False)
            self.controllers.append(ctrl)

        self.collision = None
        if ENABLE_COLLISION:
            self.collision = CollisionHandler(
                self.robots,
                steer_decay=COLLISION_STEER_DECAY,
                flash_frames=COLLISION_FLASH_FRAMES,
            )
        self.active_collisions = set()
        self.total_collisions = 0

        self.current_time = 0.0
        self.time_history = [0.0]
        self.kdtree = None

    @staticmethod
    def _build_obstacle_manager() -> ObstacleManager | None:
        """Construct ObstacleManager from OBSTACLE_DEFS config."""
        if not ENABLE_OBSTACLES:
            return None
        mgr = ObstacleManager(dt=DT, arena_half=ARENA_HALF, max_obs=5, obs_range=0.30)
        for d in OBSTACLE_DEFS:
            if not d.get("active", True):
                continue
            pos = np.array(d["pos"], dtype=float)
            if d["type"] == "static":
                mgr.add(StaticObstacle(pos=pos, radius=OBSTACLE_RADIUS))
            elif d["type"] == "moving":
                vel    = np.array(d.get("vel",    [0.0, 0.0]), dtype=float)
                bounce = bool(d.get("bounce", True))
                mgr.add(MovingObstacle(
                    pos=pos, vel=vel, radius=OBSTACLE_RADIUS,
                    arena_half=ARENA_HALF, bounce=bounce,
                ))
        return mgr

    def _goal_from_behavior(self, robot_idx: int):
        """Compute goal/reference using the active SwarmBehavior mode."""
        robot     = self.robots[robot_idx]
        
        neighbor_indices = self.kdtree.query_ball_point(robot.pos, COMM_RADIUS)
        neighbors = [self.robots[j] for j in neighbor_indices if j != robot_idx]
                    
        goal_pos, ref_theta, ref_vel = self.behavior_engine.compute_goal(
            robot     = robot,
            neighbors = neighbors,
            behavior  = BEHAVIOR_MODE,
            field     = self.field,
        )

        if ENABLE_COLLISION and self.collision is not None:
            ref_theta += self.collision.theta_bias(robot_idx)

        return goal_pos, ref_theta, ref_vel

    def _get_neighbour_trajs(self, robot_idx: int) -> list[np.ndarray]:
        """
        Collect predicted trajectories for neighbours of robot i within COMM_RADIUS.
        """
        from expert_src.controller import MAX_NEIGHBOURS as K

        pos_i   = self.robots[robot_idx].pos
        
        dists, indices = self.kdtree.query(pos_i, k=K+1, distance_upper_bound=COMM_RADIUS)
        
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
            dists = [dists]

        trajs = []
        for d, j in zip(dists, indices):
            if j == robot_idx or j >= N_ROBOTS or d > COMM_RADIUS:
                continue
            
            if self._predicted_trajs[j] is not None:
                trajs.append(self._predicted_trajs[j])
            else:
                nb   = self.robots[j]
                state4 = np.array([
                    nb.pos[0], nb.pos[1],
                    nb.velocity[0], nb.velocity[1]
                ])
                trajs.append(state4)
        return trajs

    def step(self):
        """Advance one control timestep using the trajectory-coupled consensus."""
        if self.field is not None:
            self.field.step(DT)

        if self.obstacle_manager is not None:
            self.obstacle_manager.step(DT)

        self.kdtree = KDTree(np.array([r.pos for r in self.robots]))

        all_nb_trajs = []
        for i in range(N_ROBOTS):
            all_nb_trajs.append(self._get_neighbour_trajs(i))

        controls = []
        new_predicted_trajs = []
        for i, (robot, ctrl) in enumerate(zip(self.robots, self.controllers)):

            nb_trajs = all_nb_trajs[i]
            goal_pos, ref_theta, ref_vel = self._goal_from_behavior(i)
            
            t0 = time.perf_counter()
            u  = ctrl.compute_control(
                robot,
                goal_pos,
                ref_theta=ref_theta,
                ref_vel=ref_vel,
                ref_omega=0.0,
                neighbour_trajs=nb_trajs,
                obstacle_manager=self.obstacle_manager,
                robot_name=ROBOT_NAMES[i % len(ROBOT_NAMES)],
            )
            self.solve_times[i].append((time.perf_counter() - t0) * 1e6)

            controls.append(u)
            new_predicted_trajs.append(ctrl.get_predicted_trajectory())
            self.solver_statuses[i] = ctrl.last_solver_status

        for i in range(N_ROBOTS):
            self._predicted_trajs[i] = new_predicted_trajs[i]

        for i, (robot, ctrl, u) in enumerate(zip(self.robots, self.controllers, controls)):
            ctrl.apply_control(robot, u)
            self.histories[i].append(robot.pos.copy())
            intensity = self.field.intensity(robot.pos) if self.field is not None else 0.0
            self.intensity_log[i].append(intensity)
            self.v_log[i].append(float(np.linalg.norm(robot.velocity)))
            self.w_log[i].append(float(robot.ang_velocity))

        if ENABLE_COLLISION and self.collision is not None:
            new_active_collisions = set()
            
            for ii in range(N_ROBOTS):
                ri = self.robots[ii]
                for jj in range(ii + 1, N_ROBOTS):
                    rj = self.robots[jj]
                    overlapping, pen, _ = _ellipse_overlap_fast(
                        ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                        rj.pos, float(rj.theta), rj.bot_width * 0.5, rj.bot_height * 0.5,
                    )
                    if overlapping and pen > 0.0:
                        pair = frozenset({ii, jj})
                        new_active_collisions.add(pair)
                        if pair not in self.active_collisions:
                            self.total_collisions += 1

            if self.obstacle_manager is not None:
                active_obs = self.obstacle_manager.active_obstacles
                for ii in range(N_ROBOTS):
                    ri = self.robots[ii]
                    for obs_idx, obs in enumerate(active_obs):
                        overlapping, pen, _ = _ellipse_overlap_fast(
                            ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                            obs.pos, 0.0, obs.radius, obs.radius,
                        )
                        if overlapping and pen > 0.0:
                            pair = frozenset({ii, f"obs_{obs_idx}"})
                            new_active_collisions.add(pair)
                            if pair not in self.active_collisions:
                                self.total_collisions += 1

            self.active_collisions = new_active_collisions

            self.collision.step(
                obstacle_manager=self.obstacle_manager if ENABLE_OBSTACLES else None,
                arena_half=WALL_HALF,
            )

        n_fails = sum(1 for s in self.solver_statuses if s != 0)
        step_idx = len(self.time_history)
        if n_fails > 0:
            print(f"Step {step_idx:3d} | Failed Solvers: {n_fails:2d} / {N_ROBOTS}")

        positions = np.array([r.pos for r in self.robots])
        min_d = float("inf")
        for ii in range(N_ROBOTS):
            for jj in range(ii + 1, N_ROBOTS):
                d = np.linalg.norm(positions[ii] - positions[jj])
                if d < min_d:
                    min_d = d
        self.min_dist_log.append(min_d)

        self.current_time += DT
        self.time_history.append(self.current_time)

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
                f"Distributed Predictive Flocking: {BEHAVIOR_MODE.value.replace('_', ' ').title()}",
                color=TITLE_COLOR, fontsize=13, y=0.99,
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
                f"Distributed Predictive Flocking: {BEHAVIOR_MODE.value.replace('_', ' ').title()}",
                color=TITLE_COLOR, fontsize=13, y=0.96,
            )
            ax_world     = fig.add_subplot(1, 1, 1, facecolor=PANEL_COLOR)
            ax_board     = None
            ax_intensity = None
            ax_v         = None
            ax_w         = None
            axes_to_style = [ax_world]

        for ax in axes_to_style:
            ax.tick_params(colors=LABEL_COLOR, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COLOR)
            ax.xaxis.label.set_color(LABEL_COLOR)
            ax.yaxis.label.set_color(LABEL_COLOR)

        def animate(frame):
            if frame >= MAX_STEPS:
                return []

            self.step()

            ax_world.clear()
            ax_world.set_facecolor(PANEL_COLOR)
            ax_world.set_aspect("equal")
            ax_world.set_xlabel("$x$ [m]", color=LABEL_COLOR, fontsize=9)
            ax_world.set_ylabel("$y$ [m]", color=LABEL_COLOR, fontsize=9)
            ax_world.tick_params(colors=LABEL_COLOR, labelsize=8)
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
                linewidth=1.5, edgecolor="#555577", facecolor="none",
                linestyle="--", zorder=4,
            ))

            if self.obstacle_manager is not None:
                for obs in self.obstacle_manager.active_obstacles:
                    is_moving = hasattr(obs, 'vel') and np.linalg.norm(obs.vel) > 1e-6
                    face_col  = "#546e7a" if is_moving else "#37474f"
                    edge_col  = "#ff7043" if is_moving else "#b0bec5"
                    ax_world.add_patch(Circle(
                        obs.pos, obs.radius,
                        facecolor=face_col, edgecolor=edge_col,
                        linewidth=1.5, alpha=0.85, zorder=7,
                    ))
                    ax_world.add_patch(Circle(
                        obs.pos, D_SAFE_OBS,
                        facecolor="none", edgecolor="#ff7043",
                        linewidth=0.8, linestyle=":", alpha=0.5, zorder=7,
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
                    for k in range(len(hist) - 1):
                        alpha = 0.05 + 0.55 * (k / len(hist))
                        ax_world.plot(hist[k:k+2, 0], hist[k:k+2, 1],
                                      color=color, lw=1.0, alpha=alpha, zorder=5)

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

                safe_color = VIOL_COLOR if i in in_violation else SAFE_DASH_OK
                safe_lw    = 1.5        if i in in_violation else 0.8
                safe_alpha = 0.9        if i in in_violation else 0.35
                ax_world.add_patch(Circle(
                    robot.pos, D_SAFE / 2, color=safe_color,
                    fill=False, linewidth=safe_lw,
                    linestyle="--", alpha=safe_alpha, zorder=5,
                ))

                glow_r = ROBOT_A * (1.2 + 1.5 * I_now)
                ax_world.add_patch(Circle(
                    robot.pos, glow_r, color=color,
                    alpha=0.12 + 0.15 * I_now, zorder=5, fill=True,
                ))

                edge_col = VIOL_COLOR if i in in_violation else "#222222"
                body_color = "black" if self.solver_statuses[i] != 0 else color
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
                    ROBOT_NAMES[i % len(ROBOT_NAMES)],
                    color="white", fontsize=7, ha="center", va="center",
                    fontweight="bold", zorder=8,
                )

                _, ref_theta, _ = self._goal_from_behavior(i)
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

                if ENABLE_COLLISION and self.collision:
                    bias = self.collision.theta_bias(i)
                    if abs(bias) > 0.05:
                        unbiased_theta = ref_theta - bias
                        arc_r  = ROBOT_A * 2.2
                        n_pts  = 12
                        angles = np.linspace(unbiased_theta, ref_theta, n_pts)
                        arc_x  = robot.pos[0] + arc_r * np.cos(angles)
                        arc_y  = robot.pos[1] + arc_r * np.sin(angles)
                        arc_alpha = min(abs(bias) / (np.pi / 2), 1.0)
                        ax_world.plot(arc_x, arc_y, color="white",
                                      lw=2.0 * arc_alpha, alpha=arc_alpha,
                                      zorder=10, solid_capstyle="round")
                        ax_world.plot(arc_x, arc_y, color=color,
                                      lw=1.1 * arc_alpha, alpha=0.9 * arc_alpha,
                                      zorder=11, solid_capstyle="round")
                        if n_pts >= 2:
                            ax_world.annotate(
                                "", xy=(arc_x[-1], arc_y[-1]),
                                xytext=(arc_x[-2], arc_y[-2]),
                                arrowprops=dict(arrowstyle="-|>", color=color,
                                                lw=1.0, mutation_scale=7),
                                zorder=12,
                            )

                others = []
                for j in range(N_ROBOTS):
                    if j != i:
                        dist = np.linalg.norm(robot.pos - self.robots[j].pos)
                        if dist <= COMM_RADIUS:
                            others.append(j)
                
                others.sort(key=lambda j: np.linalg.norm(robot.pos - self.robots[j].pos))
                from expert_src.controller import MAX_NEIGHBOURS as K
                nearest = others[:K]
                
                for j in nearest:
                    nb_pos = self.robots[j].pos
                    ax_world.plot([robot.pos[0], nb_pos[0]],
                                  [robot.pos[1], nb_pos[1]],
                                  color=color, lw=0.5, alpha=0.4, linestyle="--", zorder=4)

            n_viol    = len(in_violation) // 2
            min_d_now = self.min_dist_log[-1] if self.min_dist_log else 0.0
            ax_world.set_xlim(-0.5, 0.5)
            ax_world.set_ylim(-0.5, 0.5)

            n_colliding = len(colliding)
            col_str = f"  |  {n_colliding} colliding (total: {self.total_collisions})" if ENABLE_COLLISION else ""

            ax_world.set_title(
                f"t = {self.current_time:.1f} s    min dist = {min_d_now*100:.1f} cm    "
                f"violations = {n_viol}{col_str}",
                color=LABEL_COLOR, fontsize=9, pad=4,
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
                                      f"P{rank+1} {ROBOT_NAMES[ii % len(ROBOT_NAMES)]}",
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
                            label=ROBOT_NAMES[i % len(ROBOT_NAMES)],
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
                    vs = np.array(self.v_log[i])
                    if len(vs) > 1:
                        ax_v.plot(times[-len(vs):], vs,
                                  color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                                  lw=1.1, alpha=0.85,
                                  label=ROBOT_NAMES[i % len(ROBOT_NAMES)])
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
                                  label=ROBOT_NAMES[i % len(ROBOT_NAMES)])
                ax_w.axhline(0, color="#888888", lw=1.0, linestyle=":", alpha=0.5)
                ax_w.set_xlim(max(0, self.current_time - 20), self.current_time + 0.5)
                ax_w.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR,
                            ncol=5, loc="upper left")
                ax_w.grid(True, color=GRID_COLOR, alpha=0.8)

                fig.tight_layout(rect=[0, 0, 1, 0.97])
            else:
                fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
            return []

        anim_obj = animation.FuncAnimation(
            fig, animate, frames=MAX_STEPS,
            interval=DRAW_INTERVAL, blit=False,
        )

        if SAVE_GIF:
            os.makedirs(GIF_DIR, exist_ok=True)
            _gif_path = os.path.join(GIF_DIR, GIF_NAME)
            print(f"Saving animation to {_gif_path} …")
            anim_obj.save(_gif_path, writer="pillow", fps=int(1000 / DRAW_INTERVAL))
            print("Save complete.")

        if SAVE_VIDEO:
            os.makedirs(VIDEO_DIR, exist_ok=True)
            _video_path = os.path.join(VIDEO_DIR, VIDEO_NAME)
            print(f"Saving animation to {_video_path} …")
            anim_obj.save(_video_path, writer="ffmpeg", fps=int(1000 / DRAW_INTERVAL))
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
            print(f"  {ROBOT_NAMES[i % len(ROBOT_NAMES)]}: "
                  f"avg intensity {avg_I*100:.1f}%  |  "
                  f"avg solve {avg_st:.0f} μs  "
                  f"(×{MAX_CONSENSUS_ITER} = {avg_st*MAX_CONSENSUS_ITER:.0f} μs/step)")

            max_st = max(self.solve_times[i]) if self.solve_times[i] else 0
            print(f"  {ROBOT_NAMES[i % len(ROBOT_NAMES)]}: max solve {max_st:.0f} μs")
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
    print("    PHOTOTAXIS_NO_REYNOLDS - phototaxis without cohesion/alignment/separation")
    print("    NO_LIGHT - random walk with light off, Reynolds rules active")
    print("    PHOTOTAXIS_ORBITAL_CONTRACTING - stable orbit with inward spiral")
    print("    PHOTOTAXIS_ORBITAL - orbit the light source (rotated gradient)")
    print("    PHOTOTAXIS  - gradient ascent on light field")
    print("    ORBIT       - orbit a fixed center point (baseline)")
    print("=" * 60)
    PhototaxisSim().run()


if __name__ == "__main__":
    main()