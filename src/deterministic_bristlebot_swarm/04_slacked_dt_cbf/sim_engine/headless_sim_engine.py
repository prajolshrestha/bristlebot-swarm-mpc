"""The simulation loop: every robot solves its own problem each step, with no display."""

import os
import sys
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import csv
import json
from scipy.spatial import KDTree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from expert_src.Ellipse2dObj import EllipticalBot2D
from expert_src.controller import CasadiAcadosEllipticalController, D_SAFE, D_SAFE_OBS
from sim_engine.swarm_dynamics import ReynoldsBehavior, SwarmBehavior
from sim_engine.light_source_model import LightField, SourceMotionMode
from sim_engine.obstacle_model import ObstacleManager, StaticObstacle, MovingObstacle
from sim_engine.collision_response import CollisionHandler, _ellipse_overlap_fast

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "expert_src", "config", "solver_ellipse_mpc.yaml")
with open(_CFG_PATH) as _cf:
    _CFG = yaml.safe_load(_cf)
_CFG_REYNOLDS = _CFG.get("reynolds", {})
_CFG_COMM = _CFG.get("communication", {})
_CFG_COLLISION = _CFG.get("collision", {})
_CFG_CONTACT = _CFG.get("contact_response", {})
_CFG_OBSTACLE = _CFG.get("obstacle_avoidance", {})
_CFG_SOLVER = _CFG.get("solver_creation", {})
_CFG_BOUNDS = _CFG.get("model_bounds", {})

DT = float(_CFG_SOLVER.get("Ts", 0.1))
MPC_HORIZON = int(_CFG_SOLVER.get("N", 20))
ARENA_HALF = float(_CFG_BOUNDS.get("px_max", 0.4))
WALL_OUTSET = 0.05
WALL_HALF = ARENA_HALF + WALL_OUTSET
ROBOT_A = 0.03
ROBOT_B = 0.015
MAX_STEPS = 600
COMM_RADIUS = float(_CFG_COMM.get("comm_radius", 0.15))
MAX_CONSENSUS = int(_CFG_COMM.get("max_consensus", 2))

SPAWN_SEED = 42
SOURCE_MOTION_MODE = SourceMotionMode.ROTATION
SOURCE_ROT_RADIUS = 0.25
SOURCE_PULSING = False

GLOBAL_W_SOFT     = float(_CFG_COLLISION.get("w_soft", 1e8))
GLOBAL_Q_COH      = float(_CFG_REYNOLDS.get("q_cohesion", 2.0))
GLOBAL_Q_ALI      = float(_CFG_REYNOLDS.get("q_align", 5.0))
GLOBAL_W_SOFT_OBS = float(_CFG_OBSTACLE.get("w_soft_obs", 1e8))
GLOBAL_W_SOFT_LIN     = float(_CFG_COLLISION.get("w_soft_lin", 0.0))
GLOBAL_W_SOFT_OBS_LIN = float(_CFG_OBSTACLE.get("w_soft_obs_lin", 0.0))

ENABLE_COLLISION = bool(_CFG_CONTACT.get("enabled", True))
COLLISION_STEER_DECAY = float(_CFG_CONTACT.get("steer_decay", 0.85))
COLLISION_FLASH_FRAMES = 6

OBSTACLE_RADIUS   = 0.03
ENABLE_OBSTACLES  = True

OBSTACLE_DEFS = [
    {"type": "static", "pos": [ 0.15,  0.10], "active": True},
    {"type": "static", "pos": [-0.15, -0.10], "active": True},
]


OBSTACLE_DENSITY_COUNTS = {"free": 0, "sparse": 2, "cluttered": 12}


def build_obstacle_defs(count, seed=0):
    """Return an OBSTACLE_DEFS list of `count` static obstacles for the given seed.

    Same `count` every seed; positions are rejection-sampled inside the arena
    with a minimum inter-obstacle spacing so navigable corridors remain. `count`
    is one of the values in OBSTACLE_DENSITY_COUNTS (0/2/12), but any non-negative
    count is supported.
    """
    if count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    lim = 0.30
    min_sep = 0.12
    pts = []
    for _ in range(int(count)):
        cand = None
        for _try in range(2000):
            c = rng.uniform(-lim, lim, 2)
            if all(np.linalg.norm(c - p) >= min_sep for p in pts):
                cand = c
                break
        if cand is None:
            cand = rng.uniform(-lim, lim, 2)
        pts.append(cand)
    return [{"type": "static", "pos": [float(p[0]), float(p[1])], "active": True}
            for p in pts]

def _build_obstacle_manager(use_obstacles: "bool | None" = None) -> "ObstacleManager | None":
    """Build an ObstacleManager from OBSTACLE_DEFS. Returns None if disabled.

    use_obstacles=None falls back to the module ENABLE_OBSTACLES flag (the
    10-seed driver toggles it via H.ENABLE_OBSTACLES); callers may pass an
    explicit bool to override per-instance (the figure driver does this).
    """
    if use_obstacles is None:
        use_obstacles = ENABLE_OBSTACLES
    if not use_obstacles:
        return None
    mgr = ObstacleManager(dt=DT, arena_half=ARENA_HALF, max_obs=5, obs_range=0.30)
    for d in OBSTACLE_DEFS:
        if not d.get("active", True):
            continue
        pos = np.array(d["pos"], dtype=float)
        if d["type"] == "static":
            mgr.add(StaticObstacle(pos=pos, radius=OBSTACLE_RADIUS))
        elif d["type"] == "moving":
            vel    = np.array(d.get("vel", [0.0, 0.0]), dtype=float)
            bounce = bool(d.get("bounce", True))
            mgr.add(MovingObstacle(pos=pos, vel=vel, radius=OBSTACLE_RADIUS,
                                   arena_half=ARENA_HALF, bounce=bounce))
    return mgr

RESULTS_BASE_DIR = f"ablation_results_{GLOBAL_W_SOFT:g}"
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "results_v3_multiple_full_bound_sync")

N_LIST = [10, 25, 50, 75, 100, 200]

V_TARGET = 0.10
LOOKAHEAD_DIST = 0.07
ORBIT_RADIUS = 0.25


class HeadlessSim:
    def __init__(self, n_robots=10, behavior_mode=SwarmBehavior.PHOTOTAXIS,
                 q_coh=GLOBAL_Q_COH, q_ali=GLOBAL_Q_ALI, w_soft=GLOBAL_W_SOFT,
                 w_soft_obs=GLOBAL_W_SOFT_OBS,
                 w_soft_lin=GLOBAL_W_SOFT_LIN, w_soft_obs_lin=GLOBAL_W_SOFT_OBS_LIN,
                 max_consensus=MAX_CONSENSUS,
                 enable_obstacles: "bool | None" = None,
                 enable_collision: "bool | None" = None,
                 steer_decay: float = COLLISION_STEER_DECAY,
                 flash_frames: int = COLLISION_FLASH_FRAMES,
                 enable_avoidance: "bool | None" = None):
        self.n_robots = n_robots
        self.behavior_mode = behavior_mode
        self.q_coh = q_coh
        self.q_ali = q_ali
        self.w_soft = w_soft
        self.w_soft_obs = w_soft_obs
        self.w_soft_lin = w_soft_lin
        self.w_soft_obs_lin = w_soft_obs_lin
        self.max_consensus = max_consensus
        self.enable_collision = (ENABLE_COLLISION if enable_collision is None
                                 else bool(enable_collision))
        if enable_avoidance is None:
            enable_avoidance = (behavior_mode != SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS)
        self.enable_avoidance = bool(enable_avoidance)

        self.solve_times = [[] for _ in range(n_robots)]
        self.solver_statuses = [0] * n_robots
        self.failure_counts = [0] * n_robots

        motion_mode = SOURCE_MOTION_MODE
        if self.behavior_mode == SwarmBehavior.PHOTOTAXIS:
            motion_mode = SourceMotionMode.ROTATION
        elif self.behavior_mode in (
            SwarmBehavior.ORBIT,
            SwarmBehavior.PHOTOTAXIS_ORBITAL,
            SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING,
        ):
            motion_mode = SourceMotionMode.STATIONARY
        self.field = LightField(
            motion_mode=motion_mode,
            rotation_radius=SOURCE_ROT_RADIUS,
            pulsing=SOURCE_PULSING
        )
        self.behavior_engine = ReynoldsBehavior(
            arena_half=ARENA_HALF,
            orbit_center=np.zeros(2),
            orbit_radius=ORBIT_RADIUS,
            lookahead=LOOKAHEAD_DIST,
            v_target=V_TARGET,
        )

        self.robots = []
        self.controllers = []
        self.histories = []
        self.v_log = [[] for _ in range(n_robots)]
        self.heading_log = [[] for _ in range(n_robots)]
        self.w_log = [[] for _ in range(n_robots)]

        self._predicted_trajs = [None] * n_robots
        self.kdtree = None

        self.obstacle_manager = _build_obstacle_manager(enable_obstacles)

        rng = np.random.default_rng(SPAWN_SEED)
        init_min_dist = D_SAFE
        for i in range(n_robots):
            pos = None
            for _ in range(3000):
                cand = rng.uniform(-ARENA_HALF * 0.95, ARENA_HALF * 0.95, 2)
                if not all(np.linalg.norm(cand - r.pos) >= init_min_dist for r in self.robots):
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
            bot = EllipticalBot2D(r=pos.copy(), theta_rad=theta, vmag=0.0, w=0.0, a=ROBOT_A, b=ROBOT_B, npoints=30)
            self.robots.append(bot)
            self.histories.append([pos.copy()])

            ctrl = CasadiAcadosEllipticalController(
                dt=DT,
                N=MPC_HORIZON,
                w_soft=self.w_soft,
                q_cohesion=self.q_coh,
                q_align=self.q_ali,
                w_soft_obs=self.w_soft_obs,
                w_soft_lin=self.w_soft_lin,
                w_soft_obs_lin=self.w_soft_obs_lin,
            )
            ctrl.set_avoidance_enabled(self.enable_avoidance)
            self.controllers.append(ctrl)

        self._relax_spawn_overlaps(init_min_dist)

        self.current_time = 0.0
        self.time_history = [0.0]

        self.collision = None
        if self.enable_collision:
            self.collision = CollisionHandler(
                self.robots,
                steer_decay=steer_decay,
                flash_frames=flash_frames,
            )
        self.active_collisions = set()
        self.total_collisions = 0
        self.total_collisions_rr = 0
        self.total_collisions_ro = 0

        self.min_dist_log = []
        self.mean_dist_log = []
        self.coherence_log = []
        self.avg_min_sep_log = []

    def _relax_spawn_overlaps(self, min_dist, max_iters=300):
        """Resolve residual spawn overlaps by pushing robots into free space.

        The randomized spawn rejection-samples positions, but at high robot counts
        (roughly > 75 in this arena) the sampler saturates and its random fallback
        can place robots overlapping. This pass iteratively separates any pair
        whose centres are closer than ``min_dist`` (and pushes robots off
        obstacles), moving them into free space, then syncs each robot's initial
        history entry to the relaxed pose. One-time init cost; a no-op once no
        overlaps remain, so the <= 50 robot case is unaffected.
        """
        n = len(self.robots)
        if n < 2:
            return
        lim = ARENA_HALF * 0.98
        obstacles = (self.obstacle_manager.active_obstacles
                     if self.obstacle_manager is not None else [])

        for _ in range(max_iters):
            positions = np.array([r.pos for r in self.robots])
            pairs = KDTree(positions).query_pairs(r=min_dist)
            max_overlap = 0.0

            for i, j in pairs:
                pi, pj = self.robots[i].pos, self.robots[j].pos
                d_vec = pj - pi
                d = float(np.linalg.norm(d_vec))
                if d < 1e-9:
                    ang = (i * 2.3999632) % (2.0 * np.pi)
                    n_hat = np.array([np.cos(ang), np.sin(ang)])
                    d = 0.0
                else:
                    n_hat = d_vec / d
                shortfall = min_dist - d
                max_overlap = max(max_overlap, shortfall)
                push = 0.5 * shortfall + 1e-4
                ti = np.clip(pi - push * n_hat, -lim, lim)
                tj = np.clip(pj + push * n_hat, -lim, lim)
                self.robots[i].translate(ti - pi)
                self.robots[j].translate(tj - pj)

            for i in range(n):
                for obs in obstacles:
                    pi = self.robots[i].pos
                    d_vec = pi - obs.pos
                    d = float(np.linalg.norm(d_vec))
                    if d >= D_SAFE_OBS:
                        continue
                    n_hat = (d_vec / d) if d > 1e-9 else np.array([1.0, 0.0])
                    max_overlap = max(max_overlap, D_SAFE_OBS - d)
                    target = np.clip(obs.pos + n_hat * (D_SAFE_OBS + 1e-4), -lim, lim)
                    self.robots[i].translate(target - pi)

            if not pairs and max_overlap < 1e-4:
                break

        for i in range(n):
            self.histories[i][0] = self.robots[i].pos.copy()

        residual = KDTree(np.array([r.pos for r in self.robots])).query_pairs(r=min_dist - 1e-3)
        if residual:
            print(f"[spawn] warning: {len(residual)} robot pair(s) still within "
                  f"{min_dist*100:.1f} cm after relaxation "
                  f"(arena likely over-packed for n_robots={n}).")

    def _goal_from_behavior(self, robot_idx: int):
        robot = self.robots[robot_idx]

        neighbor_indices = self.kdtree.query_ball_point(robot.pos, COMM_RADIUS)
        neighbors = [self.robots[j] for j in neighbor_indices if j != robot_idx]

        return self.behavior_engine.compute_goal(
            robot=robot, neighbors=neighbors, behavior=self.behavior_mode, field=self.field
        )

    def _goal_from_behavior_with_collision(self, robot_idx: int):
        goal_pos, ref_theta, ref_vel = self._goal_from_behavior(robot_idx)
        if self.enable_collision and self.collision is not None:
            ref_theta += self.collision.theta_bias(robot_idx)
        return goal_pos, ref_theta, ref_vel

    def _detect_overlap_pairs(self):
        pairs = []
        for ii in range(self.n_robots):
            ri = self.robots[ii]
            for jj in range(ii + 1, self.n_robots):
                rj = self.robots[jj]
                overlapping, pen, _ = _ellipse_overlap_fast(
                    ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                    rj.pos, float(rj.theta), rj.bot_width * 0.5, rj.bot_height * 0.5,
                )
                if overlapping and pen > 0.0:
                    pairs.append((ii, jj))
        return pairs

    def _detect_robot_obstacle_overlaps(self):
        pairs = []
        if self.obstacle_manager is None:
            return pairs
        active_obs = self.obstacle_manager.active_obstacles
        for ii in range(self.n_robots):
            ri = self.robots[ii]
            for obs_idx, obs in enumerate(active_obs):
                overlapping, pen, _ = _ellipse_overlap_fast(
                    ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                    obs.pos, 0.0, obs.radius, obs.radius,
                )
                if overlapping and pen > 0.0:
                    pairs.append((ii, obs_idx))
        return pairs

    def _resolve_collisions(self):
        if not (self.enable_collision and self.collision is not None):
            return

        new_active_collisions = set()
        for ii in range(self.n_robots):
            ri = self.robots[ii]
            for jj in range(ii + 1, self.n_robots):
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
                        self.total_collisions_rr += 1

        if self.obstacle_manager is not None:
            active_obs = self.obstacle_manager.active_obstacles
            for ii in range(self.n_robots):
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
                            self.total_collisions_ro += 1

        self.active_collisions = new_active_collisions
        self.collision.step(
            obstacle_manager=self.obstacle_manager,
            arena_half=WALL_HALF,
        )

    def _get_neighbour_trajs(self, robot_idx: int):
        from expert_src.controller import MAX_NEIGHBOURS as K
        pos_i = self.robots[robot_idx].pos

        dists, indices = self.kdtree.query(pos_i, k=K + 1, distance_upper_bound=COMM_RADIUS)

        if isinstance(indices, (int, np.integer)):
            indices = [indices]
            dists = [dists]

        trajs = []
        for d, j in zip(dists, indices):
            if j == robot_idx or j >= self.n_robots or d > COMM_RADIUS:
                continue
            if self._predicted_trajs[j] is not None:
                trajs.append(self._predicted_trajs[j])
            else:
                nb = self.robots[j]
                state4 = np.array([nb.pos[0], nb.pos[1], nb.velocity[0], nb.velocity[1]])
                trajs.append(state4)
        return trajs

    def step(self):
        if self.field is not None and self.behavior_mode != SwarmBehavior.ORBIT:
            self.field.step(DT)
        if self.field is not None and self.behavior_mode == SwarmBehavior.ORBIT:
            self.behavior_engine.orbit_center = self.field.brightest_source_pos()

        if self.obstacle_manager is not None:
            self.obstacle_manager.step(DT)


        self.kdtree = KDTree(np.array([r.pos for r in self.robots]))

        all_nb_trajs = []
        for i in range(self.n_robots):
            all_nb_trajs.append(self._get_neighbour_trajs(i))

        controls = []
        new_predicted_trajs = []

        import expert_src.controller as ctrl_module
        original_max_cons = ctrl_module.MAX_CONSENSUS_ITER
        ctrl_module.MAX_CONSENSUS_ITER = self.max_consensus

        for i, (robot, ctrl) in enumerate(zip(self.robots, self.controllers)):
            goal_pos, ref_theta, ref_vel = self._goal_from_behavior_with_collision(i)
            nb_trajs = all_nb_trajs[i]

            t0 = time.perf_counter()
            u = ctrl.compute_control(
                robot, goal_pos, ref_theta=ref_theta, ref_vel=ref_vel, ref_omega=0.0,
                neighbour_trajs=nb_trajs,
                obstacle_manager=self.obstacle_manager,
                robot_name=f"R{i}"
            )
            self.solve_times[i].append((time.perf_counter() - t0) * 1e6)

            controls.append(u)
            new_predicted_trajs.append(ctrl.get_predicted_trajectory())
            self.solver_statuses[i] = ctrl.last_solver_status
            if self.solver_statuses[i] != 0:
                self.failure_counts[i] += 1

        ctrl_module.MAX_CONSENSUS_ITER = original_max_cons

        for i in range(self.n_robots):
            self._predicted_trajs[i] = new_predicted_trajs[i]

        for i, (robot, ctrl, u) in enumerate(zip(self.robots, self.controllers, controls)):
            ctrl.apply_control(robot, u)
            self.histories[i].append(robot.pos.copy())
            self.v_log[i].append(robot.velocity.copy())
            self.heading_log[i].append(robot.theta)
            self.w_log[i].append(float(robot.ang_velocity))

        self._resolve_collisions()

        positions = np.array([r.pos for r in self.robots])

        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        np.fill_diagonal(dists, np.inf)

        per_robot_min = np.min(dists, axis=1)

        current_global_min = np.min(per_robot_min)
        current_avg_min_sep = np.mean(per_robot_min)

        mask = np.triu(np.ones((self.n_robots, self.n_robots), dtype=bool), k=1)
        pairwise_mean = np.mean(dists[mask])

        self.min_dist_log.append(current_global_min)
        self.avg_min_sep_log.append(current_avg_min_sep)
        self.mean_dist_log.append(pairwise_mean)

        vels = np.array([r.velocity for r in self.robots])
        speeds = np.linalg.norm(vels, axis=1, keepdims=True)
        speeds[speeds < 1e-6] = 1.0
        unit_vels = vels / speeds
        order_param = np.linalg.norm(np.mean(unit_vels, axis=0))
        self.coherence_log.append(order_param)

        self.current_time += DT
        self.time_history.append(self.current_time)

    def run(self, max_steps=MAX_STEPS):
        for _ in range(max_steps):
            self.step()

    def get_snapshot(self):
        h_grid, h_extent = self.field.heatmap(20) if self.field else (None, None)
        obs_snapshots = []
        if self.obstacle_manager is not None:
            for obs in self.obstacle_manager.active_obstacles:
                obs_snapshots.append({
                    "pos":    obs.pos.copy(),
                    "radius": obs.radius,
                    "vel":    obs.vel.copy() if hasattr(obs, "vel") else np.zeros(2),
                })

        positions = np.array([r.pos for r in self.robots])
        overlap_pairs = self._detect_overlap_pairs()
        robot_obstacle_overlaps = self._detect_robot_obstacle_overlaps()
        colliding_indices = (
            list(self.collision.active_set())
            if self.enable_collision and self.collision is not None
            else []
        )
        flash_alphas = [
            self.collision.flash_alpha(i) if self.collision is not None else 0.0
            for i in range(self.n_robots)
        ]
        in_violation = set()
        for ii in range(self.n_robots):
            for jj in range(ii + 1, self.n_robots):
                if np.linalg.norm(positions[ii] - positions[jj]) < D_SAFE:
                    in_violation.add(ii)
                    in_violation.add(jj)

        return {
            'time': self.current_time,
            'positions': positions,
            'headings': np.array([r.theta for r in self.robots]),
            'field_sources': [s.pos.copy() for s in self.field.sources] if self.field else None,
            'heatmap': h_grid,
            'extent': h_extent,
            'obstacles': obs_snapshots,
            'overlap_pairs': overlap_pairs,
            'robot_obstacle_overlaps': robot_obstacle_overlaps,
            'colliding_indices': colliding_indices,
            'flash_alphas': flash_alphas,
            'in_violation': list(in_violation),
            'n_active_overlaps': len(overlap_pairs),
            'total_collisions': self.total_collisions,
        }
