"""Wraps the compiled acados solver and turns neighbour state into a control input."""

import os
import json
import numpy as np
import yaml
from acados_template import AcadosOcpSolver

_controller_path = os.path.dirname(os.path.abspath(__file__))
_config_path     = os.path.join(_controller_path, "config", "solver_ellipse_mpc.yaml")
with open(_config_path) as _f:
    _cfg = yaml.safe_load(_f)
_reynolds = _cfg.get("reynolds", {})
Q_COHESION_DEFAULT = float(_reynolds.get("q_cohesion", 5.0))
Q_ALIGN_DEFAULT    = float(_reynolds.get("q_align", 10.0))
_model_bounds      = _cfg.get("model_bounds", {})
V_MIN              = float(_model_bounds.get("v_min", 0.0))

MAX_NEIGHBOURS     = 5
MAX_OBS            = 5
D_SAFE             = 0.07
D_SAFE_OBS         = 0.10
_DUMMY_FAR         = 10.0
_communication     = _cfg.get("communication", {})
MAX_CONSENSUS_ITER = int(_communication.get("max_consensus", 2))
NB_STRIDE          = 5
OBS_STRIDE         = 3
NP_BASE            = 12


class CasadiAcadosEllipticalController:

    def __init__(
        self,
        dt: float = 0.1,
        N:  int   = 15,
        use_noisy_observations: bool  = False,
        obs_noise_std_pos:      float = 0.0005,
        obs_noise_std_vel:      float = 0.001,
        obs_noise_std_theta:    float = 0.0005,
        use_process_noise:      bool  = False,
        process_noise_std:      float = 0.0005,
        q_cohesion:             float = Q_COHESION_DEFAULT,
        q_align:                float = Q_ALIGN_DEFAULT,
    ):
        self.dt = dt
        self.N  = N

        self.use_noisy_observations = use_noisy_observations
        self.obs_noise_std_pos      = obs_noise_std_pos
        self.obs_noise_std_vel      = obs_noise_std_vel
        self.obs_noise_std_theta    = obs_noise_std_theta
        self.use_process_noise      = use_process_noise
        self.process_noise_std      = process_noise_std
        self._last_noisy_state      = None

        self.min_linear_vel  = V_MIN
        self.max_linear_vel  = 0.115
        self.min_angular_vel = -3.463
        self.max_angular_vel = 1.432

        self.q_pos   = 60.0
        self.q_theta = 80.0
        self.q_vel   = 1.5
        self.q_om    = 1.5
        self.r_u     = 0.01

        self.q_cohesion = float(q_cohesion)
        self.q_align    = float(q_align)

        json_path = os.path.join(
            os.path.dirname(__file__), "lib",
            "acados_ellipse_tracking_mpc_solver_config.json",
        )
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Acados solver JSON not found:\n  {json_path}\n"
                "Run generate_solver_via_acados.py first."
            )

        with open(json_path) as f:
            cfg = json.load(f)

        try:
            self.N  = int(cfg["solver_options"].get("N_horizon", self.N))
            self.dt = float(cfg["solver_options"].get("Tsim",     self.dt))
        except Exception:
            pass

        tc = cfg.get("trajectory_coupled_dmpc", {})
        self._max_nb        = int(tc.get("max_neighbours",  MAX_NEIGHBOURS))
        self._max_obs       = int(tc.get("max_obstacles",   MAX_OBS))
        self._d_safe        = float(tc.get("d_safe",        D_SAFE))
        self._d_safe_obs    = float(tc.get("d_safe_obs",    D_SAFE_OBS))
        self._np_total      = int(tc.get("np_total",        NP_BASE + NB_STRIDE * MAX_NEIGHBOURS + OBS_STRIDE * MAX_OBS))
        self._padding       = float(tc.get("padding_position", _DUMMY_FAR))
        self._nb_stride     = int(tc.get("nb_stride",       NB_STRIDE))
        self._obs_stride    = int(tc.get("obs_stride",      OBS_STRIDE))
        self._np_base       = int(tc.get("np_base",         NP_BASE))
        self._idx_nb_start  = int(tc.get("idx_nb_start",    NP_BASE))
        self._idx_obs_start = int(tc.get("idx_obs_start",
                                           NP_BASE + NB_STRIDE * MAX_NEIGHBOURS))

        self.solver = AcadosOcpSolver(
            None, json_file=json_path, generate=False, build=False
        )

        self.nx = 3
        self.nu = 2

        self._predicted_traj: np.ndarray | None = None
        self.last_solver_status = 0


    def get_predicted_trajectory(self) -> np.ndarray | None:
        """
        Return the most recently computed predicted trajectory.

        Returns
        -------
        traj : ndarray, shape (N+1, 4)  or  None before first solve.
            traj[k] = [px, py, vx, vy] predicted at horizon stage k.
            vx, vy derived from v_cmd * [cos(th), sin(th)] predictions.
        """
        return self._predicted_traj

    def compute_control(
        self,
        robot,
        goal:               np.ndarray,
        ref_theta:          float | None        = None,
        ref_vel:            np.ndarray | None   = None,
        ref_omega:          float | None        = None,
        neighbour_trajs:    list | None         = None,
        obstacle_manager                        = None,
        robot_name:         str                 = "?",
    ) -> np.ndarray:
        """
        Compute MPC control via trajectory-coupled consensus iterations.

        Parameters
        ----------
        robot           : EllipticalBot2D
        goal            : (2,) target position
        ref_theta       : reference heading [rad] (computed from goal if None)
        ref_vel         : (2,) reference velocity vector (optional)
        ref_omega       : reference angular rate (optional, default 0)
        neighbour_trajs : list of ndarray, each shape (N+1, 4) = [px,py,vx,vy]
            or (N+1, 2) = [px,py] (backward-compatible, vx/vy will be 0)
            or (4,) / (2,) broadcast over all stages.
            Pass None or [] for no neighbours.
        obstacle_manager : ObstacleManager, optional circular obstacles
        robot_name      : for debug printing

        Returns
        -------
        u : (2,)  [v_cmd, omega_cmd]
        """
        x0 = self._robot_to_state(robot)

        if ref_theta is None:
            delta = goal - robot.pos
            ref_theta = float(np.arctan2(delta[1], delta[0]))

        v_ref  = float(np.linalg.norm(ref_vel)) if ref_vel is not None else 0.0
        om_ref = float(ref_omega) if ref_omega is not None else 0.0

        nb_trajs_norm = self._normalise_neighbour_trajs(neighbour_trajs)

        for k in range(self.N + 1):
            self.solver.set(k, "x", x0)
        for k in range(self.N):
            self.solver.set(k, "u", np.array([self.min_linear_vel, 0.0]))

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        status = 0
        for _citer in range(MAX_CONSENSUS_ITER):

            for k in range(self.N + 1):
                p_k = self._build_stage_params(
                    goal, ref_theta, v_ref, om_ref, nb_trajs_norm,
                    obstacle_manager=obstacle_manager,
                    robot_pos=robot.pos,
                    stage=k
                )
                self.solver.set(k, "p", p_k)

            status = self.solver.solve()

            traj = np.zeros((self.N + 1, 4))
            for k in range(self.N + 1):
                state_k = self.solver.get(k, "x")
                traj[k, 0] = state_k[0]
                traj[k, 1] = state_k[1]
                if k < self.N:
                    u_k = self.solver.get(k, "u")
                    traj[k, 2] = u_k[0] * np.cos(state_k[2])
                    traj[k, 3] = u_k[0] * np.sin(state_k[2])
                else:
                    traj[k, 2] = traj[k - 1, 2]
                    traj[k, 3] = traj[k - 1, 3]

            self._predicted_traj = traj

        self.last_solver_status = status
        if status != 0:
            print(f"[{robot_name}] Solver status={status} - using fallback.")
            u = self._fallback_control(robot, goal, neighbour_trajs, obstacle_manager=obstacle_manager)
            self._predicted_traj = self._generate_fallback_trajectory(robot, u)
        else:
            u = self.solver.get(0, "u").copy()

        u[0] = float(np.clip(u[0], -self.max_linear_vel,  self.max_linear_vel))
        u[1] = float(np.clip(u[1], -self.max_angular_vel, self.max_angular_vel))
        return u

    def apply_control(self, robot, control: np.ndarray):
        """Apply [v_cmd, omega_cmd] to robot for one timestep."""
        v_cmd  = float(control[0])
        om_cmd = float(control[1])

        theta_curr = (
            float(self._last_noisy_state["theta"])
            if self._last_noisy_state is not None
            else float(robot.theta)
        )

        robot.velocity     = v_cmd * np.array([np.cos(theta_curr),
                                               np.sin(theta_curr)])
        robot.ang_velocity = om_cmd
        robot.move_bot(self.dt)

        if self.use_process_noise:
            noise = np.random.normal(0, self.process_noise_std, 2)
            robot.translate(noise)


    def _robot_to_state(self, robot) -> np.ndarray:
        """Extract (possibly noisy) state vector [px, py, theta]."""
        if not self.use_noisy_observations:
            self._last_noisy_state = {
                "pos":          robot.pos.copy(),
                "vel":          robot.velocity.copy(),
                "theta":        robot.theta,
                "ang_velocity": robot.ang_velocity,
            }
            return np.array([float(robot.pos[0]),
                             float(robot.pos[1]),
                             float(robot.theta)])

        pos_noise   = np.random.normal(0, self.obs_noise_std_pos,   2)
        theta_noise = np.random.normal(0, self.obs_noise_std_theta)
        vel_noise   = np.random.normal(0, self.obs_noise_std_vel,   2)

        noisy_pos   = robot.pos    + pos_noise
        noisy_theta = robot.theta  + theta_noise
        noisy_vel   = robot.velocity + vel_noise

        self._last_noisy_state = {
            "pos":          noisy_pos.copy(),
            "vel":          noisy_vel.copy(),
            "theta":        noisy_theta,
            "ang_velocity": robot.ang_velocity,
        }
        return np.array([float(noisy_pos[0]),
                         float(noisy_pos[1]),
                         float(noisy_theta)])

    def _normalise_neighbour_trajs(
        self,
        neighbour_trajs: list | None,
    ) -> list[np.ndarray]:
        """
        Convert raw neighbour_trajs input to a list of (N+1, 4) arrays.

        Accepts:
          - None / []                → empty list
          - list of (2,) arrays      → broadcast px,py to all stages; vx,vy=0
          - list of (4,) arrays      → broadcast px,py,vx,vy to all stages
          - list of (N+1, 2) arrays  → px,py used; vx,vy set to 0
          - list of (N+1, 4) arrays  → used as-is (full trajectory coupling)
        """
        if not neighbour_trajs:
            return []

        normalised = []
        for traj in neighbour_trajs[:self._max_nb]:
            arr = np.asarray(traj, dtype=float)

            if arr.ndim == 1:
                full = np.zeros((self.N + 1, 4))
                full[:, :min(len(arr), 4)] = arr[:min(len(arr), 4)]
                normalised.append(full)

            elif arr.ndim == 2:
                ncols = arr.shape[1]
                if arr.shape[0] >= self.N + 1 and ncols >= 2:
                    full = np.zeros((self.N + 1, 4))
                    full[:, :min(ncols, 4)] = arr[:self.N + 1, :min(ncols, 4)]
                    normalised.append(full)
                else:
                    normalised.append(np.zeros((self.N + 1, 4)))
            else:
                normalised.append(np.zeros((self.N + 1, 4)))

        return normalised

    def _build_stage_params(
        self,
        goal:             np.ndarray,
        ref_theta:        float,
        v_ref:            float,
        om_ref:           float,
        nb_trajs:         list[np.ndarray],
        obstacle_manager = None,
        robot_pos        = None,
        stage:            int  = 0,
    ) -> np.ndarray:
        """
        Build parameter vector p (12 + 5*K + 3*M) for a single horizon stage k.

        Layout:
          [0..9]       tracking params  (stage-invariant)
          [10]         q_cohesion       (Reynolds, stage-invariant)
          [11]         q_align          (Reynolds, stage-invariant)
          [12 + 5*j]  nb_j_px at stage k
          [12 + 5*j+1] nb_j_py at stage k
          [12 + 5*j+2] nb_j_vx at stage k
          [12 + 5*j+3] nb_j_vy at stage k
          [12 + 5*j+4] nb_j_active (1.0 if real, 0.0 if padded)
          [idx_obs_start + 3*m] obs_m_px
          [idx_obs_start + 3*m + 1] obs_m_py
          [idx_obs_start + 3*m + 2] obs_m_active (1.0 if active, 0.0 if padded)
        """
        p = np.empty(self._np_total)

        p[0] = float(goal[0])
        p[1] = float(goal[1])
        p[2] = ref_theta
        p[3] = v_ref
        p[4] = om_ref
        p[5] = self.q_pos
        p[6] = self.q_theta
        p[7] = self.q_vel
        p[8] = self.q_om
        p[9] = self.r_u

        p[10] = self.q_cohesion
        p[11] = self.q_align

        for j in range(self._max_nb):
            base = self._idx_nb_start + self._nb_stride * j
            if j < len(nb_trajs):
                nb = nb_trajs[j][stage]
                p[base + 0] = nb[0]
                p[base + 1] = nb[1]
                p[base + 2] = nb[2]
                p[base + 3] = nb[3]
                p[base + 4] = 1.0
            else:
                p[base + 0] = self._padding
                p[base + 1] = self._padding
                p[base + 2] = 0.0
                p[base + 3] = 0.0
                p[base + 4] = 0.0

        if obstacle_manager is not None and robot_pos is not None:
            obs_block = obstacle_manager.build_obs_param_block(
                robot_pos=robot_pos,
                stage=stage,
                dt=self.dt,
                padding_pos=self._padding,
            )
            p[self._idx_obs_start: self._idx_obs_start + len(obs_block)] = obs_block
        else:
            for m in range(self._max_obs):
                base = self._idx_obs_start + self._obs_stride * m
                p[base + 0] = self._padding
                p[base + 1] = self._padding
                p[base + 2] = 0.0

        return p

    def _fallback_control(
        self,
        robot,
        goal:            np.ndarray,
        neighbour_trajs: list | None,
        obstacle_manager = None,
    ) -> np.ndarray:
        """
        Simple goal-seeking + repulsion fallback when solver fails.
        """
        to_goal = goal - robot.pos
        dist    = np.linalg.norm(to_goal)
        v_goal  = (to_goal / dist) if dist > 1e-4 else np.zeros(2)

        v_rep = np.zeros(2)
        if neighbour_trajs:
            d_inf = max(self._d_safe * 3.0, 0.15)
            k_rep = 2.0
            for traj in neighbour_trajs:
                traj_arr = np.asarray(traj)
                nb_pos   = traj_arr[0, :2] if traj_arr.ndim == 2 else traj_arr[:2]
                diff     = robot.pos - nb_pos
                d        = np.linalg.norm(diff)
                if 1e-4 < d < d_inf:
                    v_rep += k_rep * (1.0 / d - 1.0 / d_inf) * (diff / d)

        v_obs_rep = np.zeros(2)
        if obstacle_manager is not None:
            d_inf_obs = max(self._d_safe_obs * 3.0, 0.20)
            k_obs_rep = 3.0
            for obs in obstacle_manager.get_nearest_k(robot.pos):
                diff = robot.pos - obs.pos
                d    = np.linalg.norm(diff)
                if 1e-4 < d < d_inf_obs:
                    v_obs_rep += k_obs_rep * (1.0 / d - 1.0 / d_inf_obs) * (diff / d)

        v_wall_rep  = np.zeros(2)
        arena_limit = 0.4
        d_wall_inf  = 0.10
        k_wall_rep  = 5.0

        for sign in [-1, 1]:
            dist_x = abs(robot.pos[0] - sign * arena_limit)
            if dist_x < d_wall_inf:
                v_wall_rep[0] += -sign * k_wall_rep * (1.0 / dist_x - 1.0 / d_wall_inf)
        for sign in [-1, 1]:
            dist_y = abs(robot.pos[1] - sign * arena_limit)
            if dist_y < d_wall_inf:
                v_wall_rep[1] += -sign * k_wall_rep * (1.0 / dist_y - 1.0 / d_wall_inf)

        v_total = v_goal + v_rep + v_obs_rep + v_wall_rep

        heading = np.array([np.cos(robot.theta), np.sin(robot.theta)])
        v_cmd   = float(np.clip(
            np.dot(v_total, heading), self.min_linear_vel, self.max_linear_vel
        ))

        desired_heading = (
            float(np.arctan2(v_total[1], v_total[0]))
            if np.linalg.norm(v_total) > 1e-4
            else robot.theta
        )
        ang_err = float(np.arctan2(
            np.sin(desired_heading - robot.theta),
            np.cos(desired_heading - robot.theta),
        ))
        om_cmd = float(np.clip(
            2.0 * ang_err, self.min_angular_vel, self.max_angular_vel
        ))

        return np.array([v_cmd, om_cmd])

    def _generate_fallback_trajectory(self, robot, control: np.ndarray) -> np.ndarray:
        """
        Produce a kinematic prediction based on fallback control.
        Returns (N+1, 4): [px, py, vx, vy].
        """
        v_cmd  = control[0]
        om_cmd = control[1]

        traj    = np.zeros((self.N + 1, 4))
        curr_p  = robot.pos.copy()
        curr_th = float(robot.theta)

        traj[0, :2] = curr_p
        traj[0, 2]  = v_cmd * np.cos(curr_th)
        traj[0, 3]  = v_cmd * np.sin(curr_th)

        for k in range(1, self.N + 1):
            curr_p  = curr_p + np.array([
                v_cmd * np.cos(curr_th),
                v_cmd * np.sin(curr_th)
            ]) * self.dt
            curr_th += om_cmd * self.dt
            traj[k, 0] = curr_p[0]
            traj[k, 1] = curr_p[1]
            traj[k, 2] = v_cmd * np.cos(curr_th)
            traj[k, 3] = v_cmd * np.sin(curr_th)

        return traj