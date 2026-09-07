"""Reynolds flocking rules and the behaviors that set each robot's reference."""


from __future__ import annotations
from enum import Enum, auto
import numpy as np


V_TARGET        = 0.10
LOOKAHEAD_DIST  = 0.07
EPSILON_GRAD    = 1e-6




class SwarmBehavior(Enum):
    """High-level collective behavior mode."""
    PHOTOTAXIS = "phototaxis"
    PHOTOTAXIS_NO_REYNOLDS = "phototaxis_no_reynolds"
    NO_LIGHT = "no_light"
    ORBIT      = "orbit"
    PHOTOTAXIS_ORBITAL = "phototaxis_orbital"
    PHOTOTAXIS_ORBITAL_CONTRACTING = "phototaxis_orbital_contracting"





class ReynoldsBehavior:
    """
    Compute the per-robot MPC goal reference for the chosen SwarmBehavior.

    The output (goal_pos, ref_theta, ref_vel) feeds straight into
    `CasadiAcadosEllipticalController.compute_control()`.

    Parameters
    ----------
    arena_half    : half-size of the square arena [m]
    orbit_center  : (2,) orbit center for ORBIT mode [m]
    orbit_radius  : orbit radius for ORBIT mode [m]
    orbit_speed   : signed angular speed [rad/s]; positive = CCW
    lookahead     : look-ahead distance for goal projection [m]
    v_target      : reference linear speed [m/s]
    """

    def __init__(
        self,
        arena_half:    float       = 0.4,
        orbit_center:  np.ndarray = None,
        orbit_radius:  float       = 0.20,
        orbit_speed:   float       = 0.5,
        lookahead:     float       = LOOKAHEAD_DIST,
        v_target:      float       = V_TARGET,
    ):
        self.arena_half   = float(arena_half)
        self.orbit_center = np.zeros(2) if orbit_center is None else np.asarray(orbit_center, dtype=float)
        self.orbit_radius = float(orbit_radius)
        self.orbit_speed  = float(orbit_speed)
        self.lookahead    = float(lookahead)
        self.v_target     = float(v_target)

    def compute_goal(
        self,
        robot,
        neighbors:  list,
        behavior:   SwarmBehavior,
        field=None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Return (goal_pos, ref_theta, ref_vel) for the robot under the given mode.

        Parameters
        ----------
        robot     : EllipticalBot2D, the ego robot
        neighbors : list of EllipticalBot2D, the other robots (or a subset)
        behavior  : SwarmBehavior mode
        field     : LightField (needed by the phototaxis-based modes)

        Returns
        -------
        goal_pos  : (2,) ndarray, look-ahead target position
        ref_theta : float, reference heading [rad]
        ref_vel   : (2,) ndarray, reference velocity vector
        """
        if behavior == SwarmBehavior.PHOTOTAXIS:
            return self._phototaxis(robot, field)

        elif behavior == SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS:
            return self._phototaxis_no_reynolds(robot, field)

        elif behavior == SwarmBehavior.NO_LIGHT:
            return self._no_light(robot)

        elif behavior == SwarmBehavior.ORBIT:
            return self._orbit(robot)

        elif behavior == SwarmBehavior.PHOTOTAXIS_ORBITAL:
            return self._phototaxis_orbital(robot, field)
        
        elif behavior == SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING:
            return self._phototaxis_orbital_contracting(robot, field)


        else:
            raise ValueError(f"Unknown SwarmBehavior: {behavior}")


    def _phototaxis(self, robot, field=None) -> tuple:
        pos = robot.pos

        if field is None or not hasattr(field, 'sources') or not field.sources:
            ref_theta = float(robot.theta + np.random.uniform(-0.15, 0.15))
            goal_dir = np.array([np.cos(ref_theta), np.sin(ref_theta)])
            goal_pos = np.clip(
                pos + self.lookahead * goal_dir,
                -self.arena_half, self.arena_half,
            )
            return goal_pos, ref_theta, self.v_target * goal_dir

        nearest_src = min(field.sources, key=lambda s: np.linalg.norm(pos - s.pos))
        dist_to_src = np.linalg.norm(pos - nearest_src.pos)

        if dist_to_src < self.lookahead:
            goal_pos  = nearest_src.pos.copy()
            diff      = goal_pos - pos
            ref_theta = float(np.arctan2(diff[1], diff[0]))
            ref_vel   = self.v_target * diff / max(1e-6, dist_to_src)
            return goal_pos, ref_theta, ref_vel

        grad  = field.gradient(pos)
        gnorm = np.linalg.norm(grad)

        if gnorm < EPSILON_GRAD:
            ref_theta = float(robot.theta)
        else:
            ref_theta = float(np.arctan2(grad[1], grad[0]))

        goal_dir = np.array([np.cos(ref_theta), np.sin(ref_theta)])
        goal_pos = np.clip(
            pos + self.lookahead * goal_dir,
            -self.arena_half, self.arena_half,
        )
        return goal_pos, ref_theta, self.v_target * goal_dir


    def _phototaxis_no_reynolds(self, robot, field=None) -> tuple:
        """
        Same light-field goal as PHOTOTAXIS. Pair with MPC weights
        q_cohesion=0, q_align=0, w_soft=0, w_soft_obs=0 so robots do not flock.
        """
        return self._phototaxis(robot, field)


    def _no_light(self, robot) -> tuple:
        """
        Random-walk goal with the light field off (persistence / no-attractor test).
        Ignores any field passed in. Pair with field=None in the sim and the
        default Reynolds MPC weights (q_cohesion, q_align, w_soft).
        """
        return self._phototaxis(robot, field=None)


    def _phototaxis_orbital(self, robot, field) -> tuple:
        if field is None:
            return self._phototaxis(robot, field)

        pos = robot.pos
        grad = field.gradient(pos)
        gnorm = np.linalg.norm(grad)

        if gnorm < EPSILON_GRAD:
            ref_theta = float(robot.theta)
        else:
            grad_theta = np.arctan2(grad[1], grad[0])

            sign = np.sign(self.orbit_speed) if self.orbit_speed != 0.0 else 1.0

            ref_theta = float(grad_theta - sign * np.pi / 2.0)

        goal_dir = np.array([np.cos(ref_theta), np.sin(ref_theta)])
        goal_pos = np.clip(
            pos + self.lookahead * goal_dir,
            -self.arena_half, self.arena_half,
        )
        return goal_pos, ref_theta, self.v_target * goal_dir


    def _phototaxis_orbital_contracting(self, robot, field) -> tuple:
        if field is None:
            return self._phototaxis(robot, field)

        pos = robot.pos
        grad = field.gradient(pos)
        gnorm = np.linalg.norm(grad)

        if gnorm < EPSILON_GRAD:
            ref_theta = float(robot.theta)
        else:
            grad_theta = np.arctan2(grad[1], grad[0])
            sign = np.sign(self.orbit_speed) if self.orbit_speed != 0.0 else 1.0

            contraction_angle = np.deg2rad(6.0)
            ref_theta = float(grad_theta - sign * (np.pi / 2.0 - contraction_angle))

        goal_dir = np.array([np.cos(ref_theta), np.sin(ref_theta)])
        goal_pos = np.clip(
            pos + self.lookahead * goal_dir,
            -self.arena_half, self.arena_half,
        )
        return goal_pos, ref_theta, self.v_target * goal_dir





    def _orbit(self, robot) -> tuple:
        pos    = robot.pos
        center = self.orbit_center
        radius = self.orbit_radius

        diff  = pos - center
        angle = float(np.arctan2(diff[1], diff[0]))

        arc   = self.lookahead / max(radius, 0.01)
        sign  = np.sign(self.orbit_speed) if self.orbit_speed != 0.0 else 1.0
        target_angle = angle + sign * arc

        orbit_point = center + radius * np.array([
            np.cos(target_angle),
            np.sin(target_angle),
        ])
        goal_pos = np.clip(orbit_point, -self.arena_half, self.arena_half)

        tangent_theta = angle + sign * np.pi / 2.0
        ref_dir   = np.array([np.cos(tangent_theta), np.sin(tangent_theta)])
        ref_theta = float(tangent_theta)

        return goal_pos, ref_theta, self.v_target * ref_dir

