"""Resolves overlapping robots so contacts are physical rather than ignored."""

from __future__ import annotations
import numpy as np
from typing import Set, Tuple

__all__ = ["CollisionHandler"]


STEER_DECAY   = 0.85
MAX_BIAS      = np.pi / 2
MAX_BIAS_TOTAL = np.pi * 0.75
PUSH_CORRECT  = 1.10
FLASH_FRAMES  = 12

CONTACT_SKIN      = 0.001
MAX_RESOLVE_ITERS = 4
RESOLVE_TOL       = 2e-4



def _rotate_to_local(vec: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([ c * vec[0] + s * vec[1],
                     -s * vec[0] + c * vec[1]])


def _ellipse_contact_radius(normal_world: np.ndarray,
                             theta: float, a: float, b: float) -> float:
    n_local = _rotate_to_local(normal_world, theta)
    denom   = np.sqrt((n_local[0] / a) ** 2 + (n_local[1] / b) ** 2)
    return 1.0 / denom if denom > 1e-12 else a


def _ellipse_overlap_fast(
    pos_i: np.ndarray, theta_i: float, a_i: float, b_i: float,
    pos_j: np.ndarray, theta_j: float, a_j: float, b_j: float,
) -> Tuple[bool, float, np.ndarray]:
    """
    Returns (overlapping, penetration_depth, normal_i_to_j).
    normal_i_to_j is the unit vector FROM robot_i TOWARD robot_j.
    """
    delta = pos_j - pos_i
    dist  = float(np.linalg.norm(delta))

    if dist < 1e-9:
        return True, float(a_i + a_j), np.array([1.0, 0.0])

    normal = delta / dist
    ri = _ellipse_contact_radius(normal, theta_i, a_i, b_i)
    rj = _ellipse_contact_radius(normal, theta_j, a_j, b_j)
    pen = (ri + rj) - dist
    return pen > 0.0, float(pen), normal


def _steer_bias(heading: float, normal_i_to_j: np.ndarray,
                depth_factor: float) -> float:
    """
    Compute the heading bias for robot_i to steer AWAY from robot_j.

    heading         : current robot_i heading [rad]
    normal_i_to_j   : unit vector pointing toward robot_j
    depth_factor    : penetration / reference_size, clamped [0, 1]

    Returns delta_theta [rad] to ADD to robot_i's ref_theta.
    """
    h_vec    = np.array([np.cos(heading), np.sin(heading)])
    approach = float(np.dot(h_vec, normal_i_to_j))

    if approach <= 0.0:
        return 0.0

    t_ccw = np.array([-normal_i_to_j[1],  normal_i_to_j[0]])
    t_cw  = np.array([ normal_i_to_j[1], -normal_i_to_j[0]])

    t = t_ccw if np.dot(h_vec, t_ccw) >= np.dot(h_vec, t_cw) else t_cw

    target  = float(np.arctan2(t[1], t[0]))
    diff    = float(np.arctan2(np.sin(target - heading),
                               np.cos(target - heading)))

    bias = diff * approach * (0.5 + 0.5 * depth_factor)
    return float(np.clip(bias, -MAX_BIAS, MAX_BIAS))


def _ellipse_wall_overlaps(
    pos: np.ndarray,
    theta: float,
    a: float,
    b: float,
    arena_half: float,
) -> list[tuple[float, np.ndarray]]:
    """
    Return (penetration, normal_i_to_wall) for each arena wall the ellipse hits.

    ``normal`` points from the robot centre toward the wall (same convention as
    robot-obstacle contact in ``_ellipse_overlap_fast``).
    """
    hits: list[tuple[float, np.ndarray]] = []
    px, py = float(pos[0]), float(pos[1])
    for normal, signed_coord in (
        (np.array([1.0, 0.0]), px),
        (np.array([-1.0, 0.0]), -px),
        (np.array([0.0, 1.0]), py),
        (np.array([0.0, -1.0]), -py),
    ):
        r_contact = _ellipse_contact_radius(normal, theta, a, b)
        pen = signed_coord + r_contact - arena_half
        if pen > 0.0:
            hits.append((float(pen), normal))
    return hits



class CollisionHandler:
    """
    Detects ellipse-ellipse overlaps and injects per-robot heading biases
    so the MPC steers robots away from each other naturally.

    Parameters
    ----------
    robots       : list[EllipticalBot2D]
    steer_decay  : per-step decay of theta_bias (0 = instant, 1 = permanent)
    flash_frames : visual flash duration in animation frames
    """

    def __init__(self,
                 robots,
                 steer_decay:       float = STEER_DECAY,
                 flash_frames:      int   = FLASH_FRAMES,
                 contact_skin:      float = CONTACT_SKIN,
                 max_resolve_iters: int   = MAX_RESOLVE_ITERS,
                 resolve_tol:       float = RESOLVE_TOL):

        self.robots            = robots
        self.steer_decay       = steer_decay
        self.flash_frames      = flash_frames
        self.contact_skin      = contact_skin
        self.max_resolve_iters = max_resolve_iters
        self.resolve_tol       = resolve_tol

        n = len(robots)
        self._theta_bias:  list[float]    = [0.0] * n
        self._flash_timers: dict[int, int] = {}

    def step(self, obstacle_manager=None, arena_half: float | None = None) -> None:
        """
        Call ONCE per simulation step, AFTER all MPC apply_control() calls.
        Resolves overlaps and updates theta_bias for each robot.
        Supports optional obstacle_manager to handle robot-obstacle collisions
        and optional arena_half for square arena wall contact.
        """
        n = len(self.robots)

        for i in range(n):
            self._theta_bias[i] *= self.steer_decay

        max_pen = self._resolve(n, obstacle_manager, arena_half, do_bias=True)

        it = 1
        while max_pen >= self.resolve_tol and it < self.max_resolve_iters:
            max_pen = self._resolve(n, obstacle_manager, arena_half, do_bias=False)
            it += 1

        done = [k for k, v in self._flash_timers.items() if v <= 1]
        for k in done:
            del self._flash_timers[k]
        for k in list(self._flash_timers):
            if k not in done:
                self._flash_timers[k] -= 1

    def _resolve(self, n: int, obstacle_manager, arena_half, do_bias: bool) -> float:
        """
        One Gauss-Seidel positional-separation sweep over every contact type
        (robot-robot, robot-obstacle, arena wall). Returns the largest
        penetration depth [m] found this sweep, so the caller can stop once
        overlaps have cleared.

        When ``do_bias`` is True (first sweep of the step) it also accumulates
        the per-robot heading bias and sets the flash timers; later sweeps
        only move robots apart, so the bias is computed once from the impact
        geometry and is not re-injected on every relaxation pass.
        """
        max_pen = 0.0
        skin    = self.contact_skin

        for i in range(n):
            ri = self.robots[i]
            for j in range(i + 1, n):
                rj = self.robots[j]

                dx = float(ri.pos[0] - rj.pos[0])
                dy = float(ri.pos[1] - rj.pos[1])
                reach = ri.bot_width * 0.5 + rj.bot_width * 0.5
                if dx * dx + dy * dy > reach * reach:
                    continue

                overlapping, pen, normal = _ellipse_overlap_fast(
                    ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                    rj.pos, float(rj.theta), rj.bot_width * 0.5, rj.bot_height * 0.5,
                )

                if not overlapping or pen <= 0.0:
                    continue
                if pen > max_pen:
                    max_pen = pen

                sep = (pen + skin) * 0.5 * PUSH_CORRECT
                ri.translate(-sep * normal)
                rj.translate( sep * normal)

                if not do_bias:
                    continue

                ref_size     = ri.bot_width * 0.5 + rj.bot_width * 0.5
                depth_factor = float(np.clip(pen / ref_size, 0.0, 1.0))

                bias_i = _steer_bias(float(ri.theta), normal, depth_factor)
                bias_j = _steer_bias(float(rj.theta), -normal, depth_factor)

                self._theta_bias[i] = float(np.clip(
                    self._theta_bias[i] + bias_i,
                    -MAX_BIAS_TOTAL, MAX_BIAS_TOTAL,
                ))
                self._theta_bias[j] = float(np.clip(
                    self._theta_bias[j] + bias_j,
                    -MAX_BIAS_TOTAL, MAX_BIAS_TOTAL,
                ))

                self._flash_timers[i] = self.flash_frames
                self._flash_timers[j] = self.flash_frames

        if obstacle_manager is not None:
            active_obs = obstacle_manager.active_obstacles
            for i in range(n):
                ri = self.robots[i]
                for obs in active_obs:
                    overlapping, pen, normal = _ellipse_overlap_fast(
                        ri.pos, float(ri.theta), ri.bot_width * 0.5, ri.bot_height * 0.5,
                        obs.pos, 0.0, obs.radius, obs.radius,
                    )

                    if not overlapping or pen <= 0.0:
                        continue
                    if pen > max_pen:
                        max_pen = pen

                    ri.translate(-pen * PUSH_CORRECT * normal)

                    if not do_bias:
                        continue

                    ref_size = ri.bot_width * 0.5 + obs.radius
                    depth_factor = float(np.clip(pen / ref_size, 0.0, 1.0))
                    bias_i = _steer_bias(float(ri.theta), normal, depth_factor)

                    self._theta_bias[i] = float(np.clip(
                        self._theta_bias[i] + bias_i,
                        -MAX_BIAS_TOTAL, MAX_BIAS_TOTAL,
                    ))

                    self._flash_timers[i] = self.flash_frames

        if arena_half is not None and arena_half > 0.0:
            for i in range(n):
                ri = self.robots[i]
                a = ri.bot_width * 0.5
                b = ri.bot_height * 0.5
                for pen, normal in _ellipse_wall_overlaps(
                    ri.pos, float(ri.theta), a, b, arena_half,
                ):
                    if pen > max_pen:
                        max_pen = pen
                    ri.translate(-pen * PUSH_CORRECT * normal)

                    if not do_bias:
                        continue

                    ref_size = a
                    depth_factor = float(np.clip(pen / ref_size, 0.0, 1.0))
                    bias_i = _steer_bias(float(ri.theta), normal, depth_factor)

                    self._theta_bias[i] = float(np.clip(
                        self._theta_bias[i] + bias_i,
                        -MAX_BIAS_TOTAL, MAX_BIAS_TOTAL,
                    ))
                    self._flash_timers[i] = self.flash_frames

        return max_pen


    def theta_bias(self, robot_index: int) -> float:
        """
        Heading correction [rad] to ADD to ref_theta in _goal_from_behavior.
        Positive = turn CCW, negative = turn CW.
        """
        return self._theta_bias[robot_index]

    def active_set(self) -> Set[int]:
        """Return the set of robot indices currently flashing."""
        return set(self._flash_timers.keys())

    def flash_alpha(self, robot_index: int) -> float:
        """[0, 1] flash intensity: 1.0 at impact, decays to 0 over flash_frames."""
        frames_left = self._flash_timers.get(robot_index, 0)
        return frames_left / self.flash_frames if self.flash_frames > 0 else 0.0