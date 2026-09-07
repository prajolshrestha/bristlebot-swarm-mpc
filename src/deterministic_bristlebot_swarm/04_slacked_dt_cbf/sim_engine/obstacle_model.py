"""Static obstacles and the arena layouts used in the experiments."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple



class ObstacleBase:
    """
    Abstract circular obstacle.

    Attributes
    ----------
    pos    : np.ndarray (2,)   current centre position [m]
    vel    : np.ndarray (2,)   velocity (zero for static) [m/s]
    radius : float             obstacle radius [m]
    active : bool              if False, excluded from all queries
    """

    def __init__(self, pos: np.ndarray, radius: float = 0.05, active: bool = True):
        self.pos    = np.asarray(pos, dtype=float).copy()
        self.vel    = np.zeros(2, dtype=float)
        self.radius = float(radius)
        self.active = bool(active)

    def step(self, dt: float) -> None:
        """Advance obstacle state by dt seconds (subclass overrides for motion)."""
        pass

    def predict_pos(self, stage: int, dt: float) -> np.ndarray:
        """
        Return the predicted centre position at MPC stage `stage`.

        For static obstacles this always returns `self.pos`.
        For moving obstacles it returns `self.pos + stage * dt * self.vel`
        (constant-velocity prediction; no arena clipping - the penalty
        naturally drops to zero once the obstacle is far from the robot).

        Parameters
        ----------
        stage : int    MPC shooting stage k ∈ [0, N]
        dt    : float  MPC sampling time [s]
        """
        return self.pos + stage * dt * self.vel

    def is_active(self) -> bool:
        return self.active

    def __repr__(self) -> str:
        kind = type(self).__name__
        return (f"{kind}(pos={self.pos}, vel={self.vel}, "
                f"r={self.radius}, active={self.active})")



class StaticObstacle(ObstacleBase):
    """
    Circular obstacle at a fixed position.

    vel is always [0, 0] - `predict_pos` returns a constant `self.pos`.

    Parameters
    ----------
    pos    : array-like (2,)   [x, y] centre [m]
    radius : float             obstacle radius [m]  (default 0.05 m = 5 cm)
    active : bool              set False to temporarily disable without removing
    """

    def __init__(
        self,
        pos:    "array-like",
        radius: float = 0.05,
        active: bool  = True,
    ):
        super().__init__(pos=pos, radius=radius, active=active)



class MovingObstacle(ObstacleBase):
    """
    Circular obstacle moving at a constant velocity.

    Optionally bounces off arena walls (velocity component flipped when the
    obstacle centre reaches ±(arena_half - radius)).

    Parameters
    ----------
    pos        : array-like (2,)   initial centre [m]
    vel        : array-like (2,)   constant velocity [vx, vy] [m/s]
    radius     : float             obstacle radius [m]  (default 0.05 m)
    arena_half : float | None      half-width of the square arena [m].
                                   Set to None to disable wall bounce.
    bounce     : bool              if True, reflect velocity at arena walls
    active     : bool
    """

    def __init__(
        self,
        pos:        "array-like",
        vel:        "array-like" = (0.0, 0.0),
        radius:     float         = 0.05,
        arena_half: float | None  = 0.4,
        bounce:     bool          = True,
        active:     bool          = True,
    ):
        super().__init__(pos=pos, radius=radius, active=active)
        self.vel        = np.asarray(vel, dtype=float).copy()
        self.arena_half = float(arena_half) if arena_half is not None else None
        self.bounce     = bounce

    def step(self, dt: float) -> None:
        """Advance position by one timestep; reflect at arena walls if bounce=True."""
        if not self.active:
            return

        self.pos += self.vel * dt

        if self.bounce and self.arena_half is not None:
            limit = self.arena_half - self.radius
            for ax in range(2):
                if self.pos[ax] > limit:
                    self.pos[ax] = limit
                    self.vel[ax] = -abs(self.vel[ax])
                elif self.pos[ax] < -limit:
                    self.pos[ax] = -limit
                    self.vel[ax] = abs(self.vel[ax])

    def predict_pos(self, stage: int, dt: float) -> np.ndarray:
        """
        Constant-velocity prediction: p_k = pos + stage * dt * vel.

        Note: arena-clipping is intentionally NOT applied to the prediction -
        the NMPC receives the unclamped future positions, which is fine because
        the penalty vanishes when the obstacle is far from the robot.
        """
        return self.pos + stage * dt * self.vel



class ObstacleManager:
    """
    Container and query interface for all obstacles in the arena.

    Responsibilities
    ────────────────
    • Owns a list of ObstacleBase instances.
    • Advances all moving obstacles each timestep (`step`).
    • Queries the k nearest active obstacles to a robot position (`get_nearest_k`).
    • Predicts obstacle position at a given MPC stage (`predict_pos_at_stage`).
    • Builds the MPC parameter block for a single stage (`build_obs_param_block`).

    Parameters
    ----------
    dt         : float   simulation timestep [s]
    arena_half : float   half-width of arena [m] (passed to MovingObstacle)
    max_obs    : int     maximum obstacles handed to the NMPC (compiled constant)
    """

    OBS_STRIDE: int = 3

    def __init__(
        self,
        dt:         float = 0.1,
        arena_half: float = 0.4,
        max_obs:    int   = 5,
        obs_range:  float = 0.30,
    ):
        self.dt         = float(dt)
        self.arena_half = float(arena_half)
        self.max_obs    = int(max_obs)
        self.obs_range  = float(obs_range)

        self._obstacles: List[ObstacleBase] = []


    def add(self, obstacle: ObstacleBase) -> None:
        """Register an obstacle with the manager."""
        self._obstacles.append(obstacle)

    def remove_all(self) -> None:
        """Clear all obstacles."""
        self._obstacles.clear()

    @property
    def obstacles(self) -> List[ObstacleBase]:
        """Read-only view of the full obstacle list."""
        return list(self._obstacles)

    @property
    def active_obstacles(self) -> List[ObstacleBase]:
        """Only active obstacles."""
        return [o for o in self._obstacles if o.is_active()]


    def step(self, dt: float | None = None) -> None:
        """
        Advance all moving obstacles by `dt` seconds.

        Parameters
        ----------
        dt : float | None   If None, uses the manager's default `self.dt`.
        """
        _dt = dt if dt is not None else self.dt
        for obs in self._obstacles:
            obs.step(_dt)


    def get_nearest_k(
        self,
        robot_pos: np.ndarray,
        k:         int | None = None,
    ) -> List[ObstacleBase]:
        """
        Return up to k nearest *active* obstacles, sorted by distance
        (closest first).

        Parameters
        ----------
        robot_pos : (2,) position of the querying robot [m]
        k         : max number to return (defaults to self.max_obs)

        Returns
        -------
        List of ObstacleBase, length ≤ k
        """
        k = self.max_obs if k is None else int(k)
        active = [o for o in self.active_obstacles
                  if np.linalg.norm(robot_pos - o.pos) <= self.obs_range]
        if not active:
            return []

        dists   = [np.linalg.norm(robot_pos - o.pos) for o in active]
        ordered = sorted(zip(dists, active), key=lambda t: t[0])
        return [o for _, o in ordered[:k]]


    @staticmethod
    def predict_pos_at_stage(
        obstacle: ObstacleBase,
        stage:    int,
        dt:       float,
    ) -> np.ndarray:
        """
        Predict where `obstacle` will be at MPC stage `stage`.

        Returns
        -------
        (2,) predicted [px, py] [m]
        """
        return obstacle.predict_pos(stage, dt)

    def build_obs_param_block(
        self,
        robot_pos:       np.ndarray,
        stage:           int,
        dt:              float | None = None,
        padding_pos:     float        = 10.0,
    ) -> np.ndarray:
        """
        Build the obstacle portion of the MPC parameter vector for a
        single stage k.

        Layout: for m = 0..max_obs-1:
          block[3*m + 0] = obs_m_px   (predicted x at stage k)
          block[3*m + 1] = obs_m_py   (predicted y at stage k)
          block[3*m + 2] = obs_m_active (1.0 active, 0.0 padded)

        Parameters
        ----------
        robot_pos   : (2,) current robot position (used for nearest-k query)
        stage       : MPC shooting stage index k
        dt          : sampling time [s] (defaults to self.dt)
        padding_pos : value written to px/py for inactive slots

        Returns
        -------
        block : ndarray, shape (3 * max_obs,)
        """
        _dt     = dt if dt is not None else self.dt
        nearest = self.get_nearest_k(robot_pos, k=self.max_obs)
        block   = np.zeros(self.OBS_STRIDE * self.max_obs)

        for m in range(self.max_obs):
            base = self.OBS_STRIDE * m
            if m < len(nearest):
                px, py = self.predict_pos_at_stage(nearest[m], stage, _dt)
                block[base + 0] = px
                block[base + 1] = py
                block[base + 2] = 1.0
            else:
                block[base + 0] = padding_pos
                block[base + 1] = padding_pos
                block[base + 2] = 0.0

        return block

    def __len__(self) -> int:
        return len(self._obstacles)

    def __repr__(self) -> str:
        n_active = len(self.active_obstacles)
        return (f"ObstacleManager({len(self._obstacles)} total, "
                f"{n_active} active, max_obs={self.max_obs})")
