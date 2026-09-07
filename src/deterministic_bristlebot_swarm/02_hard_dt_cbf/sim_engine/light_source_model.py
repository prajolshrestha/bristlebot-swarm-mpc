"""The light field the swarm senses, and how it moves."""

import sys
import os
import time
from enum import Enum, auto
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


N_ROBOTS       = 10
DT             = 0.1
MPC_HORIZON    = 15
ARENA_HALF    = 0.4

N_SOURCES      = 1
SOURCE_SIGMA   = 0.10
SOURCE_SPEED   = 0.15
SOURCE_SEED    = 7

V_TARGET       = 0.10
LOOKAHEAD_DIST = 0.07
EPSILON_GRAD   = 1e-6

HEATMAP_RES    = 20



class SourceMotionMode(Enum):
    RANDOM_WALK = auto()
    ROTATION    = auto()
    STATIONARY  = auto()
class LightSource:
    def __init__(self, pos, amplitude, sigma, rng, speed, arena_half, 
                 motion_mode=SourceMotionMode.RANDOM_WALK, rotation_radius=0.25, pulsing=True):
        self.pos       = pos.copy()
        self.amplitude = amplitude
        self.sigma     = sigma
        self._rng      = rng
        self._speed    = speed
        self._arena    = arena_half
        self._vel      = rng.uniform(-speed, speed, 2)
        self._phase    = rng.uniform(0, 2 * np.pi)
        
        self.motion_mode = motion_mode
        self.radius      = rotation_radius
        self.pulsing     = pulsing
        
        if not self.pulsing:
            self.amplitude = 1.0
            
        if np.linalg.norm(pos) > 1e-4:
            self._angle = np.arctan2(pos[1], pos[0])
        else:
            self._angle = 0.0

    def step(self, dt):
        if self.pulsing:
            self._phase    += dt * 0.8
            self.amplitude  = 0.7 + 0.3 * np.sin(self._phase)
        else:
            pass

        if self.motion_mode == SourceMotionMode.RANDOM_WALK:
            self._vel      += self._rng.normal(0, self._speed * 0.3, 2)
            self._vel       = np.clip(self._vel, -self._speed, self._speed)
            new_pos         = self.pos + self._vel * dt
            for ax in range(2):
                if abs(new_pos[ax]) > self._arena * 0.85:
                    self._vel[ax] *= -1.0
                    new_pos[ax]    = np.clip(new_pos[ax],
                                             -self._arena * 0.85, self._arena * 0.85)
            self.pos = new_pos
            
        elif self.motion_mode == SourceMotionMode.ROTATION:
            omega = self._speed / max(self.radius, 1e-4)
            self._angle += omega * dt
            self.pos = self.radius * np.array([np.cos(self._angle), np.sin(self._angle)])
            
        elif self.motion_mode == SourceMotionMode.STATIONARY:
            pass

    def intensity(self, pos):
        d2 = np.sum((pos - self.pos) ** 2)
        return self.amplitude * np.exp(-d2 / (2 * self.sigma ** 2))

    def gradient(self, pos):
        diff = pos - self.pos
        I    = self.intensity(pos)
        return -I * diff / (self.sigma ** 2)


class LightField:
    def __init__(self, motion_mode=SourceMotionMode.RANDOM_WALK, rotation_radius=0.25, pulsing=True):
        rng = np.random.default_rng(SOURCE_SEED)
        
        if motion_mode == SourceMotionMode.ROTATION:
            start_pos = np.array([rotation_radius, 0.0])
        else:
            start_pos = np.zeros(2)
        
        self.sources = [
            LightSource(
                start_pos,
                rng.uniform(0.7, 1.0), SOURCE_SIGMA, rng, SOURCE_SPEED, ARENA_HALF,
                motion_mode=motion_mode, rotation_radius=rotation_radius, pulsing=pulsing
            )
            for _ in range(N_SOURCES)
        ]
    def brightest_source_pos(self) -> np.ndarray:
        """Return position of the source with highest current amplitude."""
        return max(self.sources, key=lambda s: s.amplitude).pos.copy()

    def step(self, dt):
        for s in self.sources:
            s.step(dt)

    def intensity(self, pos):
        return float(sum(s.intensity(pos) for s in self.sources))

    def gradient(self, pos):
        return sum(s.gradient(pos) for s in self.sources)

    def heatmap(self, res=HEATMAP_RES):
        xs = np.linspace(-ARENA_HALF, ARENA_HALF, res)
        ys = np.linspace(-ARENA_HALF, ARENA_HALF, res)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.zeros((res, res))
        for s in self.sources:
            d2   = (XX - s.pos[0]) ** 2 + (YY - s.pos[1]) ** 2
            grid += s.amplitude * np.exp(-d2 / (2 * s.sigma ** 2))
        return grid, [-ARENA_HALF, ARENA_HALF, -ARENA_HALF, ARENA_HALF]
