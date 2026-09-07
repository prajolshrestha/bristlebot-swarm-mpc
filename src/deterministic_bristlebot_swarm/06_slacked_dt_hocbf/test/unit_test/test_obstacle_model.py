"""Unit tests for the obstacle model."""

import unittest
import numpy as np
from sim_engine.obstacle_model import ObstacleManager, StaticObstacle, MovingObstacle

class TestObstacleModel(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.arena_half = 0.4
        self.manager = ObstacleManager(dt=self.dt, arena_half=self.arena_half, max_obs=3, obs_range=0.5)

    def test_static_obstacle_predict(self):
        pos = [0.1, 0.2]
        obs = StaticObstacle(pos=pos, radius=0.05)
        pred = obs.predict_pos(stage=10, dt=self.dt)
        np.testing.assert_array_equal(pred, pos)

    def test_moving_obstacle_step(self):
        obs = MovingObstacle(pos=[0, 0], vel=[1.0, 0], radius=0.05, arena_half=0.4, bounce=True)
        obs.step(0.1)
        np.testing.assert_allclose(obs.pos, [0.1, 0])

    def test_moving_obstacle_bounce(self):
        obs = MovingObstacle(pos=[0.34, 0], vel=[0.2, 0], radius=0.05, arena_half=0.4, bounce=True)
        obs.step(0.1)
        self.assertAlmostEqual(obs.pos[0], 0.35)
        self.assertAlmostEqual(obs.vel[0], -0.2)

    def test_manager_get_nearest_k(self):
        obs1 = StaticObstacle(pos=[0.1, 0], active=True)
        obs2 = StaticObstacle(pos=[0.3, 0], active=True)
        obs3 = StaticObstacle(pos=[0.2, 0], active=False)
        self.manager.add(obs1)
        self.manager.add(obs2)
        self.manager.add(obs3)
        nearest = self.manager.get_nearest_k(np.array([0, 0]), k=2)
        self.assertEqual(len(nearest), 2)
        self.assertIs(nearest[0], obs1)
        self.assertIs(nearest[1], obs2)

    def test_manager_build_obs_param_block(self):
        obs1 = StaticObstacle(pos=[0.1, 0.2], active=True)
        self.manager.add(obs1)
        block = self.manager.build_obs_param_block(np.array([0, 0]), stage=0)
        self.assertEqual(len(block), 9)
        self.assertAlmostEqual(block[0], 0.1)
        self.assertAlmostEqual(block[1], 0.2)
        self.assertAlmostEqual(block[2], 1.0)
        self.assertAlmostEqual(block[3], 10.0)
        self.assertAlmostEqual(block[5], 0.0)

    def test_moving_obstacle_predict_no_clip(self):
        obs = MovingObstacle(pos=[0.34, 0], vel=[0.2, 0], radius=0.05, arena_half=0.4, bounce=True)
        pred = obs.predict_pos(stage=1, dt=0.1)
        self.assertAlmostEqual(pred[0], 0.36)

if __name__ == '__main__':
    unittest.main()
