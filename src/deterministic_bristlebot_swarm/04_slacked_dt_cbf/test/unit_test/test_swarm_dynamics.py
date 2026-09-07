"""Unit tests for the flocking rules and behavior modes."""

import unittest
import numpy as np
from unittest.mock import MagicMock
from sim_engine.swarm_dynamics import ReynoldsBehavior, SwarmBehavior

class TestSwarmDynamics(unittest.TestCase):
    def setUp(self):
        self.arena_half = 0.4
        self.behavior = ReynoldsBehavior(arena_half=self.arena_half, lookahead=0.07, v_target=0.1)
        
        self.robot = MagicMock()
        self.robot.pos = np.array([0.0, 0.0])
        self.robot.theta = 0.0
        self.robot.velocity = np.array([0.0, 0.0])

    def test_phototaxis_stationary_source(self):
        mock_source = MagicMock()
        mock_source.pos = np.array([0.1, 0.1])
        mock_source.amplitude = 1.0
        
        mock_field = MagicMock()
        mock_field.sources = [mock_source]
        mock_field.gradient.return_value = np.array([1.0, 1.0])
        
        goal_pos, ref_theta, ref_vel = self.behavior.compute_goal(
            self.robot, neighbors=[], behavior=SwarmBehavior.PHOTOTAXIS, field=mock_field
        )
        
        self.assertAlmostEqual(ref_theta, np.pi/4)
        expected_goal = self.robot.pos + 0.07 * np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
        np.testing.assert_allclose(goal_pos, expected_goal)

    def test_orbit_calculation(self):
        self.behavior.orbit_center = np.array([0.0, 0.0])
        self.behavior.orbit_radius = 0.2
        self.behavior.orbit_speed = 1.0
        
        self.robot.pos = np.array([0.2, 0.0])
        
        goal_pos, ref_theta, ref_vel = self.behavior.compute_goal(
            self.robot, neighbors=[], behavior=SwarmBehavior.ORBIT
        )
        
        self.assertAlmostEqual(ref_theta, np.pi/2)
        expected_angle = 0.0 + 0.35
        expected_goal = 0.2 * np.array([np.cos(expected_angle), np.sin(expected_angle)])
        np.testing.assert_allclose(goal_pos, expected_goal)


    def test_phototaxis_no_field_random_walk(self):
        goal_pos, ref_theta, ref_vel = self.behavior.compute_goal(
            self.robot, neighbors=[], behavior=SwarmBehavior.PHOTOTAXIS, field=None
        )
        self.assertEqual(goal_pos.shape, (2,))
        self.assertEqual(ref_vel.shape, (2,))

if __name__ == '__main__':
    unittest.main()
