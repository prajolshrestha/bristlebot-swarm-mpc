"""Unit tests for the light field."""

import unittest
import numpy as np
from sim_engine.light_source_model import LightSource, LightField, SourceMotionMode

class TestLightSourceModel(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        pos = np.array([0.0, 0.0])
        amplitude = 1.0
        sigma = 0.1
        speed = 0.1
        arena_half = 0.4
        self.basic_source = LightSource(pos, amplitude, sigma, self.rng, speed, arena_half, 
                                        motion_mode=SourceMotionMode.STATIONARY, pulsing=False)

    def test_light_source_intensity_center(self):
        intensity = self.basic_source.intensity(np.array([0.0, 0.0]))
        self.assertAlmostEqual(intensity, 1.0)

    def test_light_source_intensity_far(self):
        intensity = self.basic_source.intensity(np.array([1.0, 1.0]))
        self.assertLess(intensity, 1e-10)

    def test_light_source_gradient_center(self):
        grad = self.basic_source.gradient(np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0])

    def test_light_source_gradient_off_center(self):
        grad = self.basic_source.gradient(np.array([0.05, 0.0]))
        self.assertLess(grad[0], 0)
        self.assertAlmostEqual(grad[1], 0.0)

    def test_light_source_step_stationary(self):
        old_pos = self.basic_source.pos.copy()
        self.basic_source.step(0.1)
        np.testing.assert_array_equal(self.basic_source.pos, old_pos)

    def test_light_source_step_rotation(self):
        pos = np.array([0.25, 0.0])
        source = LightSource(pos, 1.0, 0.1, self.rng, 0.1, 0.4, 
                             motion_mode=SourceMotionMode.ROTATION, rotation_radius=0.25, pulsing=False)
        
        source.step(1.0)
        expected_angle = 0.4
        expected_pos = 0.25 * np.array([np.cos(expected_angle), np.sin(expected_angle)])
        np.testing.assert_allclose(source.pos, expected_pos)

    def test_light_field_intensity(self):
        field = LightField(motion_mode=SourceMotionMode.STATIONARY, pulsing=False)
        pos = np.array([0.0, 0.0])
        total_intensity = field.intensity(pos)
        self.assertTrue(0.7 <= total_intensity <= 1.0)

    def test_light_field_brightest_source(self):
        field = LightField(motion_mode=SourceMotionMode.STATIONARY, pulsing=False)
        brightest_pos = field.brightest_source_pos()
        self.assertEqual(brightest_pos.shape, (2,))

if __name__ == '__main__':
    unittest.main()
