import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sibi_bisindo.temporal import sequence_features, match


class TemporalTests(unittest.TestCase):
    def sequence(self, count=20):
        points = np.zeros((count, 21, 3))
        points[:, :, 0] = np.linspace(0, .2, 21)
        points[:, :, 1] = np.linspace(0, .3, 21)
        points[:, :, 0] += np.linspace(0, .2, count)[:, None]
        return np.linspace(0, 1, count), points

    def test_translation_invariance_preserves_motion(self):
        times, points = self.sequence()
        a, motion = sequence_features(times, points, .75)
        b, _ = sequence_features(times, points+[.1, .2, .3], .75)
        stationary, still_motion = sequence_features(times, np.repeat(points[:1], len(times), axis=0), .75)
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertGreater(motion, .5)
        self.assertEqual(still_motion, 0)
        self.assertGreater(float(np.linalg.norm(a-stationary)), 1)

    def test_time_resampling_and_class_distance(self):
        a, _ = sequence_features(*self.sequence(20), .75)
        b, _ = sequence_features(*self.sequence(40), .75)
        np.testing.assert_allclose(a, b, atol=1e-6)
        label, distance, margin = match(a, np.stack([b, b+1]), ['J', 'Z'])
        self.assertEqual(label, 'J')
        self.assertLess(distance, 1e-6)
        self.assertGreater(margin, .9)

    def test_large_landmark_gap_is_not_interpolated(self):
        times, points = self.sequence()
        times[10:] += 1
        with self.assertRaisesRegex(ValueError, 'gap'):
            sequence_features(times, points, .75)
