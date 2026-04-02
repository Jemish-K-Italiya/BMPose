from __future__ import annotations

import unittest

import numpy as np

from bmpose.metrics import mean_joint_error, mpjpe, n_mpjpe, p_mpjpe, pck_2d


class MetricTests(unittest.TestCase):
    def test_zero_errors_for_identical_inputs(self) -> None:
        pose = np.zeros((2, 17, 3), dtype=np.float32)
        self.assertAlmostEqual(mpjpe(pose, pose), 0.0)
        self.assertAlmostEqual(n_mpjpe(pose + 1.0, pose + 1.0), 0.0)

    def test_procrustes_alignment(self) -> None:
        target = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=np.float32,
        )
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        predicted = (target @ rotation.T) * 2.0 + np.array([5.0, -3.0, 2.0], dtype=np.float32)
        self.assertLess(p_mpjpe(predicted, target), 1e-4)

    def test_pck_and_mean_joint_error(self) -> None:
        target = np.zeros((1, 17, 2), dtype=np.float32)
        target[0, 5] = [0.0, 0.0]
        target[0, 6] = [2.0, 0.0]
        target[0, 11] = [0.0, -2.0]
        target[0, 12] = [2.0, -2.0]
        predicted = target.copy()
        predicted[0, 0] = [0.1, 0.1]
        self.assertGreaterEqual(pck_2d(predicted, target), 0.9)
        self.assertGreater(mean_joint_error(predicted, target), 0.0)


if __name__ == "__main__":
    unittest.main()
