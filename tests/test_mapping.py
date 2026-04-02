from __future__ import annotations

import unittest

import numpy as np

from bmpose.mapping import mediapipe33_to_coco17, mediapipe_world_to_h36m17


class MappingTests(unittest.TestCase):
    def test_mediapipe_to_coco_mapping(self) -> None:
        landmarks = np.zeros((33, 5), dtype=np.float32)
        landmarks[0, :2] = (10.0, 20.0)
        landmarks[0, 3:] = (0.9, 0.7)
        landmarks[11, :2] = (100.0, 200.0)
        landmarks[11, 3:] = (1.0, 1.0)
        landmarks[23, :2] = (150.0, 300.0)
        landmarks[23, 3:] = (0.8, 0.6)

        coco = mediapipe33_to_coco17(landmarks)
        self.assertEqual(coco.shape, (17, 3))
        np.testing.assert_allclose(coco[0, :2], [10.0, 20.0])
        self.assertAlmostEqual(float(coco[0, 2]), 0.8)
        np.testing.assert_allclose(coco[5, :2], [100.0, 200.0])
        np.testing.assert_allclose(coco[11, :2], [150.0, 300.0])

    def test_world_to_h36m_mapping(self) -> None:
        world = np.zeros((33, 5), dtype=np.float32)
        world[11, :3] = (-1.0, 1.0, 0.0)
        world[12, :3] = (1.0, 1.0, 0.0)
        world[23, :3] = (-1.0, -1.0, 0.0)
        world[24, :3] = (1.0, -1.0, 0.0)
        world[0, :3] = (0.0, 2.0, 0.0)
        world[7, :3] = (-0.5, 2.0, 0.0)
        world[8, :3] = (0.5, 2.0, 0.0)

        pose = mediapipe_world_to_h36m17(world)
        self.assertEqual(pose.shape, (17, 3))
        np.testing.assert_allclose(pose[0], [0.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
