from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class MediaPipePoseFrame:
    timestamp_ms: int
    image_size: tuple[int, int]
    landmarks_2d: np.ndarray
    landmarks_world: np.ndarray | None = None
    segmentation_mask: np.ndarray | None = None


@dataclass(slots=True)
class HybridPoseResult:
    timestamp_ms: int
    image_size: tuple[int, int]
    detection_ok: bool
    mediapipe: MediaPipePoseFrame | None = None
    coco_keypoints_2d: np.ndarray | None = None
    videopose_3d: np.ndarray | None = None
    hybrid_3d: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
