from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmarkerResult
from mediapipe.tasks.python.vision import RunningMode

from .constants import DEFAULT_MEDIAPIPE_MODEL, MEDIAPIPE_MODEL_URL
from .types import MediaPipePoseFrame


class MediaPipePoseRunner:
    def __init__(
        self,
        model_path: Path | str | None = None,
        min_confidence: float = 0.5,
        output_segmentation_masks: bool = False,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MEDIAPIPE_MODEL
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {self.model_path}. "
                f"Download it from {MEDIAPIPE_MODEL_URL} or run scripts/download_models.py."
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        if hasattr(self._landmarker, "close"):
            self._landmarker.close()

    def __enter__(self) -> "MediaPipePoseRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> MediaPipePoseFrame | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        return self._convert_result(result, frame_bgr.shape[1], frame_bgr.shape[0], int(timestamp_ms))

    @staticmethod
    def _convert_result(
        result: PoseLandmarkerResult,
        width: int,
        height: int,
        timestamp_ms: int,
    ) -> MediaPipePoseFrame | None:
        if not result.pose_landmarks:
            return None

        image_landmarks = result.pose_landmarks[0]
        landmarks_2d = np.array(
            [
                [
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z,
                    getattr(landmark, "visibility", 1.0),
                    getattr(landmark, "presence", 1.0),
                ]
                for landmark in image_landmarks
            ],
            dtype=np.float32,
        )

        landmarks_world = None
        if result.pose_world_landmarks:
            landmarks_world = np.array(
                [
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        getattr(landmark, "visibility", 1.0),
                        getattr(landmark, "presence", 1.0),
                    ]
                    for landmark in result.pose_world_landmarks[0]
                ],
                dtype=np.float32,
            )

        segmentation_mask = None
        if result.segmentation_masks:
            segmentation_mask = np.array(result.segmentation_masks[0].numpy_view(), dtype=np.float32)

        return MediaPipePoseFrame(
            timestamp_ms=timestamp_ms,
            image_size=(width, height),
            landmarks_2d=landmarks_2d,
            landmarks_world=landmarks_world,
            segmentation_mask=segmentation_mask,
        )
