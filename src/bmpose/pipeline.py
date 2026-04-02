from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DEFAULT_MEDIAPIPE_MODEL, DEFAULT_VIDEOPOSE_CHECKPOINT
from .filters import ExponentialPoseFilter
from .mapping import center_pose, fuse_h36m_poses, mediapipe33_to_coco17, mediapipe_world_to_h36m17
from .mediapipe_pose import MediaPipePoseRunner
from .types import HybridPoseResult
from .videopose.runtime import VideoPose3DLifter


@dataclass(slots=True)
class OfflineSequence:
    image_size: tuple[int, int]
    fps: float
    timestamps_ms: np.ndarray
    detection_mask: np.ndarray
    mediapipe_2d: np.ndarray
    mediapipe_world_3d: np.ndarray
    coco_2d: np.ndarray
    videopose_3d: np.ndarray | None
    hybrid_3d: np.ndarray | None
    summary: dict[str, float]


class HybridPosePipeline:
    def __init__(
        self,
        mediapipe_model_path: Path | str | None = None,
        videopose_checkpoint_path: Path | str | None = None,
        min_confidence: float = 0.5,
        fusion_weight: float = 0.7,
        two_d_smoothing: float = 0.65,
        three_d_smoothing: float = 0.5,
        output_segmentation_masks: bool = False,
    ) -> None:
        self.mediapipe = MediaPipePoseRunner(
            model_path=mediapipe_model_path or DEFAULT_MEDIAPIPE_MODEL,
            min_confidence=min_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )
        self.fusion_weight = fusion_weight
        self.two_d_filter = ExponentialPoseFilter(alpha=two_d_smoothing)
        self.three_d_filter = ExponentialPoseFilter(alpha=three_d_smoothing)
        self.lifter = None
        candidate_checkpoint = Path(videopose_checkpoint_path) if videopose_checkpoint_path else DEFAULT_VIDEOPOSE_CHECKPOINT
        if candidate_checkpoint.exists():
            self.lifter = VideoPose3DLifter(checkpoint_path=candidate_checkpoint)

        self._last_coco: np.ndarray | None = None
        self._last_pose3d: np.ndarray | None = None

    def close(self) -> None:
        self.mediapipe.close()

    def __enter__(self) -> "HybridPosePipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process_live_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HybridPoseResult:
        mediapipe_frame = self.mediapipe.detect(frame_bgr, timestamp_ms)
        width, height = frame_bgr.shape[1], frame_bgr.shape[0]

        if mediapipe_frame is None:
            self.two_d_filter.reset()
            self.three_d_filter.reset()
            self._last_coco = None
            self._last_pose3d = None
            return HybridPoseResult(
                timestamp_ms=timestamp_ms,
                image_size=(width, height),
                detection_ok=False,
                metrics={
                    "mean_confidence": 0.0,
                    "mean_visibility": 0.0,
                    "mean_presence": 0.0,
                    "jitter_2d_px": 0.0,
                    "jitter_3d": 0.0,
                },
            )

        coco = mediapipe33_to_coco17(mediapipe_frame.landmarks_2d)
        smoothed_xy = self.two_d_filter.update(coco[:, :2])
        smoothed_coco = np.concatenate([smoothed_xy, coco[:, 2:3]], axis=1)

        videopose_3d = None
        if self.lifter is not None:
            videopose_3d = self.lifter.predict_current(smoothed_coco[:, :2], mediapipe_frame.image_size)
            videopose_3d = center_pose(videopose_3d)

        mediapipe_world_h36m = None
        if mediapipe_frame.landmarks_world is not None:
            mediapipe_world_h36m = mediapipe_world_to_h36m17(mediapipe_frame.landmarks_world)

        hybrid_3d = videopose_3d
        if videopose_3d is not None and mediapipe_world_h36m is not None:
            hybrid_3d = fuse_h36m_poses(videopose_3d, mediapipe_world_h36m, primary_weight=self.fusion_weight)
        elif mediapipe_world_h36m is not None:
            hybrid_3d = mediapipe_world_h36m

        if hybrid_3d is not None:
            hybrid_3d = self.three_d_filter.update(hybrid_3d)

        mean_visibility = float(np.mean(mediapipe_frame.landmarks_2d[:, 3]))
        mean_presence = float(np.mean(mediapipe_frame.landmarks_2d[:, 4]))
        mean_confidence = float(np.mean(smoothed_coco[:, 2]))
        jitter_2d = 0.0 if self._last_coco is None else float(np.linalg.norm(smoothed_coco[:, :2] - self._last_coco, axis=-1).mean())
        jitter_3d = 0.0 if self._last_pose3d is None or hybrid_3d is None else float(
            np.linalg.norm(hybrid_3d - self._last_pose3d, axis=-1).mean()
        )

        self._last_coco = smoothed_coco[:, :2].copy()
        self._last_pose3d = hybrid_3d.copy() if hybrid_3d is not None else None

        return HybridPoseResult(
            timestamp_ms=timestamp_ms,
            image_size=(width, height),
            detection_ok=True,
            mediapipe=mediapipe_frame,
            coco_keypoints_2d=smoothed_coco,
            videopose_3d=videopose_3d,
            hybrid_3d=hybrid_3d,
            metrics={
                "mean_confidence": mean_confidence,
                "mean_visibility": mean_visibility,
                "mean_presence": mean_presence,
                "jitter_2d_px": jitter_2d,
                "jitter_3d": jitter_3d,
            },
        )

    def build_offline_sequence(
        self,
        timestamps_ms: np.ndarray,
        detection_mask: np.ndarray,
        mediapipe_2d: np.ndarray,
        mediapipe_world_3d: np.ndarray,
        coco_2d: np.ndarray,
        fps: float,
        image_size: tuple[int, int],
    ) -> OfflineSequence:
        videopose_3d = None
        if self.lifter is not None:
            videopose_3d = self.lifter.predict_sequence(coco_2d[..., :2], image_size=image_size, valid_mask=detection_mask)

        hybrid_3d = None
        if videopose_3d is not None and len(mediapipe_world_3d) > 0:
            fused = []
            for index in range(len(videopose_3d)):
                if detection_mask[index]:
                    fused.append(fuse_h36m_poses(videopose_3d[index], mediapipe_world_3d[index], primary_weight=self.fusion_weight))
                else:
                    fused.append(videopose_3d[index])
            hybrid_3d = np.asarray(fused, dtype=np.float32)
        elif videopose_3d is not None:
            hybrid_3d = videopose_3d
        elif len(mediapipe_world_3d) > 0:
            hybrid_3d = mediapipe_world_3d

        summary = {
            "frames": float(len(timestamps_ms)),
            "fps": float(fps),
            "detection_rate": float(np.mean(detection_mask)) if len(detection_mask) else 0.0,
            "mean_confidence": float(coco_2d[detection_mask, :, 2].mean()) if np.any(detection_mask) else 0.0,
        }

        return OfflineSequence(
            image_size=image_size,
            fps=fps,
            timestamps_ms=timestamps_ms.astype(np.int64),
            detection_mask=detection_mask.astype(bool),
            mediapipe_2d=mediapipe_2d.astype(np.float32),
            mediapipe_world_3d=mediapipe_world_3d.astype(np.float32),
            coco_2d=coco_2d.astype(np.float32),
            videopose_3d=None if videopose_3d is None else videopose_3d.astype(np.float32),
            hybrid_3d=None if hybrid_3d is None else hybrid_3d.astype(np.float32),
            summary=summary,
        )
