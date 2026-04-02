from __future__ import annotations

import math

import cv2
import numpy as np

from .constants import H36M_CONNECTIONS, MEDIAPIPE_CONNECTIONS
from .types import HybridPoseResult


def _put_text_block(frame: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    line_height = 20
    block_height = max(30, len(lines) * line_height + 14)
    block_width = 270
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 18), (x + block_width, y - 18 + block_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, dst=frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + (index * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


def draw_mediapipe_skeleton(frame: np.ndarray, landmarks_2d: np.ndarray) -> None:
    for start, end in MEDIAPIPE_CONNECTIONS:
        point_a = tuple(np.round(landmarks_2d[start, :2]).astype(int))
        point_b = tuple(np.round(landmarks_2d[end, :2]).astype(int))
        cv2.line(frame, point_a, point_b, (80, 220, 255), 2, cv2.LINE_AA)

    for landmark in landmarks_2d:
        x, y = np.round(landmark[:2]).astype(int)
        confidence = float(np.clip((landmark[3] + landmark[4]) * 0.5, 0.0, 1.0))
        color = (40, int(160 + 95 * confidence), 255)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)


def _rotate_for_display(pose_3d: np.ndarray) -> np.ndarray:
    yaw = math.radians(25.0)
    pitch = math.radians(-15.0)

    rot_y = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return pose_3d @ rot_y.T @ rot_x.T


def draw_3d_inset(frame: np.ndarray, pose_3d: np.ndarray, label: str = "Hybrid 3D") -> None:
    inset_width = 260
    inset_height = 260
    margin = 20
    x0 = frame.shape[1] - inset_width - margin
    y0 = margin
    inset = np.full((inset_height, inset_width, 3), 18, dtype=np.uint8)

    pose = _rotate_for_display(pose_3d.astype(np.float32))
    pose[:, 1] *= -1.0
    max_extent = float(np.max(np.abs(pose))) if pose.size else 1.0
    scale = 90.0 / max(max_extent, 1e-6)
    projected = pose[:, :2] * scale
    projected[:, 0] += inset_width / 2
    projected[:, 1] += inset_height / 2 + 40

    for start, end in H36M_CONNECTIONS:
        p1 = tuple(np.round(projected[start]).astype(int))
        p2 = tuple(np.round(projected[end]).astype(int))
        depth = float((pose[start, 2] + pose[end, 2]) * 0.5)
        color = (60, int(np.clip(180 - depth * 35, 80, 255)), 255)
        cv2.line(inset, p1, p2, color, 2, cv2.LINE_AA)

    for joint_index, point in enumerate(projected):
        depth = float(pose[joint_index, 2])
        color = (50, int(np.clip(220 - depth * 40, 70, 255)), 255)
        cv2.circle(inset, tuple(np.round(point).astype(int)), 4, color, -1, cv2.LINE_AA)

    cv2.putText(inset, label, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    frame[y0 : y0 + inset_height, x0 : x0 + inset_width] = inset


def render_result(frame_bgr: np.ndarray, result: HybridPoseResult, fps: float | None = None) -> np.ndarray:
    frame = frame_bgr.copy()
    if result.detection_ok and result.mediapipe is not None:
        draw_mediapipe_skeleton(frame, result.mediapipe.landmarks_2d)

    pose_3d = result.hybrid_3d if result.hybrid_3d is not None else result.videopose_3d
    if result.detection_ok and pose_3d is not None:
        label = "Hybrid 3D" if result.hybrid_3d is not None and result.videopose_3d is not None else "3D Pose"
        draw_3d_inset(frame, pose_3d, label=label)

    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    lines.extend(
        [
            f"Pose confidence: {result.metrics.get('mean_confidence', 0.0):.2f}",
            f"Visibility: {result.metrics.get('mean_visibility', 0.0):.2f}",
            f"Presence: {result.metrics.get('mean_presence', 0.0):.2f}",
            f"2D jitter(px): {result.metrics.get('jitter_2d_px', 0.0):.1f}",
            f"3D jitter: {result.metrics.get('jitter_3d', 0.0):.3f}",
        ]
    )
    if not result.detection_ok:
        lines.insert(0, "No person detected")
    _put_text_block(frame, lines, (18, 34))
    return frame
