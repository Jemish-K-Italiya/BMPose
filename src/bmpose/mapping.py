from __future__ import annotations

import numpy as np

from .constants import H36M_CONNECTIONS, MEDIAPIPE_TO_COCO


def _confidence_from_landmark(landmark: np.ndarray) -> float:
    if landmark.shape[0] >= 5:
        return float(np.clip((landmark[3] + landmark[4]) * 0.5, 0.0, 1.0))
    if landmark.shape[0] >= 4:
        return float(np.clip(landmark[3], 0.0, 1.0))
    return 1.0


def mediapipe33_to_coco17(landmarks_2d: np.ndarray) -> np.ndarray:
    """Convert MediaPipe's 33 landmarks into COCO's 17-keypoint layout."""

    landmarks_2d = np.asarray(landmarks_2d, dtype=np.float32)
    if landmarks_2d.shape[0] != 33:
        raise ValueError("Expected MediaPipe input with 33 landmarks.")

    coco = np.zeros((17, 3), dtype=np.float32)
    for mp_index, coco_index in MEDIAPIPE_TO_COCO.items():
        coco[coco_index, :2] = landmarks_2d[mp_index, :2]
        coco[coco_index, 2] = _confidence_from_landmark(landmarks_2d[mp_index])
    return coco


def mediapipe_world_to_h36m17(world_landmarks: np.ndarray) -> np.ndarray:
    """Approximate an H36M-style 17-joint skeleton from MediaPipe world landmarks."""

    world_landmarks = np.asarray(world_landmarks, dtype=np.float32)
    if world_landmarks.shape[0] != 33:
        raise ValueError("Expected MediaPipe world landmarks with 33 joints.")

    l_shoulder = world_landmarks[11, :3]
    r_shoulder = world_landmarks[12, :3]
    l_elbow = world_landmarks[13, :3]
    r_elbow = world_landmarks[14, :3]
    l_wrist = world_landmarks[15, :3]
    r_wrist = world_landmarks[16, :3]
    l_hip = world_landmarks[23, :3]
    r_hip = world_landmarks[24, :3]
    l_knee = world_landmarks[25, :3]
    r_knee = world_landmarks[26, :3]
    l_ankle = world_landmarks[27, :3]
    r_ankle = world_landmarks[28, :3]
    nose = world_landmarks[0, :3]
    l_ear = world_landmarks[7, :3]
    r_ear = world_landmarks[8, :3]

    pelvis = (l_hip + r_hip) * 0.5
    thorax = (l_shoulder + r_shoulder) * 0.5
    spine = (pelvis + thorax) * 0.5
    head = (nose + l_ear + r_ear) / 3.0
    neck = (thorax + head) * 0.5

    pose = np.stack(
        [
            pelvis,
            r_hip,
            r_knee,
            r_ankle,
            l_hip,
            l_knee,
            l_ankle,
            spine,
            thorax,
            neck,
            head,
            l_shoulder,
            l_elbow,
            l_wrist,
            r_shoulder,
            r_elbow,
            r_wrist,
        ],
        axis=0,
    )
    return center_pose(pose)


def center_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 2:
        return pose - pose[:1]
    return pose - pose[:, :1]


def pose_scale(pose: np.ndarray) -> float:
    centered = center_pose(pose)
    lengths = []
    for a, b in H36M_CONNECTIONS:
        length = float(np.linalg.norm(centered[a] - centered[b]))
        if length > 1e-6:
            lengths.append(length)
    return float(np.mean(lengths)) if lengths else 1.0


def fuse_h36m_poses(primary: np.ndarray, secondary: np.ndarray, primary_weight: float = 0.7) -> np.ndarray:
    """Fuse two root-centered H36M-like poses after normalizing scale."""

    primary = center_pose(primary)
    secondary = center_pose(secondary)

    primary_scale = pose_scale(primary)
    secondary_scale = pose_scale(secondary)

    if primary_scale <= 1e-6 or secondary_scale <= 1e-6:
        return primary.copy()

    target_scale = secondary_scale
    primary_norm = primary / primary_scale
    secondary_norm = secondary / secondary_scale
    fused = (primary_weight * primary_norm) + ((1.0 - primary_weight) * secondary_norm)
    return fused * target_scale
