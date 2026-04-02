from __future__ import annotations

import json
from typing import Any

import numpy as np


def mean_joint_error(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(np.linalg.norm(predicted - target, axis=-1).mean())


def pck_2d(predicted: np.ndarray, target: np.ndarray, threshold: float = 0.2) -> float:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.shape[-2] < 17:
        raise ValueError("PCK expects a COCO-like 17 joint layout.")

    left_shoulder = target[..., 5, :]
    right_shoulder = target[..., 6, :]
    left_hip = target[..., 11, :]
    right_hip = target[..., 12, :]

    torso_1 = np.linalg.norm(left_shoulder - right_hip, axis=-1)
    torso_2 = np.linalg.norm(right_shoulder - left_hip, axis=-1)
    scale = np.maximum(np.maximum(torso_1, torso_2), 1e-6)
    distances = np.linalg.norm(predicted - target, axis=-1)
    correct = distances <= (scale[..., None] * threshold)
    return float(correct.mean())


def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(np.linalg.norm(predicted - target, axis=-1).mean())


def n_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    predicted_flat = predicted.reshape(predicted.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    numerator = np.sum(target_flat * predicted_flat, axis=1, keepdims=True)
    denominator = np.sum(predicted_flat * predicted_flat, axis=1, keepdims=True) + 1e-8
    scale = numerator / denominator
    scaled = predicted * scale[:, None, None]
    return mpjpe(scaled, target)


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have matching shapes.")

    aligned_predictions = np.empty_like(predicted)
    for index in range(predicted.shape[0]):
        x = target[index]
        y = predicted[index]

        mu_x = x.mean(axis=0, keepdims=True)
        mu_y = y.mean(axis=0, keepdims=True)
        x0 = x - mu_x
        y0 = y - mu_y

        norm_x = np.linalg.norm(x0)
        norm_y = np.linalg.norm(y0)
        if norm_x < 1e-8 or norm_y < 1e-8:
            aligned_predictions[index] = y
            continue

        x0 /= norm_x
        y0 /= norm_y

        h = x0.T @ y0
        u, s, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T

        scale = s.sum() * (norm_x / norm_y)
        translation = mu_x - scale * (mu_y @ r)
        aligned_predictions[index] = scale * (y @ r) + translation

    return mpjpe(aligned_predictions, target)


def sequence_jitter(sequence: np.ndarray) -> float:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(sequence, axis=0), axis=-1).mean())


def metrics_to_json(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)
