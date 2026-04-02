from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..constants import DEFAULT_VIDEOPOSE_CHECKPOINT, VIDEOPOSE_CHECKPOINT_URL
from .model import TemporalModel


def normalize_screen_coordinates(points: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    aspect = float(height) / float(width)
    return (points / float(width) * 2.0) - np.array([1.0, aspect], dtype=np.float32)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def fill_missing_keypoints(sequence: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if sequence.ndim != 3:
        raise ValueError("Expected [time, joints, features] sequence.")

    if not valid_mask.any():
        return np.zeros_like(sequence)

    frame_indices = np.arange(len(sequence))
    for joint_index in range(sequence.shape[1]):
        for feature_index in range(sequence.shape[2]):
            values = sequence[:, joint_index, feature_index]
            known_indices = frame_indices[valid_mask]
            known_values = values[valid_mask]
            sequence[:, joint_index, feature_index] = np.interp(frame_indices, known_indices, known_values)
    return sequence


def load_videopose3d(
    checkpoint_path: Path | str | None = None,
    filter_widths: tuple[int, ...] = (3, 3, 3, 3, 3),
    channels: int = 1024,
    causal: bool = False,
    device: str | torch.device | None = None,
) -> tuple[TemporalModel, torch.device]:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_VIDEOPOSE_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"VideoPose3D checkpoint not found at {checkpoint_path}. "
            f"Download it from {VIDEOPOSE_CHECKPOINT_URL} or run scripts/download_models.py."
        )

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    state_dict = checkpoint["model_pos"] if isinstance(checkpoint, dict) and "model_pos" in checkpoint else checkpoint
    state_dict = _strip_module_prefix(state_dict)

    model = TemporalModel(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=list(filter_widths),
        causal=causal,
        dropout=0.25,
        channels=channels,
        dense=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(resolved_device)
    return model, resolved_device


class VideoPose3DLifter:
    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        filter_widths: tuple[int, ...] = (3, 3, 3, 3, 3),
        channels: int = 1024,
        causal: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        self.model, self.device = load_videopose3d(
            checkpoint_path=checkpoint_path,
            filter_widths=filter_widths,
            channels=channels,
            causal=causal,
            device=device,
        )
        self.receptive_field = self.model.receptive_field()
        self.pad = (self.receptive_field - 1) // 2
        self._buffer: deque[np.ndarray] = deque(maxlen=self.receptive_field)

    def reset(self) -> None:
        self._buffer.clear()

    def predict_current(self, keypoints_xy: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        width, height = image_size
        normalized = normalize_screen_coordinates(np.asarray(keypoints_xy, dtype=np.float32), width, height)
        self._buffer.append(normalized)

        sequence = np.stack(list(self._buffer), axis=0)
        padded = np.pad(sequence, ((self.pad, self.pad), (0, 0), (0, 0)), mode="edge")
        center = len(sequence) - 1 + self.pad
        window = padded[center - self.pad : center + self.pad + 1]

        with torch.inference_mode():
            tensor = torch.from_numpy(window[None, ...]).float().to(self.device)
            prediction = self.model(tensor)[0, 0].detach().cpu().numpy().astype(np.float32)
        return prediction - prediction[:1]

    def predict_sequence(
        self,
        keypoints_xy_sequence: np.ndarray,
        image_size: tuple[int, int],
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        width, height = image_size
        sequence = np.asarray(keypoints_xy_sequence, dtype=np.float32)
        if valid_mask is None:
            valid_mask = np.isfinite(sequence).all(axis=(1, 2))
        sequence = fill_missing_keypoints(sequence, valid_mask)
        normalized = normalize_screen_coordinates(sequence, width, height)
        padded = np.pad(normalized, ((self.pad, self.pad), (0, 0), (0, 0)), mode="edge")

        with torch.inference_mode():
            tensor = torch.from_numpy(padded[None, ...]).float().to(self.device)
            predictions = self.model(tensor)[0].detach().cpu().numpy().astype(np.float32)
        return predictions - predictions[:, :1]
