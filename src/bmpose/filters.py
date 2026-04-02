from __future__ import annotations

import numpy as np


class ExponentialPoseFilter:
    """A tiny EMA filter for pose arrays."""

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32)
        if self._state is None or self._state.shape != pose.shape:
            self._state = pose.copy()
        else:
            self._state = (self.alpha * pose) + ((1.0 - self.alpha) * self._state)
        return self._state.copy()
