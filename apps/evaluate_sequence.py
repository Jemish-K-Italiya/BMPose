from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bmpose.metrics import mean_joint_error, metrics_to_json, mpjpe, n_mpjpe, p_mpjpe, pck_2d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exported BMPose predictions against ground truth.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--pred-key", required=True, help="Key inside the prediction npz.")
    parser.add_argument("--gt-key", required=True, help="Key inside the ground-truth npz.")
    parser.add_argument("--mode", choices=["2d", "3d"], required=True)
    return parser


def load_key(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}")
        value = data[key]
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S", "O"} and value.shape == ():
            try:
                return np.asarray(json.loads(value.item()))
            except Exception:
                return value
        return np.asarray(value)


def main() -> int:
    args = build_parser().parse_args()
    predicted = load_key(args.predictions, args.pred_key)
    target = load_key(args.ground_truth, args.gt_key)

    if args.mode == "2d":
        predicted_xy = predicted[..., :2]
        target_xy = target[..., :2]
        metrics = {
            "mean_joint_error_px": mean_joint_error(predicted_xy, target_xy),
            "pck@0.2": pck_2d(predicted_xy, target_xy),
        }
    else:
        predicted_xyz = predicted[..., :3]
        target_xyz = target[..., :3]
        if predicted_xyz.ndim == 2:
            predicted_xyz = predicted_xyz[None, ...]
        if target_xyz.ndim == 2:
            target_xyz = target_xyz[None, ...]
        metrics = {
            "mpjpe_mm": mpjpe(predicted_xyz, target_xyz),
            "p_mpjpe_mm": p_mpjpe(predicted_xyz, target_xyz),
            "n_mpjpe_mm": n_mpjpe(predicted_xyz, target_xyz),
        }

    print(metrics_to_json(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
