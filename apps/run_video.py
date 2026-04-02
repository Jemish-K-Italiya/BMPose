from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bmpose.constants import DEFAULT_MEDIAPIPE_MODEL, DEFAULT_VIDEOPOSE_CHECKPOINT
from bmpose.mapping import mediapipe33_to_coco17, mediapipe_world_to_h36m17
from bmpose.pipeline import HybridPosePipeline
from bmpose.types import HybridPoseResult, MediaPipePoseFrame
from bmpose.visualization import render_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BMPose on a saved video file.")
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", type=Path, required=True, help="Output annotated video path.")
    parser.add_argument("--export", type=Path, default=None, help="Optional .npz export path.")
    parser.add_argument("--mediapipe-model", type=Path, default=DEFAULT_MEDIAPIPE_MODEL)
    parser.add_argument("--videopose-checkpoint", type=Path, default=DEFAULT_VIDEOPOSE_CHECKPOINT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.input}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamps: list[int] = []
    detection_mask: list[bool] = []
    mediapipe_2d: list[np.ndarray] = []
    mediapipe_world: list[np.ndarray] = []
    coco_2d: list[np.ndarray] = []

    with HybridPosePipeline(
        mediapipe_model_path=args.mediapipe_model,
        videopose_checkpoint_path=args.videopose_checkpoint if args.videopose_checkpoint.exists() else None,
    ) as pipeline:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int((frame_index / max(fps, 1e-6)) * 1000.0)
            mp_frame = pipeline.mediapipe.detect(frame, timestamp_ms)
            timestamps.append(timestamp_ms)

            if mp_frame is None:
                detection_mask.append(False)
                mediapipe_2d.append(np.zeros((33, 5), dtype=np.float32))
                mediapipe_world.append(np.zeros((17, 3), dtype=np.float32))
                coco_2d.append(np.zeros((17, 3), dtype=np.float32))
            else:
                detection_mask.append(True)
                mediapipe_2d.append(mp_frame.landmarks_2d.astype(np.float32))
                coco_2d.append(mediapipe33_to_coco17(mp_frame.landmarks_2d))
                if mp_frame.landmarks_world is not None:
                    mediapipe_world.append(mediapipe_world_to_h36m17(mp_frame.landmarks_world))
                else:
                    mediapipe_world.append(np.zeros((17, 3), dtype=np.float32))

            frame_index += 1
            if frame_index % 50 == 0:
                print(f"Processed {frame_index} frames...")

        cap.release()

        offline = pipeline.build_offline_sequence(
            timestamps_ms=np.asarray(timestamps, dtype=np.int64),
            detection_mask=np.asarray(detection_mask, dtype=bool),
            mediapipe_2d=np.asarray(mediapipe_2d, dtype=np.float32),
            mediapipe_world_3d=np.asarray(mediapipe_world, dtype=np.float32),
            coco_2d=np.asarray(coco_2d, dtype=np.float32),
            fps=fps,
            image_size=(frame_width, frame_height),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    cap = cv2.VideoCapture(str(args.input))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detection_ok = bool(offline.detection_mask[frame_index])
        mp_frame = None
        if detection_ok:
            mp_frame = MediaPipePoseFrame(
                timestamp_ms=int(offline.timestamps_ms[frame_index]),
                image_size=(frame_width, frame_height),
                landmarks_2d=offline.mediapipe_2d[frame_index],
                landmarks_world=None,
                segmentation_mask=None,
            )

        result = HybridPoseResult(
            timestamp_ms=int(offline.timestamps_ms[frame_index]),
            image_size=(frame_width, frame_height),
            detection_ok=detection_ok,
            mediapipe=mp_frame,
            coco_keypoints_2d=offline.coco_2d[frame_index],
            videopose_3d=None if offline.videopose_3d is None else offline.videopose_3d[frame_index],
            hybrid_3d=None if offline.hybrid_3d is None else offline.hybrid_3d[frame_index],
            metrics={
                "mean_confidence": float(offline.coco_2d[frame_index, :, 2].mean()) if detection_ok else 0.0,
                "mean_visibility": float(offline.mediapipe_2d[frame_index, :, 3].mean()) if detection_ok else 0.0,
                "mean_presence": float(offline.mediapipe_2d[frame_index, :, 4].mean()) if detection_ok else 0.0,
                "jitter_2d_px": 0.0,
                "jitter_3d": 0.0,
            },
        )
        rendered = render_result(frame, result, fps=fps)
        writer.write(rendered)
        frame_index += 1

    cap.release()
    writer.release()

    if args.export is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.export,
            timestamps_ms=offline.timestamps_ms,
            detection_mask=offline.detection_mask,
            mediapipe_2d=offline.mediapipe_2d,
            mediapipe_world_h36m=offline.mediapipe_world_3d,
            coco_2d=offline.coco_2d,
            videopose_3d=offline.videopose_3d if offline.videopose_3d is not None else np.zeros((0, 17, 3), dtype=np.float32),
            hybrid_3d=offline.hybrid_3d if offline.hybrid_3d is not None else np.zeros((0, 17, 3), dtype=np.float32),
            summary=json.dumps(offline.summary),
        )

    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(offline.summary, handle, indent=2)

    print(f"Annotated video written to {args.output}")
    if args.export is not None:
        print(f"Predictions exported to {args.export}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
