# BMPose Hybrid Pipeline

`BMPose` combines:

- MediaPipe Pose Landmarker Heavy for real-time single-person landmark detection.
- VideoPose3D for temporal 3D lifting with official pretrained weights.
- A lightweight fusion stage so the live demo can use MediaPipe geometry and VideoPose3D temporal consistency together.

This repository is organized for an academic/demo-style deliverable:

- live skeleton detection on webcam or video,
- pretrained models instead of full training,
- exportable outputs for presentations,
- measurable accuracy/error metrics for labeled data.

## Why This Design

The hybrid approach is practical because:

- MediaPipe is fast enough for live use.
- VideoPose3D improves temporal stability over frame-only pose estimates.
- Both pieces have pretrained assets, so the project is feasible without expensive training.
- Standard report metrics such as MPJPE and P-MPJPE are supported.

## Project Layout

- `src/bmpose/`: our implementation.
- `apps/`: runnable scripts for live demo, video processing, and metric evaluation.
- `scripts/`: helper scripts for downloading weights and cloning reference repos.
- `third_party/`: optional external source clones kept separate on purpose.
- `tests/`: small correctness tests for mapping and metrics.

## Setup

1. Install the package in editable mode:

```powershell
python -m pip install -e .
```

2. Download the pretrained weights:

```powershell
python scripts/download_models.py
```
- `data/models/pose_landmarker_heavy.task`
- `data/models/pretrained_h36m_detectron_coco.bin`

   OR 
 
   pretrained weights are avialable [HERE](https://drive.google.com/drive/folders/1pvnpDTCp8T7rHEojsqrg04XrGocf2gAK)



## Run The Live Demo

```powershell
python apps/live_demo.py
```

Example:

```powershell
python apps/live_demo.py --camera 0 --width 1280 --height 720 --save outputs/live_demo.mp4
```

The live view shows:

- 2D skeleton over the camera frame,
- a 3D inset pose from the hybrid pipeline,
- FPS,
- keypoint confidence,
- landmark visibility/presence,
- temporal stability indicators.

## Run On A Video File

```powershell
python apps/run_video.py --input path\to\input.mp4 --output outputs\annotated.mp4 --export outputs\predictions.npz
```

This does an offline two-pass pipeline:

1. detect MediaPipe landmarks for each frame,
2. run VideoPose3D over the whole 2D sequence,
3. fuse the 3D streams,
4. render the annotated output video,
5. export predictions as `.npz`.

## Evaluate Accuracy / Error Metrics

Ground-truth labels are required for true accuracy/error reporting. Live webcam data does not have ground truth, so it can only report confidence/stability diagnostics, not benchmark accuracy.

For labeled data exported as matching `.npz` arrays:

```powershell
python apps/evaluate_sequence.py --predictions outputs\predictions.npz --ground-truth path\to\ground_truth.npz --pred-key hybrid_3d --gt-key pose_3d --mode 3d
```

For 2D:

```powershell
python apps/evaluate_sequence.py --predictions outputs\predictions.npz --ground-truth path\to\ground_truth.npz --pred-key coco_2d --gt-key coco_2d --mode 2d
```

Implemented metrics:

- `mean_joint_error_px`
- `pck@0.2`
- `mpjpe_mm`
- `p_mpjpe_mm`
- `n_mpjpe_mm`

## Suggested Report Structure

For your report/demo, split the results into:

- Live demo metrics: FPS, detection rate, confidence, temporal jitter.
- 2D benchmark metrics: PCK@0.2 and mean joint error.
- 3D benchmark metrics: MPJPE, P-MPJPE, and N-MPJPE.

That keeps the evaluation technically correct.

## Third-Party References

This project intentionally keeps third-party code separate from the custom implementation.

- Our code lives in `src/bmpose/`.
- Optional source clones belong in `third_party/`.

If you want local copies of the reference repos for inspection, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\clone_references.ps1
```

Please keep attribution intact and do not present cloned external source code as original authorship.

## Primary Sources

- MediaPipe Pose Landmarker guide: [Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- BlazePose GHUM 3D model card: [PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)
- VideoPose3D repository: [facebookresearch/VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- VideoPose3D inference notes: [INFERENCE.md](https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md)
