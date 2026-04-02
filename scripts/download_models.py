from __future__ import annotations

import argparse
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "models"


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    url: str
    output_path: Path


ARTIFACTS = {
    "mediapipe": ModelArtifact(
        name="MediaPipe Pose Landmarker Heavy",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        ),
        output_path=MODEL_DIR / "pose_landmarker_heavy.task",
    ),
    "videopose": ModelArtifact(
        name="VideoPose3D pretrained Human3.6M COCO checkpoint",
        url="https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin",
        output_path=MODEL_DIR / "pretrained_h36m_detectron_coco.bin",
    ),
}


def download_file(artifact: ModelArtifact, force: bool) -> None:
    artifact.output_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact.output_path.exists() and not force:
        print(f"[skip] {artifact.name} already exists at {artifact.output_path}")
        return

    temp_path = artifact.output_path.with_suffix(artifact.output_path.suffix + ".part")
    print(f"[download] {artifact.name}")
    print(f"           {artifact.url}")

    with urllib.request.urlopen(artifact.url) as response, temp_path.open("wb") as destination:
        shutil.copyfileobj(response, destination)

    temp_path.replace(artifact.output_path)
    print(f"[done] {artifact.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download pretrained model assets for BMPose.")
    parser.add_argument(
        "--only",
        choices=["mediapipe", "videopose", "all"],
        default="all",
        help="Choose a specific artifact or download both.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    selected = ARTIFACTS.values() if args.only == "all" else [ARTIFACTS[args.only]]
    for artifact in selected:
        download_file(artifact, force=args.force)

    print("All requested artifacts are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
