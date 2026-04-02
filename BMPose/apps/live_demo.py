from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bmpose.constants import DEFAULT_MEDIAPIPE_MODEL, DEFAULT_VIDEOPOSE_CHECKPOINT
from bmpose.pipeline import HybridPosePipeline
from bmpose.visualization import render_result


class NullDisplay:
    def update(self, _frame) -> bool:
        return True

    def close(self) -> None:
        return None


class OpenCVDisplay:
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name

    def update(self, frame) -> bool:
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def close(self) -> None:
        cv2.destroyAllWindows()


class TkDisplay:
    def __init__(self, window_name: str) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._image_module = Image
        self._imagetk_module = ImageTk
        self._running = True
        self._root = tk.Tk()
        self._root.title(window_name)
        self._root.protocol("WM_DELETE_WINDOW", self._request_close)
        self._root.bind("<Escape>", lambda _event: self._request_close())
        self._root.bind("<KeyPress-q>", lambda _event: self._request_close())
        self._label = tk.Label(self._root)
        self._label.pack()
        self._photo = None

    def _request_close(self) -> None:
        self._running = False

    def update(self, frame) -> bool:
        if not self._running:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._image_module.fromarray(rgb)
        self._photo = self._imagetk_module.PhotoImage(image=image)
        self._label.configure(image=self._photo)
        self._root.update_idletasks()
        self._root.update()
        return self._running

    def close(self) -> None:
        if getattr(self, "_root", None) is not None:
            try:
                self._root.destroy()
            except self._tk.TclError:
                pass


def create_display(backend: str):
    if backend == "none":
        print("Live display disabled. Use Ctrl+C to stop.")
        return NullDisplay()

    if backend == "opencv":
        return OpenCVDisplay("BMPose Hybrid Demo")

    if backend == "tk":
        return TkDisplay("BMPose Hybrid Demo")

    try:
        return TkDisplay("BMPose Hybrid Demo")
    except Exception:
        return OpenCVDisplay("BMPose Hybrid Demo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BMPose live webcam demo.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--mediapipe-model", type=Path, default=DEFAULT_MEDIAPIPE_MODEL)
    parser.add_argument("--videopose-checkpoint", type=Path, default=DEFAULT_VIDEOPOSE_CHECKPOINT)
    parser.add_argument("--save", type=Path, default=None, help="Optional output video path.")
    parser.add_argument(
        "--display-backend",
        choices=["auto", "tk", "opencv", "none"],
        default="auto",
        help="Viewer backend. Use 'tk' if OpenCV HighGUI is unavailable.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    display = create_display(args.display_backend)
    last_tick = time.perf_counter()
    fps = 0.0
    start_time = time.perf_counter()

    try:
        with HybridPosePipeline(
            mediapipe_model_path=args.mediapipe_model,
            videopose_checkpoint_path=args.videopose_checkpoint if args.videopose_checkpoint.exists() else None,
        ) as pipeline:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_ms = int((time.perf_counter() - start_time) * 1000.0)
                result = pipeline.process_live_frame(frame, timestamp_ms)

                now = time.perf_counter()
                dt = max(now - last_tick, 1e-6)
                fps = 1.0 / dt
                last_tick = now

                rendered = render_result(frame, result, fps=fps)

                if writer is None and args.save is not None:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.save),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(10.0, min(60.0, fps if fps > 0 else 30.0)),
                        (rendered.shape[1], rendered.shape[0]),
                    )

                if writer is not None:
                    writer.write(rendered)

                try:
                    keep_running = display.update(rendered)
                except cv2.error as exc:
                    if args.display_backend in {"auto", "opencv"}:
                        display.close()
                        if args.display_backend == "opencv":
                            raise RuntimeError(
                                "OpenCV display backend is unavailable in this environment. "
                                "Run with --display-backend tk."
                            ) from exc
                        print("OpenCV HighGUI is unavailable. Falling back to Tk display.")
                        display = TkDisplay("BMPose Hybrid Demo")
                        keep_running = display.update(rendered)
                    else:
                        raise

                if not keep_running:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        display.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
