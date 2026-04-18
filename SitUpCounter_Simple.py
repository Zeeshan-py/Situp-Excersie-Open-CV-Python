"""Command-line sit-up monitor with webcam and video-file support."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from SitUpExercise import SitUpMonitor

CONFIG = {
    "window_name": "Sit-Up Counter",
    "camera_width": 1280,
    "camera_height": 720,
    "camera_fps": 30,
    "fps_ema_alpha": 0.2,
    "screenshot_prefix": "situp_snapshot",
}

LOGGER = logging.getLogger(__name__)


@dataclass
class CaptureSession:
    """Wrap an OpenCV capture with metadata and cleanup helpers."""

    capture: cv2.VideoCapture
    source_label: str
    is_camera: bool

    def close(self) -> None:
        """Release the capture when it is open."""

        if self.capture.isOpened():
            self.capture.release()

    def __del__(self) -> None:
        """Release resources during garbage collection as a fallback."""

        try:
            self.close()
        except (AttributeError, cv2.error):
            LOGGER.debug("Capture cleanup skipped during interpreter shutdown.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Production sit-up counter using MediaPipe Pose.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index like 0 or a video file path.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure application logging."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def open_capture(source: str) -> CaptureSession:
    """Open a webcam or video file source with validation."""

    if source.isdigit():
        camera_index = int(source)
        capture = cv2.VideoCapture(camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(CONFIG["camera_width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CONFIG["camera_height"]))
        capture.set(cv2.CAP_PROP_FPS, int(CONFIG["camera_fps"]))
        if not capture.isOpened():
            raise ValueError(f"Unable to open camera index {camera_index}.")
        LOGGER.info("Opened camera index %s.", camera_index)
        return CaptureSession(capture=capture, source_label=f"Camera {camera_index}", is_camera=True)

    video_path = Path(source).expanduser().resolve()
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    LOGGER.info("Opened video file %s.", video_path)
    return CaptureSession(capture=capture, source_label=str(video_path), is_camera=False)


def save_screenshot(frame) -> Optional[Path]:
    """Save a screenshot of the current annotated frame."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path.cwd() / f"{CONFIG['screenshot_prefix']}_{timestamp}.png"
    if cv2.imwrite(str(output_path), frame):
        LOGGER.info("Saved screenshot to %s", output_path)
        return output_path
    LOGGER.error("Failed to save screenshot to %s", output_path)
    return None


def main() -> int:
    """Run the standalone sit-up monitor."""

    configure_logging()
    args = parse_args()

    try:
        session = open_capture(args.source)
    except ValueError as error:
        LOGGER.error("%s", error)
        return 1

    monitor = SitUpMonitor()
    fps_value = 0.0
    previous_tick: Optional[float] = None

    LOGGER.info("Controls: Q or ESC quit | R reset | S save screenshot")
    LOGGER.info("For best accuracy use a side view with the full body visible.")

    try:
        while True:
            success, frame = session.capture.read()
            if not success:
                if session.is_camera:
                    LOGGER.error("Camera feed stopped unexpectedly. The camera may be disconnected.")
                else:
                    LOGGER.info("Reached the end of the video file.")
                break

            tick = cv2.getTickCount() / cv2.getTickFrequency()
            if previous_tick is not None:
                instant_fps = 1.0 / max(tick - previous_tick, 1e-6)
                alpha = float(CONFIG["fps_ema_alpha"])
                fps_value = instant_fps if fps_value == 0.0 else alpha * instant_fps + (1.0 - alpha) * fps_value
            previous_tick = tick

            annotated_frame, _ = monitor.process_frame(frame, fps=fps_value)
            cv2.imshow(str(CONFIG["window_name"]), annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                LOGGER.info("Exiting sit-up counter.")
                break
            if key == ord("r"):
                monitor.reset()
                LOGGER.info("Counter reset.")
            if key == ord("s"):
                save_screenshot(annotated_frame)
    finally:
        monitor.close()
        session.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
