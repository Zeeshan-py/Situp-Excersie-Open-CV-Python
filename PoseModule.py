"""Production pose detection utilities powered by MediaPipe pose backends."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

CONFIG: Dict[str, float | int | bool | str] = {
    "model_complexity": 1,
    "smooth_landmarks": True,
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.6,
    "default_visibility_threshold": 0.5,
    "task_model_url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}

LOGGER = logging.getLogger(__name__)

LandmarkData = Tuple[int, int, float, float]


@dataclass(frozen=True)
class DrawingSpecConfig:
    """Encapsulates landmark drawing colors and sizes."""

    landmark_color: Tuple[int, int, int] = (0, 255, 0)
    connection_color: Tuple[int, int, int] = (0, 180, 0)
    landmark_radius: int = 3
    landmark_thickness: int = 2
    connection_thickness: int = 2


def calculate_angle(
    a: Tuple[int, int],
    b: Tuple[int, int],
    c: Tuple[int, int],
) -> float:
    """Calculate the angle in degrees at point ``b`` using stable vector math."""

    ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angle)


class PoseDetector:
    """Wrap MediaPipe pose estimation with a resilient, landmark-centric API."""

    def __init__(self) -> None:
        """Initialize the best available MediaPipe backend and drawing helpers."""

        self._backend = "solutions"
        self._mp_pose = None
        self._mp_drawing = None
        self._task_drawing_utils = None
        self._task_connections = None
        self._task_image_cls = None
        self._task_image_format = None
        self._pose = self._create_backend()
        self._drawing_config = DrawingSpecConfig()
        self._results: Optional[object] = None
        self._landmarks: Dict[int, LandmarkData] = {}
        self._last_frame_shape: Optional[Tuple[int, int]] = None

    def get_landmarks(self, frame: np.ndarray) -> Dict[int, LandmarkData]:
        """Run pose estimation and return landmark id to ``(x, y, z, visibility)``."""

        if frame is None or frame.size == 0:
            LOGGER.warning("Received an empty frame for landmark extraction.")
            self._results = None
            self._landmarks = {}
            self._last_frame_shape = None
            return {}

        frame_height, frame_width = frame.shape[:2]
        self._last_frame_shape = (frame_width, frame_height)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._results = self._run_inference(rgb_frame)
        self._landmarks = {}

        normalized_landmarks, world_landmarks = self._extract_pose_sets()
        if normalized_landmarks is None:
            return {}

        for landmark_id, landmark in enumerate(normalized_landmarks):
            world_landmark = (
                world_landmarks[landmark_id]
                if world_landmarks is not None and landmark_id < len(world_landmarks)
                else None
            )
            visibility = getattr(landmark, "visibility", None)
            if visibility is None:
                visibility = getattr(landmark, "presence", 0.0)
            z_value = world_landmark.z if world_landmark is not None else landmark.z
            x_coord = int(np.clip(landmark.x * frame_width, 0, max(frame_width - 1, 0)))
            y_coord = int(np.clip(landmark.y * frame_height, 0, max(frame_height - 1, 0)))
            self._landmarks[landmark_id] = (
                x_coord,
                y_coord,
                float(z_value),
                float(visibility),
            )

        return dict(self._landmarks)

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw the MediaPipe skeleton on the provided frame."""

        if frame is None or frame.size == 0:
            return frame

        if not self._results or not self._landmarks:
            self.get_landmarks(frame)

        landmark_spec = self._create_landmark_spec()
        connection_spec = self._create_connection_spec()
        normalized_landmarks, _ = self._extract_pose_sets()
        if normalized_landmarks is None:
            return frame

        if self._backend == "solutions":
            self._mp_drawing.draw_landmarks(
                frame,
                self._results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec,
            )
        else:
            self._task_drawing_utils.draw_landmarks(
                frame,
                normalized_landmarks,
                self._task_connections,
                landmark_spec,
                connection_spec,
            )

        return frame

    def get_angle(self, frame: np.ndarray, p1_id: int, p2_id: int, p3_id: int) -> float:
        """Return the angle in degrees formed by three landmarks on the current frame."""

        if not self._landmarks and frame is not None and frame.size > 0:
            self.get_landmarks(frame)

        point_a = self._get_point(p1_id)
        point_b = self._get_point(p2_id)
        point_c = self._get_point(p3_id)
        if point_a is None or point_b is None or point_c is None:
            return 0.0

        return calculate_angle(point_a, point_b, point_c)

    def check_visibility(self, landmark_ids: List[int], threshold: float = 0.5) -> bool:
        """Return ``True`` only when all requested landmarks exceed the visibility threshold."""

        for landmark_id in landmark_ids:
            landmark = self._landmarks.get(landmark_id)
            if landmark is None or landmark[3] < threshold:
                return False
        return True

    def get_visibility_score(self, landmark_ids: List[int]) -> float:
        """Return the average visibility of the requested landmarks."""

        visibilities = [
            self._landmarks[landmark_id][3]
            for landmark_id in landmark_ids
            if landmark_id in self._landmarks
        ]
        if not visibilities:
            return 0.0
        return float(sum(visibilities) / len(visibilities))

    def get_landmark(self, landmark_id: int) -> Optional[LandmarkData]:
        """Return a landmark tuple when available."""

        return self._landmarks.get(landmark_id)

    def close(self) -> None:
        """Release MediaPipe resources."""

        if self._pose is not None and hasattr(self._pose, "close"):
            self._pose.close()
            self._pose = None

    def _run_inference(self, rgb_frame: np.ndarray) -> object:
        """Execute pose inference for the current backend."""

        if self._backend == "solutions":
            rgb_frame.flags.writeable = False
            result = self._pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            return result

        task_image = self._task_image_cls(
            image_format=self._task_image_format.SRGB,
            data=rgb_frame,
        )
        return self._pose.detect(task_image)

    def _extract_pose_sets(self) -> Tuple[Optional[object], Optional[object]]:
        """Return normalized and world landmark sequences for the current result."""

        if self._results is None:
            return None, None

        if self._backend == "solutions":
            if not getattr(self._results, "pose_landmarks", None):
                return None, None
            normalized_landmarks = self._results.pose_landmarks.landmark
            world_landmarks = (
                self._results.pose_world_landmarks.landmark
                if getattr(self._results, "pose_world_landmarks", None)
                else None
            )
            return normalized_landmarks, world_landmarks

        if not getattr(self._results, "pose_landmarks", None):
            return None, None
        normalized_landmarks = self._results.pose_landmarks[0]
        world_landmarks = (
            self._results.pose_world_landmarks[0]
            if getattr(self._results, "pose_world_landmarks", None)
            else None
        )
        return normalized_landmarks, world_landmarks

    def _create_backend(self) -> object:
        """Create the best available MediaPipe pose backend."""

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._mp_pose = mp.solutions.pose
            self._mp_drawing = mp.solutions.drawing_utils
            return self._mp_pose.Pose(
                model_complexity=int(CONFIG["model_complexity"]),
                smooth_landmarks=bool(CONFIG["smooth_landmarks"]),
                min_detection_confidence=float(CONFIG["min_detection_confidence"]),
                min_tracking_confidence=float(CONFIG["min_tracking_confidence"]),
            )

        LOGGER.info("MediaPipe solutions API unavailable; using Pose Landmarker Tasks backend.")
        self._backend = "tasks"
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import drawing_utils as task_drawing_utils
        from mediapipe.tasks.python.vision import pose_landmarker
        from mediapipe.tasks.python.vision.core.image import Image
        from mediapipe.tasks.python.vision.core.image import ImageFormat

        model_path = self._ensure_task_model()
        self._task_drawing_utils = task_drawing_utils
        self._task_connections = pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS
        self._task_image_cls = Image
        self._task_image_format = ImageFormat
        options = pose_landmarker.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_poses=1,
            min_pose_detection_confidence=float(CONFIG["min_detection_confidence"]),
            min_pose_presence_confidence=float(CONFIG["min_detection_confidence"]),
            min_tracking_confidence=float(CONFIG["min_tracking_confidence"]),
        )
        return pose_landmarker.PoseLandmarker.create_from_options(options)

    def _ensure_task_model(self) -> Path:
        """Download the official pose landmarker task model when needed."""

        cache_dir = Path.home() / ".situp_monitor"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "pose_landmarker_full.task"
        if not model_path.exists():
            LOGGER.info("Downloading MediaPipe pose model to %s", model_path)
            urllib.request.urlretrieve(str(CONFIG["task_model_url"]), model_path)
        return model_path

    def _create_landmark_spec(self) -> object:
        """Build a landmark drawing spec for the active backend."""

        if self._backend == "solutions":
            return self._mp_drawing.DrawingSpec(
                color=self._drawing_config.landmark_color,
                thickness=self._drawing_config.landmark_thickness,
                circle_radius=self._drawing_config.landmark_radius,
            )
        return self._task_drawing_utils.DrawingSpec(
            color=self._drawing_config.landmark_color,
            thickness=self._drawing_config.landmark_thickness,
            circle_radius=self._drawing_config.landmark_radius,
        )

    def _create_connection_spec(self) -> object:
        """Build a connection drawing spec for the active backend."""

        if self._backend == "solutions":
            return self._mp_drawing.DrawingSpec(
                color=self._drawing_config.connection_color,
                thickness=self._drawing_config.connection_thickness,
                circle_radius=self._drawing_config.landmark_radius,
            )
        return self._task_drawing_utils.DrawingSpec(
            color=self._drawing_config.connection_color,
            thickness=self._drawing_config.connection_thickness,
            circle_radius=self._drawing_config.landmark_radius,
        )

    def _get_point(self, landmark_id: int) -> Optional[Tuple[int, int]]:
        """Return a 2D point for a landmark id when available."""

        landmark = self._landmarks.get(landmark_id)
        if landmark is None:
            return None
        return (landmark[0], landmark[1])

    def __del__(self) -> None:
        """Best-effort cleanup for MediaPipe resources."""

        try:
            self.close()
        except (AttributeError, RuntimeError):
            LOGGER.debug("Pose detector cleanup skipped during interpreter shutdown.")
