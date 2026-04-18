"""Shared sit-up analysis engine built on MediaPipe pose landmarks."""

from __future__ import annotations

import copy
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from PoseModule import PoseDetector

CONFIG: Dict[str, float | int] = {
    "visibility_threshold": 0.5,
    "ema_alpha": 0.3,
    "recent_angle_window": 5,
    "state_confirmation_frames": 3,
    "lying_angle": 145.0,
    "sitting_angle": 90.0,
    "hysteresis": 10.0,
    "smooth_delta_limit": 15.0,
    "ankle_motion_ratio_limit": 0.08,
    "pace_window_seconds": 60.0,
    "no_person_warning_seconds": 2.0,
    "hud_padding": 16,
    "hud_width": 370,
    "hud_height": 180,
    "triangle_thickness": 3,
    "angle_arc_radius": 42,
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SideDefinition:
    """Defines the landmarks used for one body side."""

    name: str
    shoulder_id: int
    hip_id: int
    knee_id: int
    ankle_id: int


@dataclass
class SideAnalysis:
    """Holds per-frame analysis for the active body side."""

    definition: SideDefinition
    visibility_score: float
    hip_angle: float
    knee_angle: float
    shoulder_point: Tuple[int, int]
    hip_point: Tuple[int, int]
    knee_point: Tuple[int, int]
    ankle_point: Tuple[int, int]


@dataclass
class RepRecord:
    """Stores information about a completed repetition."""

    duration_seconds: float
    is_good_form: bool
    feedback: str
    completed_at: float


@dataclass
class ExerciseState:
    """Tracks the live metrics displayed to the user."""

    rep_count: int = 0
    good_reps: int = 0
    bad_reps: int = 0
    current_hip_angle: float = 0.0
    current_state: str = "UNKNOWN"
    form_feedback: str = "Great Form!"
    time_elapsed: float = 0.0
    pace_rpm: float = 0.0
    form_score: float = 0.0
    active_side: str = "NONE"
    no_person_duration: float = 0.0
    best_pace: float = 0.0
    rep_history: List[RepRecord] = field(default_factory=list)


@dataclass
class RepQualityTracker:
    """Collects rep-quality evidence from the start to the end of a rep."""

    active: bool = False
    start_time: float = 0.0
    min_angle: float = 180.0
    max_angle: float = 0.0
    smooth_ok: bool = True
    knee_ok: bool = True
    ankles_fixed: bool = True
    reached_sitting: bool = False
    baseline_left_ankle_y: Optional[int] = None
    baseline_right_ankle_y: Optional[int] = None
    last_angle: Optional[float] = None


class SitUpMonitor:
    """Analyze sit-ups, enforce robust state transitions, and render overlays."""

    LEFT_SIDE = SideDefinition("LEFT", 11, 23, 25, 27)
    RIGHT_SIDE = SideDefinition("RIGHT", 12, 24, 26, 28)

    def __init__(self) -> None:
        """Initialize the pose detector, state machine, and session metrics."""

        self.detector = PoseDetector()
        self.state = ExerciseState()
        self._smoothed_angle: Optional[float] = None
        self._recent_angles: Deque[float] = deque(maxlen=int(CONFIG["recent_angle_window"]))
        self._pace_timestamps: Deque[float] = deque()
        self._candidate_state: Optional[str] = None
        self._candidate_frames: int = 0
        self._rep_tracker = RepQualityTracker()
        self._current_side: Optional[SideAnalysis] = None
        self._session_start = time.monotonic()
        self._last_reliable_detection = self._session_start

    def reset(self) -> None:
        """Reset all counting, form, and timing state."""

        LOGGER.info("Resetting sit-up monitor state.")
        self.state = ExerciseState()
        self._smoothed_angle = None
        self._recent_angles.clear()
        self._pace_timestamps.clear()
        self._candidate_state = None
        self._candidate_frames = 0
        self._rep_tracker = RepQualityTracker()
        self._current_side = None
        self._session_start = time.monotonic()
        self._last_reliable_detection = self._session_start

    def process_frame(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
    ) -> Tuple[np.ndarray, ExerciseState]:
        """Process one frame, update metrics, and return an annotated frame."""

        annotated_frame = frame.copy()
        now = time.monotonic()
        landmarks = self.detector.get_landmarks(frame)
        self.detector.draw_landmarks(annotated_frame)

        if not landmarks:
            self._handle_unreliable_frame(now)
            self._update_time_metrics(now)
            self._draw_overlay(annotated_frame, fps)
            return annotated_frame, self.get_state_snapshot()

        side_analysis = self._select_active_side()
        if side_analysis is None:
            self._handle_unreliable_frame(now)
            self._update_time_metrics(now)
            self._draw_overlay(annotated_frame, fps)
            return annotated_frame, self.get_state_snapshot()

        self._last_reliable_detection = now
        self.state.no_person_duration = 0.0
        self._current_side = side_analysis

        smoothed_angle = self._smooth_angle(side_analysis.hip_angle)
        self._recent_angles.append(smoothed_angle)
        average_angle = float(np.mean(self._recent_angles))
        self.state.current_hip_angle = average_angle
        self.state.active_side = side_analysis.definition.name

        self._update_rep_quality(side_analysis, average_angle, frame.shape[0])
        self._advance_state_machine(average_angle, now)
        self._update_feedback()
        self._update_time_metrics(now)

        self._draw_active_geometry(annotated_frame, side_analysis)
        self._draw_overlay(annotated_frame, fps)
        return annotated_frame, self.get_state_snapshot()

    def get_state_snapshot(self) -> ExerciseState:
        """Return a safe copy of the current public state."""

        snapshot = copy.deepcopy(self.state)
        snapshot.rep_history = snapshot.rep_history[-10:]
        return snapshot

    def close(self) -> None:
        """Release detector resources."""

        self.detector.close()

    def _select_active_side(self) -> Optional[SideAnalysis]:
        """Choose the most reliable body side based on visibility and usable angles."""

        left_analysis = self._build_side_analysis(self.LEFT_SIDE)
        right_analysis = self._build_side_analysis(self.RIGHT_SIDE)
        analyses = [analysis for analysis in (left_analysis, right_analysis) if analysis is not None]
        if not analyses:
            return None
        return max(analyses, key=lambda analysis: analysis.visibility_score)

    def _build_side_analysis(self, definition: SideDefinition) -> Optional[SideAnalysis]:
        """Build angle and visibility measurements for one body side."""

        landmark_ids = [
            definition.shoulder_id,
            definition.hip_id,
            definition.knee_id,
            definition.ankle_id,
        ]
        if not self.detector.check_visibility(
            landmark_ids,
            threshold=float(CONFIG["visibility_threshold"]),
        ):
            return None

        shoulder = self.detector.get_landmark(definition.shoulder_id)
        hip = self.detector.get_landmark(definition.hip_id)
        knee = self.detector.get_landmark(definition.knee_id)
        ankle = self.detector.get_landmark(definition.ankle_id)
        if shoulder is None or hip is None or knee is None or ankle is None:
            return None

        hip_angle = self._calculate_angle(
            (shoulder[0], shoulder[1]),
            (hip[0], hip[1]),
            (knee[0], knee[1]),
        )
        knee_angle = self._calculate_angle(
            (hip[0], hip[1]),
            (knee[0], knee[1]),
            (ankle[0], ankle[1]),
        )
        return SideAnalysis(
            definition=definition,
            visibility_score=self.detector.get_visibility_score(landmark_ids),
            hip_angle=hip_angle,
            knee_angle=knee_angle,
            shoulder_point=(shoulder[0], shoulder[1]),
            hip_point=(hip[0], hip[1]),
            knee_point=(knee[0], knee[1]),
            ankle_point=(ankle[0], ankle[1]),
        )

    def _smooth_angle(self, raw_angle: float) -> float:
        """Smooth the active hip angle using an exponential moving average."""

        if self._smoothed_angle is None:
            self._smoothed_angle = raw_angle
        else:
            alpha = float(CONFIG["ema_alpha"])
            self._smoothed_angle = alpha * raw_angle + (1.0 - alpha) * self._smoothed_angle
        return float(self._smoothed_angle)

    def _advance_state_machine(self, angle: float, now: float) -> None:
        """Advance the sit-up state machine using hysteresis and frame confirmation."""

        desired_state = self._get_desired_state(angle)
        if desired_state is None:
            self._candidate_state = None
            self._candidate_frames = 0
            return

        if desired_state == self._candidate_state:
            self._candidate_frames += 1
        else:
            self._candidate_state = desired_state
            self._candidate_frames = 1

        if self._candidate_frames >= int(CONFIG["state_confirmation_frames"]):
            self._commit_transition(desired_state, now)
            self._candidate_state = None
            self._candidate_frames = 0

    def _get_desired_state(self, angle: float) -> Optional[str]:
        """Determine the next plausible state from the current angle."""

        lying_angle = float(CONFIG["lying_angle"])
        sitting_angle = float(CONFIG["sitting_angle"])
        hysteresis = float(CONFIG["hysteresis"])
        current_state = self.state.current_state

        if current_state == "UNKNOWN":
            if angle >= lying_angle:
                return "LYING"
            if angle <= sitting_angle:
                return "SITTING"
            return None
        if current_state == "LYING" and angle <= lying_angle - hysteresis:
            return "RISING"
        if current_state == "RISING":
            if angle <= sitting_angle:
                return "SITTING"
            if angle >= lying_angle:
                return "LYING"
        if current_state == "SITTING" and angle >= sitting_angle + hysteresis:
            return "LOWERING"
        if current_state == "LOWERING":
            if angle >= lying_angle:
                return "LYING"
            if angle <= sitting_angle:
                return "SITTING"
        return None

    def _commit_transition(self, next_state: str, now: float) -> None:
        """Persist a state transition and update rep lifecycle state."""

        previous_state = self.state.current_state
        if previous_state == next_state:
            return

        LOGGER.debug("State transition: %s -> %s", previous_state, next_state)
        self.state.current_state = next_state

        if previous_state == "LYING" and next_state == "RISING":
            self._start_rep(now)
        elif next_state == "SITTING" and self._rep_tracker.active:
            self._rep_tracker.reached_sitting = True
        elif previous_state == "LOWERING" and next_state == "LYING":
            self._complete_rep(now)
        elif previous_state == "RISING" and next_state == "LYING":
            self._rep_tracker = RepQualityTracker()

    def _start_rep(self, now: float) -> None:
        """Begin tracking a new repetition from the lying state."""

        left_ankle = self.detector.get_landmark(self.LEFT_SIDE.ankle_id)
        right_ankle = self.detector.get_landmark(self.RIGHT_SIDE.ankle_id)
        current_angle = self.state.current_hip_angle
        self._rep_tracker = RepQualityTracker(
            active=True,
            start_time=now,
            min_angle=current_angle,
            max_angle=current_angle,
            baseline_left_ankle_y=left_ankle[1] if left_ankle else None,
            baseline_right_ankle_y=right_ankle[1] if right_ankle else None,
            last_angle=current_angle,
        )

    def _update_rep_quality(self, side_analysis: SideAnalysis, angle: float, frame_height: int) -> None:
        """Update rep-quality checks for the current frame."""

        if not self._rep_tracker.active:
            return

        tracker = self._rep_tracker
        tracker.min_angle = min(tracker.min_angle, angle)
        tracker.max_angle = max(tracker.max_angle, angle)
        if tracker.last_angle is not None:
            delta_angle = abs(angle - tracker.last_angle)
            if delta_angle >= float(CONFIG["smooth_delta_limit"]):
                tracker.smooth_ok = False
        tracker.last_angle = angle
        tracker.reached_sitting = tracker.reached_sitting or angle <= float(CONFIG["sitting_angle"])

        if not 70.0 <= side_analysis.knee_angle <= 110.0:
            tracker.knee_ok = False

        left_ankle = self.detector.get_landmark(self.LEFT_SIDE.ankle_id)
        right_ankle = self.detector.get_landmark(self.RIGHT_SIDE.ankle_id)
        ankle_limit = frame_height * float(CONFIG["ankle_motion_ratio_limit"])
        if tracker.baseline_left_ankle_y is not None and left_ankle is not None:
            if abs(left_ankle[1] - tracker.baseline_left_ankle_y) > ankle_limit:
                tracker.ankles_fixed = False
        if tracker.baseline_right_ankle_y is not None and right_ankle is not None:
            if abs(right_ankle[1] - tracker.baseline_right_ankle_y) > ankle_limit:
                tracker.ankles_fixed = False

    def _complete_rep(self, now: float) -> None:
        """Count a full repetition after the lying-to-sitting-to-lying cycle completes."""

        tracker = self._rep_tracker
        if not tracker.active or not tracker.reached_sitting:
            self._rep_tracker = RepQualityTracker()
            return

        full_range_ok = (
            tracker.min_angle < float(CONFIG["sitting_angle"])
            and tracker.max_angle > float(CONFIG["lying_angle"])
        )
        is_good_form = tracker.ankles_fixed and tracker.smooth_ok and tracker.knee_ok and full_range_ok
        feedback = self._feedback_from_quality(
            knee_ok=tracker.knee_ok,
            full_range_ok=full_range_ok,
            movement_ok=tracker.ankles_fixed and tracker.smooth_ok,
        )

        self.state.rep_count += 1
        if is_good_form:
            self.state.good_reps += 1
        else:
            self.state.bad_reps += 1

        record = RepRecord(
            duration_seconds=max(now - tracker.start_time, 0.0),
            is_good_form=is_good_form,
            feedback=feedback,
            completed_at=now,
        )
        self.state.rep_history.append(record)
        self._pace_timestamps.append(now)
        self.state.form_feedback = feedback
        self._rep_tracker = RepQualityTracker()
        self._trim_pace_window(now)
        self._update_score_metrics()

    def _update_feedback(self) -> None:
        """Publish the current form guidance message."""

        if self._current_side is None:
            return

        if self._rep_tracker.active:
            full_range_so_far = (
                self._rep_tracker.min_angle < float(CONFIG["sitting_angle"])
                and self._rep_tracker.max_angle > float(CONFIG["lying_angle"])
            )
            movement_ok = self._rep_tracker.ankles_fixed and self._rep_tracker.smooth_ok
            self.state.form_feedback = self._feedback_from_quality(
                knee_ok=self._rep_tracker.knee_ok and 70.0 <= self._current_side.knee_angle <= 110.0,
                full_range_ok=full_range_so_far or self.state.current_state in {"RISING", "SITTING"},
                movement_ok=movement_ok,
            )
            return

        if self.state.rep_history and self.state.current_state in {"LYING", "LOWERING"}:
            self.state.form_feedback = self.state.rep_history[-1].feedback
            return

        if not 70.0 <= self._current_side.knee_angle <= 110.0:
            self.state.form_feedback = "Keep Knees Bent"
        else:
            self.state.form_feedback = "Great Form!"

    def _feedback_from_quality(
        self,
        knee_ok: bool,
        full_range_ok: bool,
        movement_ok: bool,
    ) -> str:
        """Map quality signals to one of the required feedback messages."""

        if not knee_ok:
            return "Keep Knees Bent"
        if not full_range_ok:
            return "Full Range Needed"
        if not movement_ok:
            return "Control the Movement"
        return "Great Form!"

    def _update_time_metrics(self, now: float) -> None:
        """Refresh session duration, pace, and form score metrics."""

        self.state.time_elapsed = max(now - self._session_start, 0.0)
        self._trim_pace_window(now)
        self.state.pace_rpm = float(len(self._pace_timestamps))
        self.state.best_pace = max(self.state.best_pace, self.state.pace_rpm)
        self._update_score_metrics()

    def _update_score_metrics(self) -> None:
        """Update aggregate quality metrics after any relevant change."""

        if self.state.rep_count > 0:
            self.state.form_score = (self.state.good_reps / self.state.rep_count) * 100.0
        else:
            self.state.form_score = 0.0

    def _trim_pace_window(self, now: float) -> None:
        """Keep only repetition timestamps inside the rolling pace window."""

        window = float(CONFIG["pace_window_seconds"])
        while self._pace_timestamps and now - self._pace_timestamps[0] > window:
            self._pace_timestamps.popleft()

    def _handle_unreliable_frame(self, now: float) -> None:
        """Update visibility-loss timing without counting unreliable data."""

        self.state.no_person_duration = max(now - self._last_reliable_detection, 0.0)
        if self.state.no_person_duration > float(CONFIG["no_person_warning_seconds"]):
            self._smoothed_angle = None
            self._recent_angles.clear()
            self._candidate_state = None
            self._candidate_frames = 0
            self._rep_tracker = RepQualityTracker()
            self._current_side = None
            self.state.active_side = "NONE"

    def _draw_active_geometry(self, frame: np.ndarray, side_analysis: SideAnalysis) -> None:
        """Draw the active triangle and the current hip-angle arc."""

        triangle_points = np.array(
            [
                side_analysis.shoulder_point,
                side_analysis.hip_point,
                side_analysis.knee_point,
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            frame,
            [triangle_points],
            isClosed=True,
            color=(0, 255, 255),
            thickness=int(CONFIG["triangle_thickness"]),
        )
        self._draw_angle_arc(
            frame,
            side_analysis.shoulder_point,
            side_analysis.hip_point,
            side_analysis.knee_point,
        )

    def _draw_angle_arc(
        self,
        frame: np.ndarray,
        start_point: Tuple[int, int],
        center_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> None:
        """Draw a cyan arc illustrating the active hip angle."""

        angle_start = self._normalize_cv_angle(start_point, center_point)
        angle_end = self._normalize_cv_angle(end_point, center_point)
        sweep = (angle_end - angle_start) % 360
        if sweep > 180:
            angle_start, angle_end = angle_end, angle_start
        cv2.ellipse(
            frame,
            center_point,
            (int(CONFIG["angle_arc_radius"]), int(CONFIG["angle_arc_radius"])),
            0,
            angle_start,
            angle_end,
            (255, 255, 0),
            3,
        )
        cv2.circle(frame, center_point, 5, (255, 255, 0), cv2.FILLED)

    def _draw_overlay(self, frame: np.ndarray, fps: float) -> None:
        """Draw the shared HUD overlay, FPS, and missing-person warning."""

        overlay = frame.copy()
        padding = int(CONFIG["hud_padding"])
        hud_width = int(CONFIG["hud_width"])
        hud_height = int(CONFIG["hud_height"])
        cv2.rectangle(
            overlay,
            (padding, padding),
            (padding + hud_width, padding + hud_height),
            (20, 20, 20),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        angle_value = "--" if self.state.current_hip_angle <= 0 else f"{self.state.current_hip_angle:.1f}"
        hud_lines = [
            f"REPS: {self.state.rep_count}",
            f"ANGLE: {angle_value}°",
            f"STATE: {self.state.current_state}",
            f"FORM: {self.state.form_feedback}",
            f"PACE: {self.state.pace_rpm:.1f} rpm",
            f"SCORE: {self.state.form_score:.1f}%",
        ]
        line_y = padding + 28
        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (padding + 12, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            line_y += 26

        fps_text = f"FPS: {fps:.1f}"
        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top_right_x = frame.shape[1] - text_size[0] - 24
        cv2.rectangle(
            frame,
            (top_right_x - 10, 14),
            (frame.shape[1] - 14, 46),
            (20, 20, 20),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            fps_text,
            (top_right_x, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.state.no_person_duration > float(CONFIG["no_person_warning_seconds"]):
            warning = "No Person Detected"
            text_size, _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            origin = (
                max((frame.shape[1] - text_size[0]) // 2, 20),
                max(frame.shape[0] // 2, 40),
            )
            cv2.putText(
                frame,
                warning,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

    def _calculate_angle(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
        c: Tuple[int, int],
    ) -> float:
        """Use the required vector-based angle formula for 2D points."""

        ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
        bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return float(angle)

    def _normalize_cv_angle(self, point: Tuple[int, int], center: Tuple[int, int]) -> float:
        """Convert an image-space vector to a normalized OpenCV ellipse angle."""

        radians = math.atan2(point[1] - center[1], point[0] - center[0])
        return float((math.degrees(radians) + 360.0) % 360.0)

    def __del__(self) -> None:
        """Release resources when the monitor is collected."""

        try:
            self.close()
        except (AttributeError, RuntimeError):
            LOGGER.debug("Sit-up monitor cleanup skipped during interpreter shutdown.")
