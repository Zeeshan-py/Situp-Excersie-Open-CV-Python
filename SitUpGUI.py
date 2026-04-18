"""Tkinter sit-up monitor with threaded video processing and a live dashboard."""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
from PIL import Image, ImageTk

from SitUpExercise import ExerciseState, RepRecord, SitUpMonitor

CONFIG = {
    "window_title": "Sit-Up Monitor v2.0",
    "window_size": "980x620",
    "video_width": 640,
    "video_height": 480,
    "panel_width": 280,
    "queue_size": 2,
    "queue_poll_ms": 15,
    "camera_width": 1280,
    "camera_height": 720,
    "camera_fps": 30,
    "fps_ema_alpha": 0.2,
    "bg": "#111827",
    "panel_bg": "#F3F4F6",
    "card_bg": "#FFFFFF",
    "accent": "#0F766E",
    "warning": "#D97706",
    "danger": "#B91C1C",
}

LOGGER = logging.getLogger(__name__)


@dataclass
class CaptureSession:
    """Holds capture metadata and release helpers."""

    capture: cv2.VideoCapture
    source_label: str
    is_camera: bool

    def close(self) -> None:
        """Release the underlying video capture."""

        if self.capture.isOpened():
            self.capture.release()

    def __del__(self) -> None:
        """Best-effort cleanup when the object is destroyed."""

        try:
            self.close()
        except (AttributeError, cv2.error):
            LOGGER.debug("GUI capture cleanup skipped during interpreter shutdown.")


class SitUpGUI:
    """Render the threaded GUI dashboard for live sit-up monitoring."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize application state and construct the interface."""

        self.root = root
        self.root.title(str(CONFIG["window_title"]))
        self.root.geometry(str(CONFIG["window_size"]))
        self.root.configure(bg=str(CONFIG["bg"]))

        self.monitor = SitUpMonitor()
        self.selected_source = "0"
        self.capture_session: Optional[CaptureSession] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.is_paused = False
        self.frame_queue: "queue.Queue[dict]" = queue.Queue(maxsize=int(CONFIG["queue_size"]))
        self.current_state = ExerciseState()
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self.status_var = tk.StringVar(value="Source: Camera 0")
        self.feedback_var = tk.StringVar(value="Great Form!")
        self.summary_vars = {
            "reps": tk.StringVar(value="--"),
            "form": tk.StringVar(value="--"),
            "pace": tk.StringVar(value="--"),
            "time": tk.StringVar(value="--"),
        }
        self.stat_vars = {
            "REPS": tk.StringVar(value="0"),
            "HIP ANGLE": tk.StringVar(value="0.0°"),
            "STATE": tk.StringVar(value="UNKNOWN"),
            "FORM SCORE": tk.StringVar(value="0.0%"),
            "PACE": tk.StringVar(value="0.0 rpm"),
            "TIME": tk.StringVar(value="00:00"),
        }

        self._build_layout()
        self.root.after(int(CONFIG["queue_poll_ms"]), self._poll_frame_queue)

    def _build_layout(self) -> None:
        """Create all Tkinter widgets."""

        main_frame = tk.Frame(self.root, bg=str(CONFIG["bg"]))
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left_panel = tk.Frame(main_frame, bg=str(CONFIG["bg"]))
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(
            left_panel,
            width=int(CONFIG["video_width"]),
            height=int(CONFIG["video_height"]),
            bg="black",
            bd=0,
            highlightthickness=0,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.create_text(
            int(CONFIG["video_width"]) // 2,
            int(CONFIG["video_height"]) // 2,
            text="Select a source and press Start",
            fill="white",
            font=("Segoe UI", 18, "bold"),
        )

        right_panel = tk.Frame(
            main_frame,
            bg=str(CONFIG["panel_bg"]),
            width=int(CONFIG["panel_width"]),
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right_panel.pack_propagate(False)

        header = tk.Label(
            right_panel,
            text="💪 Sit-Up Monitor",
            font=("Segoe UI", 18, "bold"),
            bg=str(CONFIG["panel_bg"]),
            fg="#111827",
        )
        header.pack(anchor="w", padx=14, pady=(12, 8))

        stats_frame = tk.Frame(right_panel, bg=str(CONFIG["panel_bg"]))
        stats_frame.pack(fill=tk.X, padx=12)
        for column in range(2):
            stats_frame.grid_columnconfigure(column, weight=1)

        stat_titles = ["REPS", "HIP ANGLE", "STATE", "FORM SCORE", "PACE", "TIME"]
        for index, title in enumerate(stat_titles):
            row = index // 2
            column = index % 2
            self._create_stat_box(stats_frame, title, row, column)

        feedback_frame = tk.LabelFrame(
            right_panel,
            text="Form Feedback",
            bg=str(CONFIG["panel_bg"]),
            fg="#111827",
            font=("Segoe UI", 10, "bold"),
        )
        feedback_frame.pack(fill=tk.X, padx=12, pady=(10, 8))
        self.feedback_label = tk.Label(
            feedback_frame,
            textvariable=self.feedback_var,
            font=("Segoe UI", 11, "bold"),
            bg="#D1FAE5",
            fg="#065F46",
            padx=10,
            pady=10,
        )
        self.feedback_label.pack(fill=tk.X, padx=8, pady=8)

        history_frame = tk.LabelFrame(
            right_panel,
            text="Rep History",
            bg=str(CONFIG["panel_bg"]),
            fg="#111827",
            font=("Segoe UI", 10, "bold"),
        )
        history_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.history_canvas = tk.Canvas(
            history_frame,
            width=240,
            height=120,
            bg=str(CONFIG["card_bg"]),
            bd=0,
            highlightthickness=0,
        )
        self.history_canvas.pack(fill=tk.X, padx=8, pady=8)
        self._draw_rep_history([])

        controls_frame = tk.LabelFrame(
            right_panel,
            text="Controls",
            bg=str(CONFIG["panel_bg"]),
            fg="#111827",
            font=("Segoe UI", 10, "bold"),
        )
        controls_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        for column in range(6):
            controls_frame.grid_columnconfigure(column, weight=1)

        buttons = [
            ("📹 Camera", self.select_camera),
            ("📁 Video", self.select_video),
            ("▶ Start", self.start_monitoring),
            ("⏸ Pause", self.toggle_pause),
            ("⏹ Stop", self.stop_monitoring),
            ("🔄 Reset", self.reset_monitoring),
        ]
        self.control_buttons = {}
        for column, (label, command) in enumerate(buttons):
            button = tk.Button(
                controls_frame,
                text=label,
                command=command,
                font=("Segoe UI", 8, "bold"),
                bg=str(CONFIG["card_bg"]),
                fg="#111827",
                relief=tk.FLAT,
                padx=2,
                pady=8,
            )
            button.grid(row=0, column=column, padx=2, pady=8, sticky="ew")
            self.control_buttons[label] = button

        summary_frame = tk.LabelFrame(
            right_panel,
            text="Session Summary",
            bg=str(CONFIG["panel_bg"]),
            fg="#111827",
            font=("Segoe UI", 10, "bold"),
        )
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        for key, label in (
            ("reps", "Total Reps"),
            ("form", "Good Form %"),
            ("pace", "Best Pace"),
            ("time", "Session Duration"),
        ):
            row = tk.Frame(summary_frame, bg=str(CONFIG["panel_bg"]))
            row.pack(fill=tk.X, padx=8, pady=4)
            tk.Label(
                row,
                text=label,
                font=("Segoe UI", 10),
                bg=str(CONFIG["panel_bg"]),
                fg="#374151",
            ).pack(side=tk.LEFT)
            tk.Label(
                row,
                textvariable=self.summary_vars[key],
                font=("Segoe UI", 10, "bold"),
                bg=str(CONFIG["panel_bg"]),
                fg="#111827",
            ).pack(side=tk.RIGHT)

        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            bg="#1F2937",
            fg="white",
            padx=12,
            pady=6,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._set_running_controls(False)

    def _create_stat_box(self, parent: tk.Frame, title: str, row: int, column: int) -> None:
        """Create one dashboard statistic box."""

        card = tk.Frame(parent, bg=str(CONFIG["card_bg"]), bd=0, highlightthickness=0)
        card.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")
        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 9, "bold"),
            bg=str(CONFIG["card_bg"]),
            fg="#6B7280",
        ).pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(
            card,
            textvariable=self.stat_vars[title],
            font=("Segoe UI", 15, "bold"),
            bg=str(CONFIG["card_bg"]),
            fg="#111827",
        ).pack(anchor="w", padx=10, pady=(0, 10))

    def select_camera(self) -> None:
        """Choose the default webcam as the active source."""

        if self.is_running:
            messagebox.showwarning("Session running", "Stop the current session before changing sources.")
            return
        self.selected_source = "0"
        self.status_var.set("Source: Camera 0")

    def select_video(self) -> None:
        """Open a file dialog and set a video file as the active source."""

        if self.is_running:
            messagebox.showwarning("Session running", "Stop the current session before changing sources.")
            return
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if file_path:
            self.selected_source = file_path
            self.status_var.set(f"Source: {Path(file_path).name}")

    def start_monitoring(self) -> None:
        """Start the threaded video-processing session."""

        if self.is_running:
            return

        try:
            self.capture_session = self._open_capture(self.selected_source)
        except ValueError as error:
            messagebox.showerror("Video source error", str(error))
            self.status_var.set(str(error))
            return

        self.stop_event.clear()
        self.is_running = True
        self.is_paused = False
        self._set_running_controls(True)
        self.status_var.set(f"Running: {self.capture_session.source_label}")
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def toggle_pause(self) -> None:
        """Pause or resume frame processing."""

        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self.control_buttons["⏸ Pause"].config(text="▶ Resume" if self.is_paused else "⏸ Pause")
        self.status_var.set("Paused" if self.is_paused else f"Running: {self.capture_session.source_label}")

    def stop_monitoring(self) -> None:
        """Stop processing and release capture resources."""

        self.stop_event.set()
        self.is_running = False
        self.is_paused = False
        self.control_buttons["⏸ Pause"].config(text="⏸ Pause")
        if self.processing_thread and self.processing_thread.is_alive() and threading.current_thread() is not self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.processing_thread = None
        if self.capture_session is not None:
            self.capture_session.close()
            self.capture_session = None
        self._set_running_controls(False)
        self._update_summary(self.current_state)
        self.status_var.set("Stopped")

    def reset_monitoring(self) -> None:
        """Reset monitor metrics and clear summary visuals."""

        self.monitor.reset()
        self.current_state = ExerciseState()
        self._update_dashboard(self.current_state)
        self._update_summary(self.current_state)
        self._draw_rep_history([])
        self.status_var.set("Counters reset")

    def _processing_loop(self) -> None:
        """Read, analyze, and enqueue frames from the selected source."""

        fps_value = 0.0
        previous_tick: Optional[float] = None
        assert self.capture_session is not None

        try:
            while not self.stop_event.is_set():
                if self.is_paused:
                    time.sleep(0.05)
                    continue

                success, frame = self.capture_session.capture.read()
                if not success:
                    message = (
                        "Camera feed stopped unexpectedly."
                        if self.capture_session.is_camera
                        else "Video playback completed."
                    )
                    self._enqueue_packet({"kind": "event", "message": message})
                    break

                tick = time.perf_counter()
                if previous_tick is not None:
                    instant_fps = 1.0 / max(tick - previous_tick, 1e-6)
                    alpha = float(CONFIG["fps_ema_alpha"])
                    fps_value = instant_fps if fps_value == 0.0 else alpha * instant_fps + (1.0 - alpha) * fps_value
                previous_tick = tick

                annotated_frame, state = self.monitor.process_frame(frame, fps=fps_value)
                self._enqueue_packet({"kind": "frame", "frame": annotated_frame, "state": state})
        except cv2.error as error:
            LOGGER.exception("OpenCV processing failed.")
            self._enqueue_packet({"kind": "event", "message": f"OpenCV error: {error}"})
        finally:
            if self.capture_session is not None:
                self.capture_session.close()

    def _enqueue_packet(self, packet: dict) -> None:
        """Put a packet into the GUI queue, dropping stale frames when needed."""

        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(packet)

    def _poll_frame_queue(self) -> None:
        """Drain queued packets from the worker thread and refresh the GUI."""

        try:
            while True:
                packet = self.frame_queue.get_nowait()
                if packet["kind"] == "frame":
                    self.current_state = packet["state"]
                    self._display_frame(packet["frame"])
                    self._update_dashboard(self.current_state)
                elif packet["kind"] == "event":
                    self.status_var.set(packet["message"])
                    if self.is_running:
                        self.stop_monitoring()
        except queue.Empty:
            pass
        finally:
            self.root.after(int(CONFIG["queue_poll_ms"]), self._poll_frame_queue)

    def _display_frame(self, frame) -> None:
        """Render an annotated OpenCV frame onto the Tkinter canvas."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame).resize(
            (int(CONFIG["video_width"]), int(CONFIG["video_height"])),
            Image.Resampling.LANCZOS,
        )
        self.current_photo = ImageTk.PhotoImage(image=image)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, image=self.current_photo, anchor=tk.NW)

    def _update_dashboard(self, state: ExerciseState) -> None:
        """Update live dashboard cards, feedback, and rep history."""

        self.stat_vars["REPS"].set(str(state.rep_count))
        self.stat_vars["HIP ANGLE"].set(f"{state.current_hip_angle:.1f}°")
        self.stat_vars["STATE"].set(state.current_state)
        self.stat_vars["FORM SCORE"].set(f"{state.form_score:.1f}%")
        self.stat_vars["PACE"].set(f"{state.pace_rpm:.1f} rpm")
        minutes = int(state.time_elapsed) // 60
        seconds = int(state.time_elapsed) % 60
        self.stat_vars["TIME"].set(f"{minutes:02d}:{seconds:02d}")
        self.feedback_var.set(state.form_feedback)
        self._set_feedback_colors(state.form_feedback)
        self._draw_rep_history(state.rep_history[-10:])

    def _set_feedback_colors(self, feedback: str) -> None:
        """Set the feedback bar color based on form quality."""

        if feedback == "Great Form!":
            bg_color, fg_color = "#D1FAE5", "#065F46"
        elif feedback in {"Keep Knees Bent", "Full Range Needed"}:
            bg_color, fg_color = "#FEF3C7", "#92400E"
        else:
            bg_color, fg_color = "#FEE2E2", "#991B1B"
        self.feedback_label.config(bg=bg_color, fg=fg_color)

    def _draw_rep_history(self, records: list[RepRecord]) -> None:
        """Render the last ten reps as a simple duration bar chart."""

        self.history_canvas.delete("all")
        width = int(self.history_canvas.cget("width"))
        height = int(self.history_canvas.cget("height"))
        if not records:
            self.history_canvas.create_text(
                width // 2,
                height // 2,
                text="No reps yet",
                fill="#6B7280",
                font=("Segoe UI", 10, "bold"),
            )
            return

        max_duration = max(record.duration_seconds for record in records) or 1.0
        bar_width = max(width // max(len(records), 1) - 8, 10)
        for index, record in enumerate(records):
            normalized = record.duration_seconds / max_duration
            bar_height = max(int(normalized * (height - 30)), 8)
            x1 = 12 + index * (bar_width + 6)
            y1 = height - bar_height - 18
            x2 = x1 + bar_width
            y2 = height - 18
            color = "#10B981" if record.is_good_form else "#F97316"
            self.history_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            self.history_canvas.create_text(
                (x1 + x2) // 2,
                height - 8,
                text=str(index + 1),
                fill="#6B7280",
                font=("Segoe UI", 8),
            )

    def _update_summary(self, state: ExerciseState) -> None:
        """Refresh the session summary section after a stop or reset."""

        self.summary_vars["reps"].set(str(state.rep_count) if state.rep_count else "--")
        self.summary_vars["form"].set(f"{state.form_score:.1f}%" if state.rep_count else "--")
        self.summary_vars["pace"].set(f"{state.best_pace:.1f} rpm" if state.best_pace else "--")
        if state.time_elapsed > 0:
            minutes = int(state.time_elapsed) // 60
            seconds = int(state.time_elapsed) % 60
            self.summary_vars["time"].set(f"{minutes:02d}:{seconds:02d}")
        else:
            self.summary_vars["time"].set("--")

    def _set_running_controls(self, running: bool) -> None:
        """Enable or disable controls to match the session state."""

        self.control_buttons["▶ Start"].config(state=tk.DISABLED if running else tk.NORMAL)
        self.control_buttons["📹 Camera"].config(state=tk.DISABLED if running else tk.NORMAL)
        self.control_buttons["📁 Video"].config(state=tk.DISABLED if running else tk.NORMAL)
        self.control_buttons["⏸ Pause"].config(state=tk.NORMAL if running else tk.DISABLED)
        self.control_buttons["⏹ Stop"].config(state=tk.NORMAL if running else tk.DISABLED)

    def _open_capture(self, source: str) -> CaptureSession:
        """Open a camera or file source with validation for the GUI."""

        if source.isdigit():
            camera_index = int(source)
            capture = cv2.VideoCapture(camera_index)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(CONFIG["camera_width"]))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CONFIG["camera_height"]))
            capture.set(cv2.CAP_PROP_FPS, int(CONFIG["camera_fps"]))
            if not capture.isOpened():
                raise ValueError(f"Unable to open camera index {camera_index}.")
            return CaptureSession(capture=capture, source_label=f"Camera {camera_index}", is_camera=True)

        video_path = Path(source).expanduser().resolve()
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        return CaptureSession(capture=capture, source_label=str(video_path), is_camera=False)

    def on_close(self) -> None:
        """Handle app shutdown safely."""

        if self.is_running and not messagebox.askyesno("Quit", "A session is still running. Quit anyway?"):
            return
        self.stop_monitoring()
        self.monitor.close()
        self.root.destroy()


def configure_logging() -> None:
    """Configure GUI logging."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    """Start the Tkinter sit-up monitor."""

    configure_logging()
    root = tk.Tk()
    app = SitUpGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
