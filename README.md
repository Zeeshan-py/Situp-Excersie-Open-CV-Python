# Sit-Up Monitor

Production-ready sit-up counter and form monitor built with OpenCV, MediaPipe Pose, and Tkinter. The project now uses MediaPipe's 33 landmark model end-to-end and removes all background subtraction and contour-based body hacks.

## Highlights

- MediaPipe Pose with landmark smoothing and confidence filtering
- Robust sit-up state machine: `UNKNOWN -> LYING -> RISING -> SITTING -> LOWERING -> LYING`
- Rep counting only after a full lying-to-sitting-to-lying cycle
- Form-quality scoring with knee-angle, range-of-motion, smoothness, and ankle-stability checks
- Simple OpenCV app for quick sessions
- Threaded Tkinter GUI for live monitoring and session review

## Requirements

- Python 3.10+
- Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dependencies

```text
opencv-python>=4.8.0
mediapipe>=0.10.0
Pillow>=10.0.0
numpy>=1.24.0
```

## File Structure

```text
main.py
SitUpGUI.py
SitUpCounter_Simple.py
SitUpExercise.py
PoseModule.py
requirements.txt
README.md
```

## Running the Project

Launch the menu:

```bash
python main.py
```

Run the simple OpenCV app directly:

```bash
python SitUpCounter_Simple.py --source 0
python SitUpCounter_Simple.py --source path/to/video.mp4
```

Run the Tkinter dashboard directly:

```bash
python SitUpGUI.py
```

## Controls

### Simple App

- `Q` or `Esc`: quit
- `R`: reset session counters
- `S`: save a timestamped screenshot

### GUI App

- `📹 Camera`: choose webcam input
- `📁 Video`: choose a video file
- `▶ Start`: begin processing
- `⏸ Pause`: pause or resume
- `⏹ Stop`: end the session
- `🔄 Reset`: clear metrics and history

## Camera Setup Recommendations

Best accuracy comes from:

- Side view with the camera at roughly 90 degrees to body direction
- Full body visible in frame
- Good lighting without strong backlight
- Uncluttered background
- Camera placed near hip height

## Detection Reliability Rules

- Frames are skipped when MediaPipe detects no pose
- Frames are skipped when active-side landmarks fall below visibility threshold
- Hip angle is smoothed with an exponential moving average
- State changes require three consecutive confirming frames
- Hysteresis prevents bounce counts near threshold boundaries

## Metrics

- `REPS`: completed repetitions
- `ANGLE`: current smoothed hip angle
- `STATE`: current state-machine stage
- `FORM`: live form feedback
- `PACE`: reps per minute over the last 60 seconds
- `SCORE`: good reps divided by total reps

## Troubleshooting

### Camera will not open

- Make sure another application is not using the webcam
- Try a different camera index if your default camera is not index `0`
- Reconnect the camera if the feed stops mid-session

### Tracking looks unreliable

- Reposition the camera to a true side view
- Keep ankles, knees, hips, and shoulders visible at all times
- Reduce motion blur with better lighting

### GUI feels slow

- Close other apps using the webcam
- Use the simple app on lower-spec machines
- Lower the input resolution if needed

## Architecture Notes

- `PoseModule.py` wraps MediaPipe Pose and exposes safe landmark and angle helpers
- `SitUpExercise.py` contains the shared state machine, form analysis, and HUD renderer
- `SitUpCounter_Simple.py` runs the OpenCV-only experience
- `SitUpGUI.py` runs threaded capture and updates the Tkinter dashboard through a queue
