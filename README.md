# Sit-Up Exercise Monitor

A real-time sit-up counter and exercise monitoring system using computer vision and pose detection. This project provides automated sit-up counting with form analysis and quality assessment.

## 🎯 Features

- **Real-time Sit-Up Counting**: Automatically counts sit-ups using hip angle analysis
- **Form Quality Assessment**: Provides feedback on exercise form (Good Form, Poor Form)
- **Multiple Input Sources**: Supports both webcam and video file input
- **Dual Interface Options**:
  - **GUI Version**: Full-featured interface with video controls, statistics dashboard, and file management
  - **Simple Version**: Lightweight standalone application for quick sessions
- **Advanced Pose Detection**: Uses background subtraction and contour analysis for accurate body tracking
- **Performance Metrics**: Tracks rep count, time elapsed, and form quality statistics
- **Visual Feedback**: Real-time angle visualization and body landmark detection

## 📋 Requirements

- Python 3.13 or higher
- OpenCV >= 4.5.0
- MediaPipe >= 0.8.9
- Pillow >= 9.0.0
- NumPy >= 1.21.0

## 🚀 Installation

1. Clone or download this repository:
```bash
cd burhan
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start - Launcher Menu

Run the main launcher to choose your preferred version:

```bash
python main.py
```

This will display a menu with three options:
1. **GUI Version** - Full-featured interface (Recommended)
2. **Simple Version** - Lightweight standalone application
3. **Exit**

### Option 1: GUI Version (Recommended)

Launch the full GUI application:

```bash
python SitUpGUI.py
```

**Features:**
- Video file upload and playback
- Start/Pause/Stop controls
- Real-time statistics dashboard
- Exercise history tracking
- Reset functionality

**GUI Controls:**
- 📹 **Open Camera**: Start webcam feed
- 📁 **Open Video File**: Load a video for analysis
- ▶ **Start**: Begin monitoring
- ⏸ **Pause**: Pause the session
- ⏹ **Stop**: End the session
- 🔄 **Reset**: Clear all statistics

### Option 2: Simple Version

Launch the lightweight version:

```bash
python SitUpCounter_Simple.py
```

**Features:**
- Webcam or video file input
- Real-time counting display
- Minimal resource usage
- Keyboard controls:
  - `ESC` or `Q`: Exit
  - `R`: Reset counter

## 🏗️ Project Structure

```

├── main.py                    # Application launcher with menu
├── SitUpGUI.py                # Full GUI application
├── SitUpCounter_Simple.py     # Lightweight standalone version
├── SitUpExercise.py           # Sit-up monitoring logic
├── PoseModule.py              # Pose detection module
├── requirements.txt           # Python dependencies
└── __pycache__/              # Python cache files
```

## 🔧 How It Works

1. **Pose Detection**: Uses computer vision techniques including:
   - Background subtraction (MOG2 algorithm)
   - Contour analysis for body detection
   - Landmark estimation for body keypoints

2. **Angle Calculation**: Measures hip angle using shoulder-hip-knee alignment

3. **Sit-Up Detection**: 
   - **Lying Position**: Hip angle > 140°
   - **Sitting Position**: Hip angle < 120°
   - Counter increments when transitioning from lying to sitting

4. **Form Assessment**: Analyzes movement quality and provides real-time feedback

## 📊 Key Metrics

- **Rep Count**: Number of completed sit-ups
- **Hip Angle**: Real-time angle measurement (shoulder-hip-knee)
- **Form Quality**: Good form vs. poor form tracking
- **Time Elapsed**: Session duration
- **Average Pace**: Reps per minute

## 🎨 GUI Interface Overview

The GUI application features:
- **Large video display area**: Shows real-time feed with overlay graphics
- **Control panel**: Easy-to-use buttons for all functions
- **Statistics dashboard**: 
  - Current rep count
  - Hip angle display
  - Time elapsed
  - Form feedback
  - Quality metrics
- **Status indicators**: Visual feedback on system state

## 🔬 Technical Details

### Pose Detection Algorithm

The `PoseModule.py` implements a custom pose detection system:
- Background subtraction for person segmentation
- Contour-based body region detection
- Proportional body landmark estimation
- Smoothing algorithms for stable tracking

### Angle Thresholds

- **Sit Threshold**: 120° (below = sitting position)
- **Lie Threshold**: 140° (above = lying position)
- **Hysteresis**: 5° (prevents rapid state changes)

## 🐛 Troubleshooting

**Camera not detected:**
- Ensure your webcam is properly connected
- Check if another application is using the camera
- Try running as administrator (Windows)

**Poor detection accuracy:**
- Ensure good lighting conditions
- Position yourself in the center of the frame
- Maintain clear visibility of shoulders, hips, and knees
- Avoid cluttered backgrounds

**Performance issues:**
- Close other applications to free up resources
- Use the Simple version for lower-spec systems
- Reduce video resolution if possible

## 📝 Notes

- Best results achieved with:
  - Good lighting conditions
  - Uncluttered background
  - Side-view camera angle
  - Full body visibility
  
- The system builds a background model during the first ~30 frames
- Hip angle is calculated using the right side of the body (landmarks 12-24-26)

## 🎓 Project Information

**Project**: Human Activity Monitoring System  
**Date**: December 2025  
**Focus**: Real-time exercise monitoring using computer vision

## 📄 License

This project is developed for educational purposes as part of the PBL-CP-II course.

## 🤝 Contributing

This is an academic project. For suggestions or improvements, feel free to fork and experiment!

---

**Made with 💪 for fitness tracking and computer vision learning**
