"""
Professional Sit-Up Counter - Standalone Version
Human Activity Monitoring System

This is a standalone implementation using OpenCV-based body detection
for sit-up counting with hip angle analysis. Supports both webcam and video file input.

Features:
- Real-time sit-up counting
- Hip angle measurement
- Video file upload support
- Background subtraction for body detection
- Exponential smoothing for stability

Author: PBL-CP-II Project
Date: December 2025
"""

import cv2
import numpy as np
import math
from collections import deque
import time


class SitUpCounter:
    """Professional sit-up counter using hip angle analysis"""
    
    def __init__(self):
        # Sit-up counting logic
        self.count = 0
        self.state = "UNKNOWN"  # States: DOWN, UP, UNKNOWN
        
        # Thresholds for state detection (adjusted for better counting)
        self.DOWN_THRESHOLD = 130  # Lying down if angle > 130°
        self.UP_THRESHOLD = 100     # Sitting up if angle < 100°
        self.HYSTERESIS = 5        # Prevent rapid state changes
        
        # Smoothing for angle stability (less aggressive for responsiveness)
        self.smoothing_alpha = 0.5
        self.smoothed_angle = None
        self.angle_history = deque(maxlen=50)
        
        # Performance tracking
        self.start_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Background subtractor for person detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Body reference points (will be updated each frame)
        self.shoulder_pos = None
        self.hip_pos = None
        self.knee_pos = None
        
        # Debug mode
        self.debug = True
        self.last_state = None
        
    def detect_body_points(self, frame):
        """
        Detect body keypoints using computer vision
        
        Returns:
            (shoulder, hip, knee) positions or (None, None, None) if not found
        """
        height, width = frame.shape[:2]
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        # Find largest contour (person)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Minimum area threshold
        if area < 5000:
            return None, None, None
        
        # Get bounding box and moments
        x, y, w, h = cv2.boundingRect(largest_contour)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None, None, None
        
        # Calculate centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # More precise body point estimation for side view
        # Use more vertical spacing for better angle variation
        
        # Shoulder: upper region
        shoulder = (x + int(w * 0.3), y + int(h * 0.2))
        
        # Hip: center point (this is the vertex for angle calculation)
        hip = (cx, cy)
        
        # Knee: lower region with more separation from hip
        knee = (x + int(w * 0.3), y + int(h * 0.8))
        
        return shoulder, hip, knee
    
    def calculate_angle_vector_method(self, p1, p2, p3):
        """
        Calculate angle at p2 (vertex) using vector dot product
        
        Mathematical approach:
            cos(θ) = (v1·v2) / (|v1||v2|)
        """
        if None in (p1, p2, p3):
            return None
        
        # Create vectors from vertex (p2)
        vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
        vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
        
        # Calculate magnitudes
        mag1 = np.linalg.norm(vector1)
        mag2 = np.linalg.norm(vector2)
        
        if mag1 == 0 or mag2 == 0:
            return None
        
        # Calculate dot product
        dot_product = np.dot(vector1, vector2)
        
        # Calculate angle
        cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def smooth_angle(self, raw_angle):
        """Apply exponential smoothing to reduce jitter"""
        if raw_angle is None:
            return self.smoothed_angle
        
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            self.smoothed_angle = (self.smoothing_alpha * raw_angle + 
                                 (1 - self.smoothing_alpha) * self.smoothed_angle)
        
        return self.smoothed_angle
    
    def update_state(self, angle):
        """
        Update state machine based on hip angle
        
        Complete rep: DOWN → UP → DOWN (count increment)
        Uses hysteresis to prevent false counts
        """
        if angle is None:
            return
        
        if self.state == "UNKNOWN":
            # Initialize state based on first valid angle
            if angle > self.DOWN_THRESHOLD + self.HYSTERESIS:
                self.state = "DOWN"
            elif angle < self.UP_THRESHOLD - self.HYSTERESIS:
                self.state = "UP"
        
        elif self.state == "DOWN":
            # Transition to UP when angle drops significantly
            if angle < self.UP_THRESHOLD:
                self.last_state = "DOWN"
                self.state = "UP"
                if self.debug:
                    print(f"✓ State: DOWN -> UP at angle {angle:.1f}°")
        
        elif self.state == "UP":
            # Transition to DOWN and count when angle rises significantly
            if angle > self.DOWN_THRESHOLD:
                self.last_state = "UP"
                self.state = "DOWN"
                self.count += 1
                if self.debug:
                    print(f"✓✓✓ COUNT +1 | UP -> DOWN at angle {angle:.1f}° | Total: {self.count} ✓✓✓")
    
    def draw_visualization(self, frame, angle):
        """Draw clean UI with landmarks"""
        h, w, _ = frame.shape
        
        # Draw precise skeleton if points are detected
        if all([self.shoulder_pos, self.hip_pos, self.knee_pos]):
            # Draw skeleton lines with better visibility
            cv2.line(frame, self.shoulder_pos, self.hip_pos, (0, 255, 0), 4)
            cv2.line(frame, self.hip_pos, self.knee_pos, (0, 255, 0), 4)
            
            # Draw joints with precise circles
            joint_color = (0, 255, 255) if self.state == "UP" else (255, 0, 255)
            cv2.circle(frame, self.shoulder_pos, 10, joint_color, -1)
            cv2.circle(frame, self.shoulder_pos, 12, (255, 255, 255), 2)
            
            cv2.circle(frame, self.hip_pos, 14, (0, 0, 255), -1)
            cv2.circle(frame, self.hip_pos, 16, (255, 255, 255), 2)
            
            cv2.circle(frame, self.knee_pos, 10, joint_color, -1)
            cv2.circle(frame, self.knee_pos, 12, (255, 255, 255), 2)
        
        # Compact info panel (semi-transparent background)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Count (large and prominent)
        cv2.putText(frame, f"Count: {self.count}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Time elapsed
        elapsed_time = int(time.time() - self.start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Hip Angle with threshold indicators
        if angle is not None:
            # Color based on zone
            if angle < self.UP_THRESHOLD:
                angle_color = (0, 255, 0)  # Green = UP zone
                zone = "UP"
            elif angle > self.DOWN_THRESHOLD:
                angle_color = (255, 0, 255)  # Magenta = DOWN zone
                zone = "DOWN"
            else:
                angle_color = (0, 255, 255)  # Cyan = TRANSITION zone
                zone = "MID"
            
            cv2.putText(frame, f"Angle: {angle:.1f}° [{zone}]", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_color, 2)
        else:
            cv2.putText(frame, "Angle: -- [NO DETECT]", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # State indicator (shows UP/DOWN for debugging)
        state_color = (0, 255, 255) if self.state == "UP" else (255, 0, 255)
        if self.state == "UNKNOWN":
            state_color = (128, 128, 128)
        cv2.putText(frame, f"State: {self.state}", (20, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        return frame
    

    
    def process_frame(self, frame):
        """Main processing pipeline"""
        # Detect body points
        shoulder, hip, knee = self.detect_body_points(frame)
        
        # Store for visualization
        self.shoulder_pos = shoulder
        self.hip_pos = hip
        self.knee_pos = knee
        
        # Calculate hip angle
        raw_angle = self.calculate_angle_vector_method(shoulder, hip, knee)
        
        # Smooth angle
        angle = self.smooth_angle(raw_angle)
        
        # Debug: Print angle every 30 frames (once per second at 30fps)
        if self.debug and self.frame_count % 30 == 0 and angle is not None:
            print(f"Angle: {angle:.1f}° | State: {self.state} | Thresholds: UP<{self.UP_THRESHOLD}°, DOWN>{self.DOWN_THRESHOLD}°")
        
        # Update history
        self.angle_history.append(angle)
        
        # Update state machine
        self.update_state(angle)
        
        # Draw visualization
        frame = self.draw_visualization(frame, angle)
        
        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        return frame
    
    def reset(self):
        """Reset counter"""
        self.count = 0
        self.state = "UNKNOWN"
        self.smoothed_angle = None
        self.angle_history.clear()
        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        print("Counter reset! Please allow 2-3 seconds for recalibration.")


def main():
    """Main application"""
    print("=" * 60)
    print("PROFESSIONAL SIT-UP COUNTER")
    print("=" * 60)
    
    # Ask user for input source
    print("\nSelect input source:")
    print("  1. Webcam (Live)")
    print("  2. Video File")
    print()
    choice = input("Enter choice (1 or 2): ").strip()
    
    video_path = None
    if choice == "2":
        print("\nEnter video file path (or drag and drop file):")
        video_path = input("Path: ").strip().strip('"').strip("'")
        if not video_path:
            print("✗ No file path provided")
            return
    
    print("\nInitializing...")
    counter = SitUpCounter()
    print("✓ Counter initialized")
    
    # Open video source
    if video_path:
        print(f"\nOpening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        is_webcam = False
    else:
        print("\nOpening webcam...")
        cap = cv2.VideoCapture(0)
        is_webcam = True
    
    if not cap.isOpened():
        print("✗ Could not open video source")
        return
    
    # Set camera properties (only for webcam)
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_path:
        print(f"✓ Video opened | FPS: {fps:.1f} | Frames: {total_frames}")
    else:
        print("✓ Webcam opened")
    
    print("\n" + "=" * 60)
    print("READY TO COUNT SIT-UPS!")
    print("=" * 60)
    print("\nIMPORTANT:")
    print("  1. Lie on your RIGHT side facing the camera")
    print("  2. Keep your FULL BODY in frame at all times")
    if is_webcam:
        print("  3. Wait 2-3 seconds for background calibration")
        print("  4. Perform sit-ups at a CONTROLLED pace")
    print("\nControls:")
    print("  'q' = Quit")
    print("  'r' = Reset counter")
    if video_path:
        print("  'p' = Pause/Resume")
        print("  SPACE = Next frame (when paused)")
    print("\n" + "=" * 60)
    
    # Allow time for background calibration (webcam only)
    if is_webcam:
        print("\nCalibrating background (stay still)...")
        for i in range(60):  # ~2 seconds at 30fps
            ret, frame = cap.read()
            if ret:
                counter.bg_subtractor.apply(frame)
        print("✓ Calibration complete!\n")
    else:
        print("\nProcessing video...\n")
    
    paused = False
    frame_delay = int(1000 / fps) if fps > 0 else 33
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    print("\n✓ Video processing complete!")
                else:
                    print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame = counter.process_frame(frame)
            
            # Add video progress bar for video files
            if video_path and total_frames > 0:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                progress = current_frame / total_frames
                bar_width = processed_frame.shape[1] - 40
                bar_height = 20
                bar_y = processed_frame.shape[0] - 40
                
                # Background
                cv2.rectangle(processed_frame, (20, bar_y), 
                            (20 + bar_width, bar_y + bar_height),
                            (50, 50, 50), -1)
                # Progress
                cv2.rectangle(processed_frame, (20, bar_y), 
                            (20 + int(bar_width * progress), bar_y + bar_height),
                            (0, 255, 0), -1)
                # Border
                cv2.rectangle(processed_frame, (20, bar_y), 
                            (20 + bar_width, bar_y + bar_height),
                            (255, 255, 255), 2)
                # Text
                progress_text = f"{int(progress * 100)}% ({current_frame}/{total_frames})"
                cv2.putText(processed_frame, progress_text, 
                           (30, bar_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Professional Sit-Up Counter', processed_frame)
        
        # Handle keyboard
        key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            counter.reset()
        elif key == ord('p') and video_path:
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == 32 and paused and video_path:  # SPACE key
            ret, frame = cap.read()
            if ret:
                processed_frame = counter.process_frame(frame)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"SESSION COMPLETE")
    print(f"Total sit-ups: {counter.count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
