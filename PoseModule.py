"""
Pose Module - Using Background Subtraction and Contour Analysis
Improved pose detection for sit-up tracking
"""
import cv2
import numpy as np
import math


class PoseDetector:
    """
    Pose detection using advanced computer vision techniques
    """

    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.lmList = []
        self.bboxInfo = {}
        self.frame_count = 0
        self.prev_angle = 90
        self.movement_detected = False
        
        # Background subtractor for better person detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.use_bg_subtraction = False
        self.background_frames = 0
        
    def findPose(self, img, draw=True):
        """
        Find pose using multiple detection methods
        """
        self.frame_count += 1
        h, w = img.shape[:2]
        
        # Method 1: Use background subtraction if available
        if self.frame_count > 30:  # Build background model first
            self.use_bg_subtraction = True
        
        if self.use_bg_subtraction:
            fg_mask = self.bg_subtractor.apply(img, learningRate=0.001)
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in foreground mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Method 2: Use edge detection and contour finding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate to connect edges
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assumed to be the person)
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 5000:  # Minimum area threshold
                # Get bounding box and center
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                
                # Calculate moments for better center detection
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    body_center_x = int(M["m10"] / M["m00"])
                    body_center_y = int(M["m01"] / M["m00"])
                else:
                    body_center_x = x + bw // 2
                    body_center_y = y + bh // 2
                
                # Find the topmost and bottommost points
                topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
                rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
                
                # Estimate body proportions
                body_height = bottommost[1] - topmost[1]
                body_width = rightmost[0] - leftmost[0]
                
                # Head (top 10% of body)
                head_y = topmost[1] + int(body_height * 0.05)
                
                # Shoulders (approximately 18% from top)
                shoulder_y = topmost[1] + int(body_height * 0.18)
                left_shoulder_x = body_center_x - int(body_width * 0.20)
                right_shoulder_x = body_center_x + int(body_width * 0.20)
                
                # Hips (approximately 45-50% from top)
                hip_y = topmost[1] + int(body_height * 0.48)
                left_hip_x = body_center_x - int(body_width * 0.15)
                right_hip_x = body_center_x + int(body_width * 0.15)
                
                # Knees (approximately 70% from top)
                knee_y = topmost[1] + int(body_height * 0.70)
                left_knee_x = body_center_x - int(body_width * 0.12)
                right_knee_x = body_center_x + int(body_width * 0.12)
                
                # Ankles (approximately 90% from top)
                ankle_y = topmost[1] + int(body_height * 0.90)
                
                # Elbows (approximately 35% from top)
                elbow_y = topmost[1] + int(body_height * 0.35)
                left_elbow_x = body_center_x - int(body_width * 0.28)
                right_elbow_x = body_center_x + int(body_width * 0.28)
                
                # Wrists (approximately 50% from top, wider)
                wrist_y = topmost[1] + int(body_height * 0.50)
                left_wrist_x = body_center_x - int(body_width * 0.35)
                right_wrist_x = body_center_x + int(body_width * 0.35)
                
                # Create landmark list with improved positions
                self.lmList = [
                    [0, body_center_x, head_y, 0],  # 0: nose/head
                    [1, body_center_x, head_y + 5, 0],  # 1: neck top
                    [2, body_center_x - 15, head_y, 0],  # 2-10: face features
                    [3, body_center_x - 20, head_y, 0],
                    [4, body_center_x + 15, head_y, 0],
                    [5, body_center_x + 20, head_y, 0],
                    [6, body_center_x + 20, head_y, 0],
                    [7, body_center_x - 30, head_y + 5, 0],
                    [8, body_center_x + 30, head_y + 5, 0],
                    [9, body_center_x - 8, head_y + 15, 0],
                    [10, body_center_x + 8, head_y + 15, 0],
                    [11, left_shoulder_x, shoulder_y, 0],  # 11: LEFT SHOULDER ***
                    [12, right_shoulder_x, shoulder_y, 0],  # 12: RIGHT SHOULDER ***
                    [13, left_elbow_x, elbow_y, 0],  # 13: left elbow
                    [14, right_elbow_x, elbow_y, 0],  # 14: right elbow
                    [15, left_wrist_x, wrist_y, 0],  # 15: left wrist
                    [16, right_wrist_x, wrist_y, 0],  # 16: right wrist
                    [17, left_wrist_x - 5, wrist_y + 10, 0],  # 17-22: hands
                    [18, left_wrist_x, wrist_y + 10, 0],
                    [19, left_wrist_x + 5, wrist_y + 10, 0],
                    [20, right_wrist_x - 5, wrist_y + 10, 0],
                    [21, right_wrist_x, wrist_y + 10, 0],
                    [22, right_wrist_x + 5, wrist_y + 10, 0],
                    [23, left_hip_x, hip_y, 0],  # 23: LEFT HIP ***
                    [24, right_hip_x, hip_y, 0],  # 24: RIGHT HIP ***
                    [25, left_knee_x, knee_y, 0],  # 25: LEFT KNEE ***
                    [26, right_knee_x, knee_y, 0],  # 26: RIGHT KNEE ***
                    [27, left_knee_x, ankle_y, 0],  # 27: left ankle
                    [28, right_knee_x, ankle_y, 0],  # 28: right ankle
                    [29, left_knee_x, bottommost[1], 0],  # 29: left heel
                    [30, right_knee_x, bottommost[1], 0],  # 30: right heel
                    [31, left_knee_x + 10, bottommost[1], 0],  # 31: left foot
                ]
                
                self.movement_detected = True
                
                if draw:
                    # Draw skeleton connections
                    connections = [
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                        (11, 23), (12, 24), (23, 24),  # Torso
                        (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                    ]
                    
                    for connection in connections:
                        if connection[0] < len(self.lmList) and connection[1] < len(self.lmList):
                            pt1 = (self.lmList[connection[0]][1], self.lmList[connection[0]][2])
                            pt2 = (self.lmList[connection[1]][1], self.lmList[connection[1]][2])
                            cv2.line(img, pt1, pt2, (0, 255, 0), 3)
                    
                    # Draw key landmarks with labels
                    key_points = {
                        11: 'L.Shoulder', 12: 'R.Shoulder',
                        23: 'L.Hip', 24: 'R.Hip',
                        25: 'L.Knee', 26: 'R.Knee'
                    }
                    
                    for i, lm in enumerate(self.lmList):
                        if i in key_points:
                            # Draw larger circles for key points
                            cv2.circle(img, (lm[1], lm[2]), 8, (0, 0, 255), cv2.FILLED)
                            cv2.circle(img, (lm[1], lm[2]), 10, (255, 255, 255), 2)
                            # Add labels
                            cv2.putText(img, key_points[i], (lm[1] + 12, lm[2] - 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        else:
                            # Regular landmarks
                            cv2.circle(img, (lm[1], lm[2]), 4, (255, 255, 0), cv2.FILLED)
                    
                    # Draw contour outline
                    cv2.drawContours(img, [largest_contour], -1, (0, 255, 255), 2)
                    
                    # Draw extreme points for reference
                    cv2.circle(img, topmost, 8, (255, 0, 0), -1)
                    cv2.circle(img, bottommost, 8, (255, 0, 0), -1)
        
        return img

    def findPosition(self, img, draw=True, bboxWithHands=False):
        """
        Get landmark positions
        """
        h, w, c = img.shape
        
        if len(self.lmList) == 0:
            self.findPose(img, draw=False)
        
        # Calculate bounding box
        if len(self.lmList) >= 32:
            ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][1] - ad
                x2 = self.lmList[15][1] + ad
            else:
                x1 = self.lmList[12][1] - ad
                x2 = self.lmList[11][1] + ad

            y2 = self.lmList[29][2] + ad
            y1 = self.lmList[1][2] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculate angle between three points
        """
        if len(self.lmList) == 0:
            return 0

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return angle
