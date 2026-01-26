"""
Sit-Up Exercise Monitor
Tracks and counts sit-up exercises with quality assessment
"""
import cv2
import time
from PoseModule import PoseDetector


class SitUpMonitor:
    """
    Monitor sit-up exercise performance using pose detection
    """

    def __init__(self):
        """
        Initialize the Sit-Up Monitor
        """
        self.detector = PoseDetector(detectionCon=0.7, trackCon=0.7)
        self.count = 0
        self.direction = 0  # 0: going down, 1: going up
        self.form_feedback = "Starting..."
        self.angle = 0
        
        # Simplified angle thresholds for sit-up
        # We'll just track crossing thresholds
        self.sit_threshold = 120    # Below this = sitting
        self.lie_threshold = 140    # Above this = lying
        self.in_sit_zone = False
        self.in_lie_zone = False
        
        # Time tracking
        self.start_time = None
        self.elapsed_time = 0
        
        # Quality metrics
        self.good_form_count = 0
        self.poor_form_count = 0

    def reset(self):
        """
        Reset all counters and metrics
        """
        self.count = 0
        self.direction = 0
        self.form_feedback = "Starting..."
        self.angle = 0
        self.start_time = None
        self.elapsed_time = 0
        self.good_form_count = 0
        self.poor_form_count = 0
        self.in_sit_zone = False
        self.in_lie_zone = False

    def check_form(self, angle):
        """
        Check if the sit-up form is correct
        :param angle: Hip angle during sit-up
        :return: Form quality message
        """
        if self.direction == 1:  # Going up (sitting)
            if 70 <= angle <= 110:
                return "Good Form"
            elif angle < 70:
                return "Too Far Up"
            else:
                return "Not Sitting Enough"
        else:  # Going down (lying)
            if 150 <= angle <= 180:
                return "Good Form"
            elif angle < 150:
                return "Lie Flatter"
            else:
                return "Form OK"

    def process_frame(self, img):
        """
        Process a single frame to detect sit-up
        :param img: Input image frame
        :return: Processed image with overlays
        """
        img = self.detector.findPose(img, draw=True)
        lmList, bboxInfo = self.detector.findPosition(img, draw=False)

        # Show detection status
        cv2.putText(img, f'Landmarks: {len(lmList)}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if len(lmList) >= 27:  # Need at least 27 landmarks for knee detection
            # Calculate hip angle (Shoulder-Hip-Knee)
            # Using right side: 12-24-26
            self.angle = self.detector.findAngle(img, 12, 24, 26, draw=True)
            
            h, w = img.shape[:2]
            
            # Draw angle scale on right side
            bar_x = w - 150
            bar_y = 100
            bar_height = 400
            
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + 40, bar_y + bar_height), (100, 100, 100), 2)
            
            # Mark thresholds
            sit_pos = bar_y + int((180 - self.sit_threshold) / 180 * bar_height)
            lie_pos = bar_y + int((180 - self.lie_threshold) / 180 * bar_height)
            
            cv2.line(img, (bar_x - 10, sit_pos), (bar_x + 50, sit_pos), (0, 255, 0), 2)
            cv2.putText(img, f'SIT<{self.sit_threshold}', (bar_x - 140, sit_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.line(img, (bar_x - 10, lie_pos), (bar_x + 50, lie_pos), (255, 0, 0), 2)
            cv2.putText(img, f'LIE>{self.lie_threshold}', (bar_x - 140, lie_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw current angle position
            angle_pos = bar_y + int((180 - min(180, max(0, self.angle))) / 180 * bar_height)
            cv2.circle(img, (bar_x + 20, angle_pos), 15, (0, 255, 255), cv2.FILLED)
            cv2.putText(img, str(int(self.angle)), (bar_x + 50, angle_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show angle prominently
            cv2.putText(img, f'ANGLE: {int(self.angle)}', (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            
            # Simplified counting logic - just track zone crossings
            current_status = ""
            
            if self.angle < self.sit_threshold:
                current_status = "IN SIT ZONE"
                if not self.in_sit_zone:
                    self.in_sit_zone = True
                    if self.in_lie_zone:  # Coming from lie zone
                        self.count += 1
                        cv2.putText(img, 'COUNT +1!', (w//2 - 100, h//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                        self.in_lie_zone = False
                        
            elif self.angle > self.lie_threshold:
                current_status = "IN LIE ZONE"
                if not self.in_lie_zone:
                    self.in_lie_zone = True
                    self.in_sit_zone = False
                    
            else:
                current_status = "IN TRANSITION"
            
            # Display status
            cv2.putText(img, f'STATUS: {current_status}', (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.putText(img, f'COUNT: {int(self.count)}', (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Show zones status
            sit_color = (0, 255, 0) if self.in_sit_zone else (100, 100, 100)
            lie_color = (255, 0, 0) if self.in_lie_zone else (100, 100, 100)
            
            cv2.putText(img, f'Sit Zone: {"YES" if self.in_sit_zone else "NO"}', (50, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, sit_color, 2)
            cv2.putText(img, f'Lie Zone: {"YES" if self.in_lie_zone else "NO"}', (50, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, lie_color, 2)
            
            # Form feedback
            if self.angle < self.sit_threshold:
                self.form_feedback = "Good Sit Position"
            elif self.angle > self.lie_threshold:
                self.form_feedback = "Good Lie Position"
            else:
                self.form_feedback = "Keep Moving"
        else:
            # No detection
            cv2.putText(img, 'NO BODY DETECTED', (img.shape[1]//2 - 200, img.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(img, 'Position yourself in frame', (img.shape[1]//2 - 250, img.shape[0]//2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

    def get_statistics(self):
        """
        Get current exercise statistics
        :return: Dictionary with statistics
        """
        total = int(self.count)
        quality_percentage = (self.good_form_count / total * 100) if total > 0 else 0
        
        return {
            'count': total,
            'time': self.elapsed_time,
            'good_form': self.good_form_count,
            'poor_form': self.poor_form_count,
            'quality': quality_percentage,
            'angle': int(self.angle)
        }


def main():
    """
    Test function for sit-up monitor without GUI
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    monitor = SitUpMonitor()
    start_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = monitor.process_frame(img)

        # Update time
        monitor.elapsed_time = int(time.time() - start_time)

        # Display statistics
        stats = monitor.get_statistics()
        cv2.putText(img, f"Count: {stats['count']}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(img, f"Time: {stats['time']}s", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Quality: {stats['quality']:.1f}%", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sit-Up Monitor", img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            monitor.reset()
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
