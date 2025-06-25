import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import math
import time


class BikeFitAnalyzer:
    def __init__(self):
        # ===== WEBCAM CONFIGURATION - PLEASE ADJUST =====
        # For MacBook: Usually index 0 or 1
        # For Windows: Usually index 0
        # For external webcams: Try index 1, 2, 3...
        # If problems occur, test different indices

        camera_index = 0  # <- ADJUST HERE IF NEEDED

        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_index)

        # Check if camera was successfully opened
        if not self.cap.isOpened():
            print(f"ERROR: Camera with index {camera_index} could not be opened!")
            print("Try a different index (0, 1, 2...)")
            exit(1)

        # Set resolution (can be adjusted depending on camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ===== END WEBCAM CONFIGURATION =====

        # Initialize Pose Detector
        self.detector = PoseDetector()

        # Variables for angle calculation
        self.max_angle = 0
        self.current_angle = 0

        # Optimal angle range (can be adjusted)
        self.optimal_angle_min = 140
        self.optimal_angle_max = 150

        # Colors for visualization (BGR format)
        self.color_optimal = (0, 255, 0)  # Green
        self.color_too_low = (0, 165, 255)  # Orange
        self.color_too_high = (0, 0, 255)  # Red
        self.color_neutral = (255, 255, 255)  # White

        # Timer and measurement state variables
        self.state = "idle"  # States: idle, countdown, measuring, finished
        self.countdown_start = None
        self.measurement_start = None
        self.countdown_duration = 5  # seconds
        self.measurement_duration = 5  # seconds

    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        p1: Hip (x, y)
        p2: Knee (x, y) - vertex
        p3: Ankle (x, y)
        """
        # Calculate vectors
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clipping to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Convert to degrees
        return np.degrees(angle)

    def get_feedback(self, angle):
        """
        Provide feedback based on measured angle
        """
        if angle < self.optimal_angle_min:
            return "Saddle too low - please raise", self.color_too_low
        elif angle > self.optimal_angle_max:
            return "Saddle too high - please lower", self.color_too_high
        else:
            return "Saddle height optimal", self.color_optimal

    def draw_angle_visualization(self, img, p1, p2, p3, angle, color):
        """
        Draw angle visualization on the image
        """
        # Draw lines between points
        cv2.line(img, p1, p2, color, 3)
        cv2.line(img, p2, p3, color, 3)

        # Draw joint points as circles
        cv2.circle(img, p1, 8, color, -1)  # Hip
        cv2.circle(img, p2, 10, color, -1)  # Knee (larger)
        cv2.circle(img, p3, 8, color, -1)  # Ankle

        # Display angle at knee (using 'deg' instead of degree symbol)
        cv2.putText(img, f"{int(angle)} deg",
                    (p2[0] - 40, p2[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_timer_overlay(self, img, remaining_time, message):
        """
        Draw countdown timer overlay
        """
        # Semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # Large countdown number
        cv2.putText(img, str(int(remaining_time)),
                    (img.shape[1] // 2 - 50, img.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, self.color_neutral, 8)

        # Message
        cv2.putText(img, message,
                    (img.shape[1] // 2 - len(message) * 12, img.shape[0] // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.color_neutral, 3)

    def run(self):
        """
        Main program loop
        """
        print("Bike Fit Analyzer started!")
        print("Instructions:")
        print("  1. Position your computer so you can see yourself sideways")
        print("  2. Press SPACEBAR to start measurement")
        print("  3. You have 5 seconds to get on your bike")
        print("  4. Pedal for 5 seconds during measurement")
        print("  'q' - Quit program")
        print("-" * 60)

        while True:
            success, img = self.cap.read()
            if not success:
                print("Error reading camera")
                break

            # Mirror image for more natural display
            img = cv2.flip(img, 1)

            # State machine for measurement process
            current_time = time.time()

            if self.state == "countdown":
                remaining = self.countdown_duration - (current_time - self.countdown_start)
                if remaining > 0:
                    self.draw_timer_overlay(img, remaining, "Get on your bike!")
                else:
                    # Start measurement
                    self.state = "measuring"
                    self.measurement_start = current_time
                    self.max_angle = 0  # Reset max angle
                    print("Measurement started - pedal now!")

            elif self.state == "measuring":
                remaining = self.measurement_duration - (current_time - self.measurement_start)
                if remaining > 0:
                    # During measurement - update max angle
                    # Detect pose
                    img = self.detector.findPose(img, draw=False)
                    lmList, bboxInfo = self.detector.findPosition(img, draw=False)

                    if lmList:
                        try:
                            # Right side analysis
                            hip = (lmList[23][0], lmList[23][1])
                            knee = (lmList[25][0], lmList[25][1])
                            ankle = (lmList[27][0], lmList[27][1])

                            # Calculate angle
                            self.current_angle = self.calculate_angle(hip, knee, ankle)

                            # Update maximum angle during measurement
                            if self.current_angle > self.max_angle:
                                self.max_angle = self.current_angle

                            # Visualization during measurement
                            self.draw_angle_visualization(img, hip, knee, ankle,
                                                          self.current_angle, self.color_neutral)
                        except IndexError:
                            pass

                    # Show measurement progress
                    cv2.putText(img, f"MEASURING... {int(remaining)}s",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 255), 3)
                else:
                    # Measurement finished
                    self.state = "finished"
                    print(f"Measurement complete! Maximum angle: {int(self.max_angle)} degrees")
                    feedback_text, _ = self.get_feedback(self.max_angle)
                    print(f"Result: {feedback_text}")

            else:  # idle or finished states
                # Normal pose detection (no angle updates in finished state)
                img = self.detector.findPose(img, draw=False)
                lmList, bboxInfo = self.detector.findPosition(img, draw=False)

                if lmList:
                    try:
                        # Right side analysis
                        hip = (lmList[23][0], lmList[23][1])
                        knee = (lmList[25][0], lmList[25][1])
                        ankle = (lmList[27][0], lmList[27][1])

                        # Calculate current angle (display only)
                        self.current_angle = self.calculate_angle(hip, knee, ankle)

                        # Get color based on max angle (if we have one)
                        if self.state == "finished" and self.max_angle > 0:
                            _, color = self.get_feedback(self.max_angle)
                        else:
                            color = self.color_neutral

                        # Visualization
                        self.draw_angle_visualization(img, hip, knee, ankle,
                                                      self.current_angle, color)

                    except IndexError:
                        # If joints are not detected
                        if self.state == "idle":
                            cv2.putText(img, "Position yourself sideways and press SPACEBAR",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 0, 255), 2)
                else:
                    if self.state == "idle":
                        cv2.putText(img, "No person detected - press SPACEBAR when ready",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 2)

            # Info box (always visible)
            if self.state in ["idle", "finished"]:
                # Determine feedback and color
                if self.max_angle > 0:
                    feedback_text, color = self.get_feedback(self.max_angle)
                else:
                    feedback_text = "No measurement yet"
                    color = self.color_neutral

                # Draw info box
                cv2.rectangle(img, (10, 10), (450, 170), (0, 0, 0), -1)
                cv2.rectangle(img, (10, 10), (450, 170), color, 3)

                # Text in info box
                cv2.putText(img, f"Current angle: {int(self.current_angle)} deg",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            self.color_neutral, 2)

                if self.state == "finished":
                    cv2.putText(img, f"FINAL Maximum: {int(self.max_angle)} deg",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                color, 2)
                else:
                    cv2.putText(img, f"Maximum: {int(self.max_angle)} deg",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                self.color_neutral, 2)

                cv2.putText(img, feedback_text,
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color, 2)

                # Display optimal range
                cv2.putText(img, f"Optimal: {self.optimal_angle_min}-{self.optimal_angle_max} deg",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            self.color_neutral, 1)

                # State indicator
                state_text = "MEASUREMENT COMPLETE" if self.state == "finished" else "Press SPACE to start"
                cv2.putText(img, state_text,
                            (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            self.color_neutral, 1)

            # Help text at bottom
            cv2.putText(img, "SPACE = Start measurement | 'q' = Quit",
                        (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.color_neutral, 1)

            # Display image
            cv2.imshow("Bike Fit Analyzer", img)

            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting program...")
                break
            elif key == ord(' '):  # Spacebar
                if self.state in ["idle", "finished"]:
                    # Start countdown
                    self.state = "countdown"
                    self.countdown_start = current_time
                    print("Starting countdown...")

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        print("Program terminated.")


# Main program
if __name__ == "__main__":
    # Check if all required libraries are installed
    try:
        import cv2
        import cvzone

        print("All required libraries are installed.")
        print("-" * 60)
        print("NOTE FOR MACBOOK USERS:")
        print("If the camera doesn't work, please adjust")
        print("the 'camera_index' variable in __init__().")
        print("Common values: 0, 1, or 2")
        print("-" * 60)
    except ImportError as e:
        print("Error: Please install missing libraries:")
        print("pip install -r requirements.txt")
        print("or manually:")
        print("pip install opencv-python cvzone mediapipe numpy")
        exit(1)

    # Start analyzer
    analyzer = BikeFitAnalyzer()
    analyzer.run()