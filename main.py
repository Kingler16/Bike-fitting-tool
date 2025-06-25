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
        self.angle_history = []  # Store angle history for graph
        self.max_history_points = 150  # About 5 seconds at 30fps

        # Optimal angle range (can be adjusted)
        self.optimal_angle_min = 140
        self.optimal_angle_max = 150

        # Modern color palette
        self.color_primary = (255, 195, 0)  # Cyan/Blue
        self.color_success = (0, 255, 150)  # Green
        self.color_warning = (0, 140, 255)  # Orange
        self.color_danger = (100, 100, 255)  # Red
        self.color_dark = (40, 40, 40)  # Dark gray
        self.color_light = (250, 250, 250)  # Almost white
        self.color_mid = (180, 180, 180)  # Mid gray

        # Timer and measurement state variables
        self.state = "idle"  # States: idle, countdown, measuring, finished
        self.countdown_start = None
        self.measurement_start = None
        self.countdown_duration = 5  # seconds
        self.measurement_duration = 5  # seconds

        # UI Layout constants
        self.ui_margin = 30
        self.card_radius = 15

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
            return "RAISE SADDLE", "Too Low", self.color_warning
        elif angle > self.optimal_angle_max:
            return "LOWER SADDLE", "Too High", self.color_danger
        else:
            return "OPTIMAL", "Perfect Height", self.color_success

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness, radius):
        """
        Draw a rounded rectangle
        """
        x1, y1 = pt1
        x2, y2 = pt2

        if thickness < 0:  # Filled
            # Draw filled rounded rectangle
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:  # Outline only
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def draw_modern_card(self, img, x, y, width, height, title, content, color_accent):
        """
        Draw a modern card UI element
        """
        # Card background with shadow
        shadow_offset = 5
        cv2.rectangle(img, (x + shadow_offset, y + shadow_offset),
                      (x + width + shadow_offset, y + height + shadow_offset),
                      (20, 20, 20), -1)

        # Main card
        self.draw_rounded_rect(img, (x, y), (x + width, y + height),
                               self.color_dark, -1, self.card_radius)

        # Accent line
        cv2.rectangle(img, (x, y), (x + 5, y + height), color_accent, -1)

        # Title
        cv2.putText(img, title, (x + 20, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_light, 2)

        # Content
        if isinstance(content, list):
            for i, line in enumerate(content):
                cv2.putText(img, line, (x + 20, y + 60 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_mid, 1)
        else:
            cv2.putText(img, content, (x + 20, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_light, 2)

    def draw_progress_bar(self, img, x, y, width, height, progress, color):
        """
        Draw a modern progress bar
        """
        # Background
        self.draw_rounded_rect(img, (x, y), (x + width, y + height),
                               (60, 60, 60), -1, height // 2)

        # Progress
        if progress > 0:
            progress_width = int(width * progress)
            self.draw_rounded_rect(img, (x, y), (x + progress_width, y + height),
                                   color, -1, height // 2)

    def draw_angle_visualization(self, img, p1, p2, p3, angle, color):
        """
        Draw modern angle visualization
        """
        # Thicker, smoother lines
        cv2.line(img, p1, p2, color, 4)
        cv2.line(img, p2, p3, color, 4)

        # Modern joint visualization
        for point, size in [(p1, 10), (p2, 14), (p3, 10)]:
            # Outer circle
            cv2.circle(img, point, size, color, -1)
            # Inner circle for depth
            cv2.circle(img, point, size - 4, self.color_dark, -1)

        # Angle arc
        angle_rad = np.radians(angle)
        arc_radius = 50
        start_angle = np.degrees(np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
        end_angle = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]))

        cv2.ellipse(img, p2, (arc_radius, arc_radius), 0,
                    min(start_angle, end_angle), max(start_angle, end_angle),
                    color, 2)

        # Angle text with background
        angle_text = f"{int(angle)}"
        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = p2[0] - text_size[0] // 2
        text_y = p2[1] - 30

        # Text background
        padding = 8
        self.draw_rounded_rect(img,
                               (text_x - padding, text_y - text_size[1] - padding),
                               (text_x + text_size[0] + padding, text_y + padding),
                               self.color_dark, -1, 5)

        # Angle text
        cv2.putText(img, angle_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def draw_mini_graph(self, img, x, y, width, height, data, color):
        """
        Draw a mini line graph of angle history
        """
        if len(data) < 2:
            return

        # Background
        self.draw_rounded_rect(img, (x, y), (x + width, y + height),
                               (50, 50, 50), -1, 10)

        # Grid lines
        for i in range(1, 4):
            grid_y = y + int(height * i / 4)
            cv2.line(img, (x + 10, grid_y), (x + width - 10, grid_y),
                     (80, 80, 80), 1)

        # Data points
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val - min_val > 0 else 1

        points = []
        for i, value in enumerate(data):
            px = x + 10 + int((width - 20) * i / (len(data) - 1))
            py = y + height - 10 - int((height - 20) * (value - min_val) / range_val)
            points.append((px, py))

        # Draw line
        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], color, 2)

        # Draw optimal range
        if min_val <= self.optimal_angle_max and max_val >= self.optimal_angle_min:
            opt_y1 = y + height - 10 - int((height - 20) * (self.optimal_angle_max - min_val) / range_val)
            opt_y2 = y + height - 10 - int((height - 20) * (self.optimal_angle_min - min_val) / range_val)
            cv2.rectangle(img, (x + 10, opt_y1), (x + width - 10, opt_y2),
                          self.color_success, 1)

    def draw_countdown_overlay(self, img, remaining_time):
        """
        Draw modern countdown overlay
        """
        height, width = img.shape[:2]

        # Semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Circular progress
        center = (width // 2, height // 2)
        radius = 150

        # Background circle
        cv2.circle(img, center, radius, (80, 80, 80), 8)

        # Progress arc
        progress = 1 - (remaining_time / self.countdown_duration)
        end_angle = int(360 * progress)
        cv2.ellipse(img, center, (radius, radius), -90, 0, end_angle,
                    self.color_primary, 8)

        # Countdown number
        count_text = str(int(remaining_time + 1))
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(img, count_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, self.color_light, 4)

        # Instruction text
        instruction = "GET ON YOUR BIKE"
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + radius + 50
        cv2.putText(img, instruction, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_light, 2)

    def draw_measurement_progress(self, img, remaining_time):
        """
        Draw measurement progress bar
        """
        height, width = img.shape[:2]

        # Progress bar at top
        bar_height = 8
        progress = 1 - (remaining_time / self.measurement_duration)

        # Background
        cv2.rectangle(img, (0, 0), (width, bar_height), self.color_dark, -1)

        # Progress
        cv2.rectangle(img, (0, 0), (int(width * progress), bar_height),
                      self.color_primary, -1)

        # Status text
        status_text = f"MEASURING - {int(remaining_time + 1)}s"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = width // 2 - text_size[0] // 2

        # Text background
        self.draw_rounded_rect(img, (text_x - 20, 20),
                               (text_x + text_size[0] + 20, 60),
                               self.color_dark, -1, 20)

        cv2.putText(img, status_text, (text_x, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_primary, 2)

    def draw_modern_ui(self, img):
        """
        Draw the main modern UI
        """
        height, width = img.shape[:2]

        # Header
        header_height = 80
        cv2.rectangle(img, (0, 0), (width, header_height), self.color_dark, -1)

        # Logo/Title
        cv2.putText(img, "BIKE FIT ANALYZER", (self.ui_margin, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color_light, 2)

        # Status indicator
        status_color = self.color_mid
        status_text = "READY"

        if self.state == "measuring":
            status_color = self.color_primary
            status_text = "RECORDING"
        elif self.state == "finished":
            _, _, status_color = self.get_feedback(self.max_angle)
            status_text = "COMPLETE"

        status_x = width - 200
        cv2.circle(img, (status_x, 40), 8, status_color, -1)
        cv2.putText(img, status_text, (status_x + 20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_light, 2)

        # Main data cards
        if self.state in ["idle", "finished"]:
            # Current angle card
            self.draw_modern_card(img, self.ui_margin, 100, 200, 120,
                                  "CURRENT ANGLE", f"{int(self.current_angle)}",
                                  self.color_mid)

            # Maximum angle card
            if self.max_angle > 0:
                _, feedback_short, feedback_color = self.get_feedback(self.max_angle)
                self.draw_modern_card(img, self.ui_margin, 240, 200, 120,
                                      "MAXIMUM ANGLE", f"{int(self.max_angle)}",
                                      feedback_color)

                # Feedback card
                feedback_long, _, _ = self.get_feedback(self.max_angle)
                self.draw_modern_card(img, self.ui_margin, 380, 200, 120,
                                      "ACTION REQUIRED", feedback_long,
                                      feedback_color)

            # Mini graph
            if len(self.angle_history) > 10:
                self.draw_mini_graph(img, width - 250, 100, 220, 100,
                                     self.angle_history[-50:], self.color_primary)

        # Instructions at bottom
        instruction_y = height - 30

        if self.state == "idle":
            instruction = "Press SPACE to start measurement"
        elif self.state == "finished":
            instruction = "Press SPACE for new measurement | Q to quit"
        else:
            instruction = ""

        if instruction:
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = width // 2 - text_size[0] // 2
            cv2.putText(img, instruction, (text_x, instruction_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_mid, 1)

    def run(self):
        """
        Main program loop
        """
        print("Bike Fit Analyzer started!")
        print("=" * 60)
        print("INSTRUCTIONS:")
        print("1. Position your computer so you can see yourself sideways")
        print("2. Press SPACEBAR to start measurement")
        print("3. You have 5 seconds to get on your bike")
        print("4. Pedal for 5 seconds during measurement")
        print("5. Press Q to quit")
        print("=" * 60)

        while True:
            success, img = self.cap.read()
            if not success:
                print("Error reading camera")
                break

            # Mirror image for more natural display
            img = cv2.flip(img, 1)

            # Apply slight blur for smoother appearance
            img = cv2.bilateralFilter(img, 5, 50, 50)

            # State machine for measurement process
            current_time = time.time()

            # Always draw modern UI (except during countdown)
            if self.state != "countdown":
                self.draw_modern_ui(img)

            if self.state == "countdown":
                remaining = self.countdown_duration - (current_time - self.countdown_start)
                if remaining > 0:
                    self.draw_countdown_overlay(img, remaining)
                else:
                    # Start measurement
                    self.state = "measuring"
                    self.measurement_start = current_time
                    self.max_angle = 0  # Reset max angle
                    self.angle_history = []  # Reset history
                    print("Measurement started - pedal now!")

            elif self.state == "measuring":
                remaining = self.measurement_duration - (current_time - self.measurement_start)
                if remaining > 0:
                    self.draw_measurement_progress(img, remaining)

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
                            self.angle_history.append(self.current_angle)

                            # Keep only recent history
                            if len(self.angle_history) > self.max_history_points:
                                self.angle_history.pop(0)

                            # Update maximum angle during measurement
                            if self.current_angle > self.max_angle:
                                self.max_angle = self.current_angle

                            # Visualization during measurement
                            self.draw_angle_visualization(img, hip, knee, ankle,
                                                          self.current_angle, self.color_primary)
                        except IndexError:
                            pass
                else:
                    # Measurement finished
                    self.state = "finished"
                    print(f"Measurement complete! Maximum angle: {int(self.max_angle)} degrees")
                    feedback_text, _, _ = self.get_feedback(self.max_angle)
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

                        # Update history even in idle for graph
                        if self.state == "idle":
                            self.angle_history.append(self.current_angle)
                            if len(self.angle_history) > self.max_history_points:
                                self.angle_history.pop(0)

                        # Get color based on max angle (if we have one)
                        if self.state == "finished" and self.max_angle > 0:
                            _, _, color = self.get_feedback(self.max_angle)
                        else:
                            color = self.color_mid

                        # Visualization
                        self.draw_angle_visualization(img, hip, knee, ankle,
                                                      self.current_angle, color)

                    except IndexError:
                        pass

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
        print("=" * 60)
        print("NOTE FOR MACBOOK USERS:")
        print("If the camera doesn't work, please adjust")
        print("the 'camera_index' variable in __init__().")
        print("Common values: 0, 1, or 2")
        print("=" * 60)
    except ImportError as e:
        print("Error: Please install missing libraries:")
        print("pip install -r requirements.txt")
        print("or manually:")
        print("pip install opencv-python cvzone mediapipe numpy")
        exit(1)

    # Start analyzer
    analyzer = BikeFitAnalyzer()
    analyzer.run()