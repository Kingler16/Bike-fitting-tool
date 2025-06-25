import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import math
import time
from collections import deque


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
        self.current_angle = 0
        self.max_angle = 0  # Maximum angle while pedaling

        # Data collection for accuracy
        self.measurements = deque(maxlen=10)  # Moving average window

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
        self.state = "instructions"  # States: instructions, idle, countdown, measuring, finished
        self.countdown_start = None
        self.measurement_start = None

        # Duration settings
        self.countdown_duration = 15  # seconds to get on bike
        self.measurement_duration = 8  # seconds for measurement

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

    def calculate_moving_average(self):
        """
        Calculate moving average of measurements
        """
        if len(self.measurements) > 0:
            return np.mean(self.measurements)
        return 0

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

    def draw_instructions_screen(self, img):
        """
        Draw instructions overlay at program start
        """
        height, width = img.shape[:2]

        # Dark overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        # Main container
        container_width = 700
        container_height = 450
        container_x = (width - container_width) // 2
        container_y = (height - container_height) // 2

        # Container background
        self.draw_rounded_rect(img,
                               (container_x, container_y),
                               (container_x + container_width, container_y + container_height),
                               self.color_dark, -1, 20)

        # Header
        header_y = container_y + 50
        cv2.putText(img, "BIKE FIT ANALYZER",
                    (container_x + container_width // 2 - 150, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.color_primary, 3)

        # Instructions
        instructions = [
            "MEASUREMENT PROCESS:",
            "",
            "1. Position yourself sideways to the camera",
            "2. Press SPACE to start 10-second countdown",
            "3. Mount your bike during countdown",
            "4. Pedal normally for 5 seconds",
            "5. System tracks your maximum knee angle",
            "6. View results and adjust saddle if needed",
            "",
            "TIPS:",
            "- Ensure good lighting",
            "- Wear fitted clothing",
            "- Keep entire leg visible in frame",
            "- Pedal at your normal cadence"
        ]

        start_y = header_y + 60
        for i, instruction in enumerate(instructions):
            if instruction == "":
                continue
            elif instruction.endswith(":"):
                color = self.color_light
                font_size = 0.8
            else:
                color = self.color_mid
                font_size = 0.6

            cv2.putText(img, instruction,
                        (container_x + 50, start_y + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)

        # Footer
        footer_text = "Press SPACE to continue"
        text_size = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        footer_x = container_x + (container_width - text_size[0]) // 2
        footer_y = container_y + container_height - 30

        # Pulsing effect for footer
        pulse = int(128 + 127 * np.sin(time.time() * 3))
        footer_color = (pulse, pulse, pulse)

        cv2.putText(img, footer_text,
                    (footer_x, footer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, footer_color, 2)

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
        instruction = "Get on your bike and start pedalling"
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + radius + 50
        cv2.putText(img, instruction, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_light, 2)

    def draw_measurement_overlay(self, img, remaining_time):
        """
        Draw measurement progress overlay
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
        status_text = f"PEDAL NORMALLY - {int(remaining_time + 1)}s"
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

            # Maximum angle card (if measured)
            if self.max_angle > 0:
                _, feedback_short, feedback_color = self.get_feedback(self.max_angle)
                self.draw_modern_card(img, 250, 100, 200, 120,
                                      "MAXIMUM ANGLE", f"{int(self.max_angle)}",
                                      feedback_color)

                # Feedback card
                feedback_long, _, _ = self.get_feedback(self.max_angle)
                self.draw_modern_card(img, 470, 100, 250, 120,
                                      "ACTION REQUIRED", feedback_long,
                                      feedback_color)

                # Optimal range card
                optimal_text = f"{self.optimal_angle_min}-{self.optimal_angle_max} deg"
                self.draw_modern_card(img, self.ui_margin, 240, 200, 120,
                                      "OPTIMAL RANGE", optimal_text,
                                      self.color_success)

        # Instructions at bottom
        instruction_y = height - 30

        if self.state == "idle":
            instruction = "Press SPACE to start measurement | Q to quit"
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
        print("Loading camera and pose detection...")
        print("Instructions will appear on screen.")
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

            # Instructions screen
            if self.state == "instructions":
                self.draw_instructions_screen(img)

            # Always draw modern UI (except during countdown and instructions)
            elif self.state != "countdown":
                self.draw_modern_ui(img)

            # State handling
            if self.state == "countdown":
                remaining = self.countdown_duration - (current_time - self.countdown_start)
                if remaining > 0:
                    self.draw_countdown_overlay(img, remaining)
                else:
                    # Start measurement
                    self.state = "measuring"
                    self.measurement_start = current_time
                    self.measurements.clear()
                    self.max_angle = 0
                    print("Start pedaling!")

            elif self.state == "measuring":
                remaining = self.measurement_duration - (current_time - self.measurement_start)
                if remaining > 0:
                    self.draw_measurement_overlay(img, remaining)

                    # Detect pose and measure
                    img = self.detector.findPose(img, draw=False)
                    lmList, bboxInfo = self.detector.findPosition(img, draw=False)

                    if lmList:
                        try:
                            # Right side analysis
                            hip = (lmList[23][0], lmList[23][1])
                            knee = (lmList[25][0], lmList[25][1])
                            ankle = (lmList[27][0], lmList[27][1])

                            # Calculate angle
                            angle = self.calculate_angle(hip, knee, ankle)
                            self.current_angle = angle

                            # Add to moving average
                            self.measurements.append(angle)

                            # Calculate moving average
                            moving_avg = self.calculate_moving_average()

                            # Track maximum based on moving average
                            if moving_avg > self.max_angle:
                                self.max_angle = moving_avg

                            # Visualization
                            self.draw_angle_visualization(img, hip, knee, ankle,
                                                          angle, self.color_primary)

                            # Show moving average
                            cv2.putText(img, f"Moving Avg: {int(moving_avg)}",
                                        (img.shape[1] - 200, img.shape[0] - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        self.color_primary, 2)
                        except IndexError:
                            pass
                else:
                    # Measurement finished
                    self.state = "finished"
                    print(f"Maximum angle: {int(self.max_angle)} degrees")
                    feedback_text, _, _ = self.get_feedback(self.max_angle)
                    print(f"Result: {feedback_text}")
                    print("=" * 60)

            else:  # idle or finished states
                # Normal pose detection
                img = self.detector.findPose(img, draw=False)
                lmList, bboxInfo = self.detector.findPosition(img, draw=False)

                if lmList:
                    try:
                        # Right side analysis
                        hip = (lmList[23][0], lmList[23][1])
                        knee = (lmList[25][0], lmList[25][1])
                        ankle = (lmList[27][0], lmList[27][1])

                        # Calculate current angle
                        self.current_angle = self.calculate_angle(hip, knee, ankle)

                        # Get color based on state
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
                if self.state == "instructions":
                    # Move from instructions to idle
                    self.state = "idle"
                    print("\nReady to start measurement. Press SPACE again to begin.")
                elif self.state in ["idle", "finished"]:
                    # Start countdown
                    self.state = "countdown"
                    self.countdown_start = current_time
                    print("\nStarting new measurement...")

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