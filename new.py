import cv2
import numpy as np
import mediapipe as mp
import time
from PyQt6.QtWidgets import QApplication, QMessageBox, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
import sys
import os
from rembg import remove  # Import rembg for background removal

# Define cloak color ranges in HSV for detecting all shades
cloak_colors = {
    "red1": ([0, 50, 50], [10, 255, 255]),  # Broad red range
    "red2": ([170, 50, 50], [180, 255, 255]),
    "blue": ([85, 50, 50], [135, 255, 255]),  # Broad blue range
    "black": ([0, 0, 0], [180, 255, 60])  # Broad black range
}

# Initialize MediaPipe Selfie Segmentation
mp_selfie_seg = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_seg.SelfieSegmentation(model_selection=1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Gesture control variables
invisibility_enabled = False

# Start webcam
cap = cv2.VideoCapture(0)
time.sleep(2)  # Warm-up camera

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("phantom_weave.avi", fourcc, 20.0, (frame_width, frame_height))

print("Phantom Weave Ready! Show thumbs-up to start invisibility or thumbs-down to stop it.")

prev_frame_time = 0

def invisibility_effect(frame, background, cloak_colors, selfie_segmentation, hands):
    """Applies the invisibility effect."""
    global invisibility_enabled  # Declare as global to modify the global variable
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        thumbs_up_detected = False
        thumbs_down_detected = False

        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            if thumb_tip.y < thumb_ip.y < index_tip.y:
                thumbs_up_detected = True
            elif thumb_tip.y > thumb_ip.y > index_tip.y:
                thumbs_down_detected = True

        if thumbs_up_detected:
            invisibility_enabled = True
            time.sleep(0.5)
        elif thumbs_down_detected:
            invisibility_enabled = False
            time.sleep(0.5)

    if invisibility_enabled:
        mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        for color, (lower, upper) in cloak_colors.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.bitwise_or(mask, color_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
        mask = cv2.bilateralFilter(mask, d=9, sigmaColor=75, sigmaSpace=75)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        results = selfie_segmentation.process(rgb_frame)
        mediapipe_mask = (results.segmentation_mask > 0.6).astype(np.uint8) * 255

        combined_mask = cv2.bitwise_and(mask, mediapipe_mask)
        mask_inv = cv2.bitwise_not(combined_mask)

        res1 = cv2.bitwise_and(background, background, mask=combined_mask)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    else:
        final_output = frame

    return final_output

def background_change(frame, custom_background):
    """Replaces the background with a custom image."""
    try:
        # Resize frame to a standard size for rembg
        resized_frame = cv2.resize(frame, (256, 256))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Debugging: Log frame dimensions and type
        print(f"Processing resized frame with shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")

        # Remove background using rembg
        removed_background = remove(rgb_frame)

        # Convert the result back to BGR for OpenCV
        person_only = cv2.imdecode(np.frombuffer(removed_background, np.uint8), cv2.IMREAD_UNCHANGED)

        if person_only is None:
            print("Error: rembg failed to process the frame.")
            return frame  # Return the original frame as a fallback

        # Resize custom background to match frame dimensions
        custom_background_resized = cv2.resize(custom_background, (frame.shape[1], frame.shape[0]))

        # Replace the background
        if person_only.shape[2] == 4:  # If alpha channel exists
            alpha_channel = person_only[:, :, 3] / 255.0
            for c in range(3):  # Blend each color channel
                custom_background_resized[:, :, c] = (
                    alpha_channel * person_only[:, :, c] +
                    (1 - alpha_channel) * custom_background_resized[:, :, c]
                )
            modified_frame = custom_background_resized
        else:
            modified_frame = person_only  # Fallback if no alpha channel

        return modified_frame

    except Exception as e:
        print(f"Error in background_change: {e}")
        return frame  # Return the original frame as a fallback

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phantom Weave Menu")
        self.setGeometry(100, 100, 800, 600)  # Adjusted size for video display

        # Central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Buttons for functionalities
        invisibility_button = QPushButton("Invisibility Effect")
        invisibility_button.clicked.connect(self.start_invisibility)

        background_button = QPushButton("Background Change")
        background_button.clicked.connect(self.start_background_change)

        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close_application)

        # Add buttons to layout
        layout.addWidget(invisibility_button)
        layout.addWidget(background_button)
        layout.addWidget(exit_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Timer for video display
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Variables for background change
        self.custom_background = None
        self.is_background_change_active = False

    def update_frame(self):
        """Updates the video frame in the QLabel."""
        ret, frame = cap.read()
        if not ret:
            self.timer.stop()
            return

        frame = cv2.flip(frame, 1)

        if self.is_background_change_active and self.custom_background is not None:
            frame = background_change(frame, self.custom_background)

        # Convert frame to QImage for QLabel
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Display the image in the QLabel
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def start_invisibility(self):
        """Starts the invisibility effect."""
        global background, prev_frame_time  # Declare global variables
        self.hide()  # Hide the menu while the functionality runs

        # Capture background when the user selects this option
        ret, background = cap.read()
        if not ret:
            print("Error: Unable to capture background.")
            self.show()
            return
        background = cv2.flip(background, 1)

        # Start the timer for video display
        self.timer.start(30)

        # Wait for ESC key to stop invisibility
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.timer.stop()
        self.show()  # Reopen the menu after the functionality ends

    def start_background_change(self):
        """Starts the background change functionality."""
        self.hide()  # Hide the menu while the functionality runs

        # Open file dialog to select a custom background image
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if not file_path:  # If no file is selected, return to the menu
            print("No file selected.")
            self.show()
            return

        self.custom_background = cv2.imread(file_path)
        if self.custom_background is None:
            print("Error: Unable to load the selected image.")
            self.show()
            return

        self.custom_background = cv2.resize(self.custom_background, (frame_width, frame_height))
        self.is_background_change_active = True

        # Start the timer for video display
        self.timer.start(30)

    def close_application(self):
        """Closes the application."""
        self.timer.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_menu = MainMenu()
    main_menu.show()
    sys.exit(app.exec())
