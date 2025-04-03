import cv2
import numpy as np
import mediapipe as mp
import time
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPixmap
import sys
import os

ICON_PATH = "C:\\Users\\HP\\Desktop\\Others\\Phantom-Weave-\\Icon1.webp"

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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

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

# Capture background immediately
ret, background = cap.read()
background = cv2.flip(background, 1)
cv2.imshow("Background Captured", background)
cv2.waitKey(1000)

print("Phantom Weave Ready! Show thumbs-up to start invisibility or thumbs-down to stop it.")

prev_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect hand gestures
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks for better visualization
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Check for thumbs-up gesture
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            if thumb_tip.y < thumb_ip.y < index_tip.y:  # Thumbs-up condition
                invisibility_enabled = True
                time.sleep(0.5)  # Debounce gesture detection
            elif thumb_tip.y > thumb_ip.y > index_tip.y:  # Thumbs-down condition
                invisibility_enabled = False
                time.sleep(0.5)  # Debounce gesture detection

    # Start FPS calculation
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time + 1e-8))
    prev_frame_time = new_frame_time

    if invisibility_enabled:
        mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Combine all color masks
        for color, (lower, upper) in cloak_colors.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.bitwise_or(mask, color_mask)

        # Edge refinement (Phase 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
        mask = cv2.bilateralFilter(mask, d=9, sigmaColor=75, sigmaSpace=75)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # MediaPipe segmentation fallback
        results = selfie_segmentation.process(rgb_frame)
        mediapipe_mask = (results.segmentation_mask > 0.6).astype(np.uint8) * 255

        # Combine cloak mask with MediaPipe body mask for better edges
        combined_mask = cv2.bitwise_and(mask, mediapipe_mask)
        mask_inv = cv2.bitwise_not(combined_mask)

        # Cloak replacement with background
        res1 = cv2.bitwise_and(background, background, mask=combined_mask)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    else:
        final_output = frame  # Show normal frame if invisibility is disabled

    # Display FPS on screen
    cv2.putText(final_output, "FPS: " + str(fps), (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display invisibility status
    status_text = "Invisibility: ON" if invisibility_enabled else "Invisibility: OFF"
    cv2.putText(final_output, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Phantom Weave", final_output)
    out.write(final_output)  # Write frame to video file

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
out.release()
hands.close()
cv2.destroyAllWindows()

# Ask user if they want to save the video using PyQt5
app = QApplication(sys.argv)
msg_box = QMessageBox()
msg_box.setWindowTitle("Save Video")
msg_box.setText("Do you want to save the video?")
msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
msg_box.setIconPixmap(QPixmap(ICON_PATH))  # Use the specified icon
response = msg_box.exec_()

if response == QMessageBox.Yes:
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Video Saved")
    msg_box.setText("Video saved as 'phantom_weave.avi'.")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setIconPixmap(QPixmap(ICON_PATH))  # Use the specified icon
    msg_box.exec_()
else:
    os.remove("phantom_weave.avi")
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Video Discarded")
    msg_box.setText("Video discarded.")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setIconPixmap(QPixmap(ICON_PATH))  # Use the specified icon
    msg_box.exec_()
