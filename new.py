import cv2
import numpy as np
import mediapipe as mp
import time

# Define cloak color ranges in HSV
cloak_colors = {
    "red1": ([0, 120, 70], [10, 255, 255]),
    "red2": ([170, 120, 70], [180, 255, 255]),
    "blue": ([94, 80, 2], [126, 255, 255]),
    "black": ([0, 0, 0], [180, 255, 50])
}

# Initialize MediaPipe Selfie Segmentation
mp_selfie_seg = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_seg.SelfieSegmentation(model_selection=1)

# Start webcam
cap = cv2.VideoCapture(0)
time.sleep(2)  # Warm-up camera

# Countdown before background capture
for i in range(5, 0, -1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Capturing background in {i}s...", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imshow("Phantom Weave Setup", frame)
    cv2.waitKey(1000)

# Capture background
ret, background = cap.read()
background = cv2.flip(background, 1)
cv2.imshow("Background Captured", background)
cv2.waitKey(1000)

print("Phantom Weave Started!")

prev_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Start FPS calculation
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time + 1e-8))
    prev_frame_time = new_frame_time

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb_frame)
    mediapipe_mask = (results.segmentation_mask > 0.6).astype(np.uint8) * 255

    # Combine cloak mask with MediaPipe body mask for better edges
    combined_mask = cv2.bitwise_and(mask, mediapipe_mask)
    mask_inv = cv2.bitwise_not(combined_mask)

    # Cloak replacement with background
    res1 = cv2.bitwise_and(background, background, mask=combined_mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display FPS on screen
    cv2.putText(final_output, "FPS: " + str(fps), (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Phantom Weave", final_output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
