import cv2
import numpy as np

# Global variables for color bounds and background
lower_bound = None
upper_bound = None
background = None

# Function to select cloak color with mouse click
def pick_color(event, x, y, flags, param):
    global lower_bound, upper_bound, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        color = frame[y, x]
        b, g, r = color
        lower_bound = np.array([max(0, b - 40), max(0, g - 40), max(0, r - 40)])
        upper_bound = np.array([min(255, b + 40), min(255, g + 40), min(255, r + 40)])
        print(f"Color chosen: {color}")
        print(f"Lower bound: {lower_bound}")
        print(f"Upper bound: {upper_bound}")

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Select Color")
cv2.setMouseCallback("Select Color", pick_color)

print("Click on the cloak color. Press 'b' to capture background. Press ESC to exit.")

# Capture background frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Select Color", frame)

    key = cv2.waitKey(1)
    if key == ord('b'):
        background = frame.copy()
        print("Background saved!")
    elif key == 27:
        break

cv2.destroyWindow("Select Color")

if background is not None and lower_bound is not None:
    print("Invisibility cloak activated. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Create mask for selected color
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Create inverse mask
        mask_inv = cv2.bitwise_not(mask)

        # Replace cloak area with background
        cloak_part = cv2.bitwise_and(background, background, mask=mask)
        non_cloak_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Combine results
        output = cv2.addWeighted(cloak_part, 1, non_cloak_part, 1, 0)

        cv2.imshow("Invisibility Cloak", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
