import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import time  # Added for timeout mechanism

def select_background():
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Background Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        return cv2.imread(file_path)
    return None

def main():
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    
    bg_image = select_background()
    if bg_image is None:
        print("No background image selected. Exiting...")
        cap.release()
        return
    
    plt.ion()  # Enable interactive mode for matplotlib
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # Placeholder image
    plt.axis('off')

    start_time = time.time()  # Start timer for timeout
    timeout = 300  # Timeout in seconds (e.g., 5 minutes)
    blur_mode = False  # Flag to toggle between replacement and blur modes

    def on_key(event):
        nonlocal blur_mode
        if event.key == '1':  # Toggle blur mode on pressing '1'
            blur_mode = not blur_mode
            print(f"Blur mode {'enabled' if blur_mode else 'disabled'}")
        elif event.key == 'q' or event.key == 'escape':  # Exit on pressing 'q' or 'Esc'
            print("Exiting...")
            plt.close()  # Close the plot window to exit the loop

    fig.canvas.mpl_connect('key_press_event', on_key)  # Bind key press event

    try:
        while cap.isOpened():
            # Check for timeout
            if time.time() - start_time > timeout:
                print("Timeout reached. Exiting...")
                break
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            bg_resized = cv2.resize(bg_image, (w, h))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = segment.process(rgb_frame)
            
            mask = result.segmentation_mask
            mask = np.clip(mask, 0, 1)  # Ensure mask values are between 0 and 1
            
            # Apply advanced edge detection to refine the mask
            mask = cv2.bilateralFilter(mask, 9, 75, 75)  # Smooth the mask while preserving edges
            edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)  # Detect edges
            edges = cv2.dilate(edges, None, iterations=2)  # Dilate edges for better blending
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Close gaps in edges
            
            # Convert edges to uint8 and invert for masking
            edges_mask = (edges == 0).astype(np.uint8) * 255
            refined_mask = cv2.bitwise_and((mask * 255).astype(np.uint8), (mask * 255).astype(np.uint8), mask=edges_mask)
            refined_mask = refined_mask / 255.0  # Normalize back to [0, 1]
            
            refined_mask = np.expand_dims(refined_mask, axis=-1)
            refined_mask = np.repeat(refined_mask, 3, axis=-1)
            
            # Apply Gaussian blur to smooth the mask for better blending
            refined_mask_blurred = cv2.GaussianBlur(refined_mask, (15, 15), 0)
            
            if blur_mode:
                # Blur the original background
                blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
                output = (frame * refined_mask_blurred + blurred_bg * (1 - refined_mask_blurred)).astype(np.uint8)
            else:
                # Replace the background with the selected image
                output = (frame * refined_mask_blurred + bg_resized * (1 - refined_mask_blurred)).astype(np.uint8)
            
            # Convert BGR to RGB for matplotlib
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            img_display.set_data(output_rgb)
            plt.pause(0.01)  # Pause to update the frame
    finally:
        cap.release()
        plt.close()

if __name__ == "__main__":
    main()
