import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, filedialog, messagebox
import time

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

    # Hardcoded the paths for up to 9 background images here
    background_paths = [
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\1_library.jpg",  
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\2_sunset.jpg", 
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\3_nightcity.webp", 
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\4_space.jpeg", 
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\5_colors.jpg", 
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\6_rainywindow.jpg", 
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\7_garden.jpg",  
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\8_mountains.jpg",  
    r"C:\Users\HP\Desktop\Others\Phantom-Weave-\Backgrounds\9_street.jpg",
]

    # Preload and resize backgrounds
    backgrounds = []
    for i, path in enumerate(background_paths):
        if path:
            bg = cv2.imread(path)
            if bg is None:
                print(f"Warning: Background {i + 1} ({path}) is invalid or missing. Skipping.")
                backgrounds.append(None)
            else:
                backgrounds.append(bg)
        else:
            print(f"Background {i + 1} skipped.")
            backgrounds.append(None)

    blur_mode = True  # Start with blur mode enabled
    current_bg_index = 0  # 0 for blur, 1-10 for backgrounds
    timeout = 300
    start_time = time.time()

    # Resize backgrounds to match webcam frame size
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        cap.release()
        return
    h, w, _ = frame.shape
    for i in range(len(backgrounds)):
        if backgrounds[i] is not None:
            backgrounds[i] = cv2.resize(backgrounds[i], (w, h))

    # Initialize video writer but defer saving decision
    out = None
    video_frames = []

    def on_key(key):
        nonlocal blur_mode, current_bg_index
        if key == ord('0'):  # Blur mode
            blur_mode = True
            current_bg_index = 0
            print("Blur mode enabled")
        elif ord('1') <= key <= ord('9'):  # Backgrounds 1-9
            blur_mode = False
            current_bg_index = key - ord('0')
            if backgrounds[current_bg_index - 1] is None:
                print(f"Background {current_bg_index} is missing. Skipping.")
                return
            print(f"Background {current_bg_index} selected")
        elif key == ord('q') or key == 27:  # Exit
            print("Exiting...")
            return False
        return True

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = segment.process(rgb_frame)
            
            mask = result.segmentation_mask
            mask = np.clip(mask, 0, 1)  # Ensure mask values are between 0 and 1
            
            # Refine the mask with Gaussian blur
            refined_mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            if blur_mode:
                # Blur the original background
                blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
                output = (frame * refined_mask[..., None] + blurred_bg * (1 - refined_mask[..., None])).astype(np.uint8)
            else:
                # Use the selected background
                selected_bg = backgrounds[current_bg_index - 1]
                if selected_bg is not None:
                    output = (frame * refined_mask[..., None] + selected_bg * (1 - refined_mask[..., None])).astype(np.uint8)
                else:
                    print(f"Background {current_bg_index} is missing. Defaulting to blur mode.")
                    blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
                    output = (frame * refined_mask[..., None] + blurred_bg * (1 - refined_mask[..., None])).astype(np.uint8)

            # Store frames in memory for saving later
            video_frames.append(output)

            try:
                # Display the output using OpenCV
                cv2.imshow("Virtual Background", output)
            except cv2.error:
                print("GUI functions are not supported.")

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if not on_key(key):
                break

            # Ensure the OpenCV window remains responsive
            if cv2.getWindowProperty("Virtual Background", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Exiting...")
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            print("Error: Unable to destroy OpenCV windows. Skipping cleanup.")

        # Ask the user if they want to save the video using a GUI prompt
        Tk().withdraw()  # Hide the root window
        save_video = messagebox.askyesno("Save Video", "Do you want to save the video output?")

        if save_video:
            print("Saving video...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))
            for frame in video_frames:
                out.write(frame)
            out.release()
            print("Video saved as 'output.avi'.")
        else:
            print("Video not saved.")

if __name__ == "__main__":
    main()
