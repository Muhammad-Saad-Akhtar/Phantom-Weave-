<!-- # Phantom Weave

Phantom Weave is an innovative computer vision project that combines real-time video processing with gesture-based controls to create two exciting features:
1. **Invisibility Cloak**: Make yourself invisible using a specific cloak color.
2. **Virtual Background**: Replace your background with preloaded images or a blurred effect.

This project leverages cutting-edge technologies like OpenCV, MediaPipe, and PyQt/Tkinter for seamless user interaction and real-time video manipulation.

---

## Features

### 1. Invisibility Cloak
- Detects specific cloak colors (Red,Blue,Black) in real-time and replaces them with the captured background.
- Gesture-based control using thumbs-up (to enable invisibility) and thumbs-down (to disable invisibility).
- Uses MediaPipe's Selfie Segmentation for enhanced edge refinement.

### 2. Virtual Background
- Replace your background with preloaded images or a blurred effect.
- Switch between up to 9 preloaded backgrounds using keyboard shortcuts.
- Includes a timeout feature to automatically exit after a specified duration.

### 3. Graphical User Interface (GUI)
- A black-themed menu built with PyQt6 for easy navigation.
- Custom dialog boxes with a professional design for user prompts.

---

## Technologies Used
- **Python**: Core programming language.
- **OpenCV**: For real-time video processing.
- **MediaPipe**: For gesture recognition and segmentation.
- **PyQt6**: For the main menu and GUI dialogs.
- **Tkinter**: For file selection and custom dialogs in the Virtual Background feature.
- **Pillow**: For handling `.webp` icons and images.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Install the required Python libraries:
  ```bash
  pip install opencv-python mediapipe PyQt6 Pillow
  ```

### Clone the Repository
```bash
git clone https://github.com/your-repo/phantom-weave.git
cd phantom-weave
```

---

## Usage

### 1. Launch the Menu
Run the `menu.py` file to access the main menu:
```bash
python menu.py
```

### 2. Menu Options
- **Invisibility Cloak**: Select this option to start the invisibility feature.
- **Virtual Background**: Select this option to replace your background with preloaded images or a blur effect.
- **Exit**: Close the application.

### 3. Virtual Background Controls
- Press `0`: Enable blur mode.
- Press `1-9`: Switch to preloaded backgrounds.
- Press `q` or `ESC`: Exit the application.

---

## File Structure
```
Phantom-Weave-
│
├── invisibility.py       # Invisibility Cloak feature
├── background.py         # Virtual Background feature
├── menu.py               # Main menu for navigation
├── Icon1.webp            # Icon used in dialogs and GUI
├── Backgrounds/          # Folder containing preloaded background images
│   ├── 1_library.jpg
│   ├── 2_sunset.jpg
│   ├── ...
│   └── 9_street.jpg
└── README.md             # Project documentation
```

---

## How It Works

### Invisibility Cloak
1. Captures the background before starting the invisibility effect.
2. Detects specific cloak colors using HSV color ranges.
3. Combines MediaPipe's segmentation mask with the cloak mask for better edge refinement.
4. Replaces the cloak area with the captured background.

### Virtual Background
1. Uses MediaPipe's segmentation to separate the user from the background.
2. Allows switching between preloaded backgrounds or enabling a blur effect.
3. Saves the video output if the user chooses to do so.

---

## Customization

### Adding New Backgrounds
1. Place your image files in the `Backgrounds/` folder.
2. Update the `background_paths` list in `background.py` with the new file paths.

### Changing Cloak Colors
1. Modify the `cloak_colors` dictionary in `invisibility.py` to add or adjust HSV ranges for new colors.

---

## Known Issues
- **Performance**: Real-time processing may lag on low-end systems.
- **Icon Compatibility**: Ensure the icon file is in `.webp` format or convert it to `.png` for compatibility.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [PyQt](https://riverbankcomputing.com/software/pyqt/intro)
- [Pillow](https://python-pillow.org/)

---

## Contact
For questions or feedback, please contact:
- **Name**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [Your GitHub Profile](https://github.com/your-profile) -->