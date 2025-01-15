# AI Gesture Control

AI Gesture Control is a Python-based project that enables users to control their computer using hand gestures. It leverages computer vision techniques to detect and interpret hand movements, providing an intuitive and touchless interface for human-computer interaction.

## Features

- **Hand Tracking**: Detects and tracks hand movements in real-time.
- **Gesture Recognition**: Identifies specific hand gestures to execute corresponding commands.
- **Face Detection**: Includes functionality for detecting faces, enhancing interaction capabilities.
- **Pose Estimation**: Estimates body poses to interpret user movements.

## Installation

### Prerequisites
- Python 3.6 or higher
- Webcam for gesture detection

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/iamscratches/AI_gesture_control.git
   cd AI_gesture_control
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Interact**:
   - Ensure your webcam is connected.
   - Perform predefined hand gestures to control your computer.

## Project Structure

- `HandTracking/`: Contains modules related to hand tracking functionality.
- `FaceDetection/`: Includes scripts for detecting faces.
- `PoseEstimation/`: Houses code for estimating body poses.
- `main.py`: The main script to run the application.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your enhancements.
.

## Acknowledgements

- [OpenCV](https://opencv.org/) for computer vision functionalities.
- [MediaPipe](https://mediapipe.dev/) for hand and pose detection models.

---

Feel free to customize this `README.md` to better fit your project's specifics and to add any additional information that might be helpful for users.
