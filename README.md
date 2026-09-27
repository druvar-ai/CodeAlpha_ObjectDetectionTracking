# CodeAlpha Object Detection and Tracking

This is a computer-vision project created for **CodeAlpha Artificial Intelligence Internship Task 4**.

It combines a pretrained YOLO object detection model with the SORT (Simple Online and Realtime Tracking) algorithm to detect objects in real-time video streams and track them consistently across frames.

## Features

* **Webcam & Video File Input**: Seamlessly switch between processing local video files or a live webcam feed.
* **Pretrained Object Detection**: Utilizes the lightweight `yolov8n.pt` model for fast, CPU-friendly inference.
* **Object Tracking**: Uses the SORT algorithm to associate detections across frames, generating persistent tracking IDs.
* **Bounding Boxes & Labels**: Draws clean bounding boxes, detected object class names, and unique tracking IDs on the frame.
* **Real-time Display**: Renders the video feed directly to your screen using OpenCV.
* **Configurable Confidence Threshold**: Filter out low-confidence detections easily via command-line arguments.

## Technologies

* **Python 3.x**
* **OpenCV (`opencv-python`)**: For video capture, frame processing, and GUI display.
* **Ultralytics YOLO**: The core object detection neural network framework.
* **SORT (Simple Online and Realtime Tracking)**: A Kalman-filter and Hungarian-algorithm based tracking implementation.
* **NumPy & FilterPy**: Under-the-hood dependencies for SORT's matrix operations.

## Architecture

The processing pipeline for every frame is as follows:

```text
Webcam / Video Stream
      ↓
OpenCV (Frame capture)
      ↓
YOLO Inference (Generates boxes, classes, confidence scores)
      ↓
Confidence Filtering (Ignores weak detections)
      ↓
SORT Tracker (Associates current boxes with previous tracks via IoU & Kalman filters)
      ↓
Tracked Objects (Returns persistent tracking IDs)
      ↓
OpenCV (Draw Bounding Boxes + Class Labels + IDs)
      ↓
Display (Show frame to user)
```

### How detection works
The Ultralytics YOLO model receives the raw image frame and passes it through its neural network. It returns coordinates for bounding boxes, integer class predictions (like `0` for person, `2` for car), and confidence scores representing the model's certainty.

### How tracking works
Because YOLO evaluates every frame independently, it has no concept of "the same person" moving over time. The **SORT** algorithm bridges this gap by utilizing a Kalman filter to predict where a moving object will be in the next frame. When the next frame arrives, SORT calculates the Intersection over Union (IoU) between its predicted boxes and the new YOLO detections. It uses the Hungarian algorithm to optimally associate the new detections with the existing tracks, assigning the same ID across consecutive frames.

## Requirements

* Python 3.x
* A webcam (if using webcam mode)
* Supports execution on CPU (GPU is completely optional).

## Installation

1. Open a terminal and navigate to the project directory:
   ```bash
   cd CodeAlpha_ObjectDetectionTracking
   ```
2. Create a virtual environment:
   * **Windows**: `python -m venv .venv`
   * **macOS/Linux**: `python3 -m venv .venv`
3. Activate the virtual environment:
   * **Windows**: `.\.venv\Scripts\activate`
   * **macOS/Linux**: `source .venv/bin/activate`
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### First-Run Model Download
This project uses the `yolov8n.pt` model weights. You do not need to download this manually. When you run the application for the very first time, the `ultralytics` package will automatically download the necessary model weights file (approx. 6MB) into your project folder.

## Running the Application

### 1. Webcam Mode (Default)
To process the live feed from your default webcam:
```bash
python main.py
```

### 2. Video File Mode
To process a pre-recorded video file instead of the webcam, use the `--source` argument:
```bash
python main.py --source my_video.mp4
```

### 3. Adjusting Confidence
To ignore weak detections (for example, hiding objects with less than 50% confidence):
```bash
python main.py --confidence 0.50
```

## Controls

* **`Q`** or **`ESC`**: Safely stop processing, close the display window, and quit the application.

## CodeAlpha requirement mapping

* **Video/webcam input** → Implemented (via OpenCV `VideoCapture` and `--source`).
* **Pretrained detector** → Implemented (Uses Ultralytics YOLOv8 nano).
* **Object detection** → Implemented (Processes frames iteratively).
* **Bounding boxes** → Implemented (Drawn via OpenCV).
* **Object tracking** → Implemented (Uses the SORT algorithm to associate IDs).
