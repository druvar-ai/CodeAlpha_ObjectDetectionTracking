import argparse
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="CodeAlpha AI Task 4: Object Detection & Tracking")
    parser.add_argument("--source", type=str, default="0", help="Video file path or '0' for webcam")
    parser.add_argument("--confidence", type=float, default=0.40, help="Confidence threshold for YOLO detections")
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    # 2. Initialize YOLO Model
    print("Loading YOLO model...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)

    # 3. Initialize SORT Tracker
    # max_age: Maximum number of frames to keep alive a track without associated detections.
    # min_hits: Minimum number of associated detections before track is initialised.
    # iou_threshold: Minimum IOU for match.
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)

    # 4. Open Video Stream
    print(f"Opening source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}.")
        sys.exit(1)

    print("Starting processing. Press 'q' or 'ESC' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.")
            break

        # 5. Run YOLO Inference
        # verbose=False reduces console spam
        results = model(frame, verbose=False)
        
        # 6. Parse Detections
        # Prepare detections for SORT: [[x1, y1, x2, y2, score, class_id], ...]
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < args.confidence:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                
                detections.append([x1, y1, x2, y2, conf, cls_id])

        detections = np.array(detections)
        if len(detections) == 0:
            detections = np.empty((0, 6))

        # 7. Update SORT Tracker
        tracked_objects = tracker.update(detections)

        # 8. Draw Bounding Boxes and IDs
        for obj in tracked_objects:
            # Output from SORT: [x1, y1, x2, y2, track_id, class_id]
            x1, y1, x2, y2 = map(int, obj[:4])
            track_id = int(obj[4])
            class_id = int(obj[5]) if len(obj) > 5 and obj[5] is not None else -1
            
            # Get class name
            class_name = model.names[class_id] if class_id in model.names else "Unknown"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} | ID: {track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add Overlay Info
        cv2.putText(frame, "CodeAlpha Object Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Objects tracked: {len(tracked_objects)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 9. Display Frame
        cv2.imshow("CodeAlpha Task 4 - Detection & Tracking", frame)

        # 10. Handle Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC
            print("Quit requested by user.")
            break

    # 11. Cleanup Resources
    cap.release()
    cv2.destroyAllWindows()
    print("Cleaned up resources successfully.")

if __name__ == "__main__":
    main()
