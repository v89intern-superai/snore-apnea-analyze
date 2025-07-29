from ultralytics import YOLO
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load your YOLOv8 model
model = YOLO("/Users/gusgoodman/Documents/V89/Fall_Detection_Using_Yolov8/fall_det_1.pt")

# Open the default camera (0). If you have multiple cameras, try 1, 2, etc.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Optionally set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Persist tracks between frames
    results = model.track(frame, persist=True, conf=0.5)

    # Draw boxes/tracks on the frame
    annotated_frame = results[0].plot()

    # Show it
    cv2.imshow("YOLOv8 Real-Time Tracking", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
