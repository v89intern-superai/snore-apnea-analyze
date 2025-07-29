from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque, defaultdict
import time
from datetime import datetime
import os

# Load the custom YOLOv12 model
model = YOLO(
    "/Users/gusgoodman/Documents/V89/Human-Activity-Recognition-with-YOLOv12-for-Smart-Cities/best.pt"
)
class_names = model.names

# Update the colors dictionary to include normal and abnormal colors for new activities
colors = {
    "walking": {"normal": (0, 255, 0), "abnormal": (0, 165, 255)},
    "standing": {"normal": (255, 0, 0), "abnormal": (0, 0, 255)},
    "running": {"normal": (255, 255, 0), "abnormal": (0, 0, 255)},
    "fighting": {"normal": (128, 0, 128), "abnormal": (0, 0, 255)},
    "jumping": {"normal": (0, 255, 255), "abnormal": (0, 0, 255)},
    "robbery": {"normal": (255, 165, 0), "abnormal": (0, 0, 255)},
    "sitting": {"normal": (0, 128, 255), "abnormal": (0, 0, 255)},
    "armed": {"normal": (128, 128, 128), "abnormal": (0, 0, 255)},
    "lying_down": {"normal": (0, 255, 128), "abnormal": (0, 0, 255)}
}

# Background subtraction & crowd thresholds
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
crowd_density_threshold = 0.3
activity_thresholds = {
    "walking": 0.25,
    "standing": 0.2,
    "running": 0.15,
    "fighting": 0.1,
    "jumping": 0.2,
    "robbery": 0.05,
    "sitting": 0.3,
    "armed": 0.05,
    "lying_down": 0.2
}

# Video capture
video_path = "/Users/gusgoodman/Documents/V89/Human-Activity-Recognition-with-YOLOv12-for-Smart-Cities/demo.mp4"
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

# Utility functions
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_activity_abnormal(activity, density, sudden):
    always_abnormal = {"fighting", "robbery", "armed", "lying_down"}
    if activity in always_abnormal:
        return True
    if activity in {"standing", "walking"}:
        return sudden
    thresh = activity_thresholds.get(activity, 0.3)
    return density > thresh

def detect_sudden_movement(prev_boxes, curr_boxes, prox_thresh=100):
    for (x1, y1, x2, y2, _), (X1, Y1, X2, Y2, _) in zip(prev_boxes, curr_boxes):
        c1 = ((x1 + x2)//2, (y1 + y2)//2)
        c2 = ((X1 + X2)//2, (Y1 + Y2)//2)
        if calculate_distance(c1, c2) > prox_thresh:
            return True
    return False

# Prepare output folders
abnormal_frames_folder = "abnormal_frames"
heatmap_folder = "heatmaps"
for folder in (abnormal_frames_folder, heatmap_folder):
    os.makedirs(folder, exist_ok=True)

# Heatmap handling
def save_heatmap(hm, ts=None):
    if hm is None:
        return
    norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
    cm = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(heatmap_folder, f"heatmap_{ts}.jpg"), cm)

# Main loop variables
heatmap = None
prev_boxes = []
frame_count = 0
last_heatmap_time = time.time()
heatmap_interval = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Crowd analysis
    fg = backSub.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    density = cv2.countNonZero(fg) / (frame.shape[0]*frame.shape[1])
    heatmap = fg.astype(np.float32) if heatmap is None else heatmap + fg.astype(np.float32)
    
    if time.time() - last_heatmap_time >= heatmap_interval:
        save_heatmap(heatmap)
        last_heatmap_time = time.time()

    # Detection
    results = model.predict(frame)
    curr_boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            curr_boxes.append((x1, y1, x2, y2, label))

    sudden = detect_sudden_movement(prev_boxes, curr_boxes)

    # Annotate
    for x1, y1, x2, y2, label in curr_boxes:
        abnormal = is_activity_abnormal(label, density, sudden)
        col = colors[label]["abnormal" if abnormal else "normal"]
        text = f"{label} ({'ABNORMAL' if abnormal else 'Normal'})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if abnormal:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(abnormal_frames_folder, f"abn_{label}_{ts}.jpg"), frame)
            print(f"ALERT: abnormal {label} detected at {ts}")

    prev_boxes = curr_boxes

    # Overlay HUD
    cv2.putText(frame, f"Density: {density:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Humans: {len(curr_boxes)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    overall = "ABNORMAL" if any(is_activity_abnormal(lbl, density, sudden) for *_,lbl in curr_boxes) else "NORMAL"
    cv2.putText(frame, overall, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,0,255) if overall=="ABNORMAL" else (0,255,0), 2)

    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

save_heatmap(heatmap)
cap.release()
cv2.destroyAllWindows()