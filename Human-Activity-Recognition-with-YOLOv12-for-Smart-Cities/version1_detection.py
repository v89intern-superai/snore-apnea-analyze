from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time
import datetime
import os
import winsound
import threading
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import pyresearch

# Load the custom YOLO model
model = YOLO("best.pt")
class_names = model.names

# Update the colors dictionary to include normal and abnormal colors for new activities
colors = {
    "walking": {"normal": (0, 255, 0), "abnormal": (0, 165, 255)},     # Green / Orange
    "standing": {"normal": (255, 0, 0), "abnormal": (0, 0, 255)},      # Blue / Red
    "running": {"normal": (255, 255, 0), "abnormal": (0, 0, 255)},     # Cyan / Red
    "fighting": {"normal": (128, 0, 128), "abnormal": (0, 0, 255)},    # Purple / Red
    "jumping": {"normal": (0, 255, 255), "abnormal": (0, 0, 255)},     # Yellow / Red
    "robbery": {"normal": (255, 165, 0), "abnormal": (0, 0, 255)},     # Orange / Red
    "sitting": {"normal": (0, 128, 255), "abnormal": (0, 0, 255)},     # Light Orange / Red
    "armed": {"normal": (128, 128, 128), "abnormal": (0, 0, 255)},     # Gray / Red
    "lying_down": {"normal": (0, 255, 128), "abnormal": (0, 0, 255)}   # Light Green / Red
}

# Initialize tracking variables
motion_history = deque(maxlen=30)
standing_duration = {}
abnormal_threshold = 10  # Seconds threshold for standing
gathering_threshold = 3  # Number of people threshold for gathering
proximity_threshold = 100  # Pixels threshold for proximity between people

# Background subtraction parameters
motion_threshold = 2000  # Threshold for motion detection
crowd_density_threshold = 0.3  # 30% threshold for high density
activity_thresholds = {
    "walking": 0.25,   # walking becomes abnormal at 25% density
    "standing": 0.2,   # standing becomes abnormal at 20% density
    "running": 0.15,   # running becomes abnormal at 15% density
    "fighting": 0.1,   # fighting becomes abnormal at 10% density
    "jumping": 0.2,    # jumping becomes abnormal at 20% density
    "robbery": 0.05,   # robbery becomes abnormal at 5% density
    "sitting": 0.3,    # sitting becomes abnormal at 30% density
    "armed": 0.05,     # armed becomes abnormal at 5% density
    "lying_down": 0.2  # lying down becomes abnormal at 20% density
}

# Initialize background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Load the video
cap = cv2.VideoCapture("5783005-hd_1920_1080_30fps.mp4")
count = 0
prev_boxes = []
start_time = time.time()

# Remove the continuous alarm related code and simplify the alert system
ALERT_SETTINGS = {
    "running": {"frequency": 2500, "duration": 1000},  # 1 second beep for abnormal running
    "fighting": {"frequency": 3000, "duration": 1500}, # 1.5 second beep for fighting
    "robbery": {"frequency": 3500, "duration": 2000},  # 2 second beep for robbery
    "armed": {"frequency": 4000, "duration": 2000}     # 2 second beep for armed detection
}

# Remove continuous alarm variables and functions
last_alert_time = 0
ALERT_COOLDOWN = 2  # Minimum seconds between alerts

def trigger_alert(activity_type):
    """Play single alert sound and log for abnormal activities"""
    global last_alert_time
    current_time = time.time()
    
    # Only trigger for specific activities with cooldown
    if (activity_type in ALERT_SETTINGS and 
        current_time - last_alert_time >= ALERT_COOLDOWN):
        
        # Play single beep
        alert = ALERT_SETTINGS.get(activity_type)
        winsound.Beep(alert["frequency"], alert["duration"])
        last_alert_time = current_time
        
        # Log the alert
        with open("abnormal_activity_log.txt", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - ALERT: Abnormal {activity_type} detected\n")

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point2[1] - point2[1])**2)

def analyze_crowd_behavior(frame):
    # Apply background subtraction
    fgMask = backSub.apply(frame)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    
    # Calculate motion intensity
    motion_pixels = cv2.countNonZero(fgMask)
    total_pixels = frame.shape[0] * frame.shape[1]
    crowd_density = motion_pixels / total_pixels
    
    # Analyze crowd behavior
    is_crowded = crowd_density > crowd_density_threshold
    
    return is_crowded, fgMask, crowd_density

def is_activity_abnormal(activity, crowd_density, sudden_movement):
    """Determine if an activity is abnormal based on crowd density or always abnormal"""
    # Always abnormal activities
    always_abnormal_activities = {"fighting", "robbery", "armed", "lying_down"}
    if activity in always_abnormal_activities:
        return True
    
     # Standing and walking are only abnormal if there is sudden movement
    if activity in {"standing", "walking"}:
        return sudden_movement
    
    # Density-dependent abnormal activities
    threshold = activity_thresholds.get(activity, 0.3)
    return crowd_density > threshold

def check_gathering(current_boxes):
    centers = []
    for box in current_boxes:
        x1, y1, x2, y2, _ = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        centers.append(center)
    
    gathering_groups = []
    checked = set()
    
    for i, center1 in enumerate(centers):
        if i in checked:
            continue
            
        group = [i]
        for j, center2 in enumerate(centers):
            if i != j and j not in checked:
                if calculate_distance(center1, center2) < proximity_threshold:
                    group.append(j)
                    
        if len(group) >= gathering_threshold:
            gathering_groups.append(group)
            checked.update(group)
    
    return gathering_groups

def detect_sudden_movement(prev_boxes, current_boxes):
    """Detect sudden movement by comparing previous and current bounding boxes"""
    for prev_box, curr_box in zip(prev_boxes, current_boxes):
        prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
        curr_center = ((curr_box[0] + curr_box[2]) // 2, (curr_box[1] + curr_box[3]) // 2)
        distance = calculate_distance(prev_center, curr_center)
        if distance > proximity_threshold:  # Threshold for sudden movement
            return True
    return False

# Initialize variables for heatmap and tracking
heatmap = None
trajectories = defaultdict(list)  # To store trajectories of people

# Create a folder for saving abnormal frames
abnormal_frames_folder = "abnormal_frames"
if not os.path.exists(abnormal_frames_folder):
    os.makedirs(abnormal_frames_folder)

# Create a folder for saving heatmaps
heatmap_folder = "heatmaps"
if not os.path.exists(heatmap_folder):
    os.makedirs(heatmap_folder)

# Function to save abnormal frames within the folder
def save_abnormal_frame(frame, activity):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(abnormal_frames_folder, f"abnormal_{activity}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)

# Function to generate and save heatmap
def save_heatmap(heatmap, timestamp=None):
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(heatmap_folder, f"heatmap_{timestamp}.jpg")
    cv2.imwrite(filename, heatmap_colored)

# Function to update heatmap
def update_heatmap(heatmap, fg_mask):
    if heatmap is None:
        heatmap = np.zeros_like(fg_mask, dtype=np.float32)
    heatmap += fg_mask.astype(np.float32)
    return heatmap

# Function to track people using optical flow (without drawing trajectories)
def track_people(prev_gray, curr_gray, prev_boxes, trajectories):
    if prev_gray is None or len(prev_boxes) == 0:
        return trajectories

    # Calculate optical flow
    prev_points = np.array([((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) for box in prev_boxes], dtype=np.float32)
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)

    # Update trajectories
    for i, (new, old) in enumerate(zip(curr_points, prev_points)):
        if status[i]:
            trajectories[i].append((int(new[0]), int(new[1])))

    return trajectories

# Main loop
prev_gray = None
last_heatmap_time = time.time()  # Track the last time a heatmap was saved
heatmap_interval = 30  # Interval in seconds to save heatmaps

while True:
    ret, img = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    img = cv2.resize(img, (1020, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Analyze crowd behavior
    is_crowded, fg_mask, crowd_density = analyze_crowd_behavior(img)
    heatmap = update_heatmap(heatmap, fg_mask)

    # Save heatmap every 30 seconds
    current_time = time.time()
    if current_time - last_heatmap_time >= heatmap_interval:
        save_heatmap(heatmap)
        last_heatmap_time = current_time

    results = model.predict(img)
    current_boxes = []
    abnormal_activities = []
    sudden_movement = detect_sudden_movement(prev_boxes, current_boxes)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls)
            class_label = class_names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            box_id = f"{box_center[0]}_{box_center[1]}"

            # Track duration for standing activity
            if class_label == "standing":
                if box_id not in standing_duration:
                    standing_duration[box_id] = current_time
                activity_duration = current_time - standing_duration[box_id]
            else:
                activity_duration = 0
                standing_duration.pop(box_id, None)

            # Determine if activity is abnormal
            is_abnormal = is_activity_abnormal(class_label, crowd_density, sudden_movement)
            color = colors[class_label]["abnormal" if is_abnormal else "normal"]

            current_boxes.append((x1, y1, x2, y2, class_label))

            # Update display and alerts
            status = "ABNORMAL" if is_abnormal else "Normal"
            display_label = f"{class_label} ({status})"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, display_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_abnormal:
                abnormal_activities.append(class_label)
                save_abnormal_frame(img, class_label)  # Save abnormal frame

                # Trigger alerts for specific activities
                if class_label in ALERT_SETTINGS:
                    trigger_alert(class_label)

    # Track people using optical flow (no trajectory drawing)
    trajectories = track_people(prev_gray, gray, prev_boxes, trajectories)

    prev_boxes = current_boxes.copy()
    prev_gray = gray

    # Display density and number of humans in the corner
    num_humans = len(current_boxes)
    density_text = f"Density: {crowd_density:.2f}"
    human_count_text = f"Humans: {num_humans}"
    cv2.putText(img, density_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, human_count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display "ABNORMAL" or "NORMAL" based on alarm status
    overall_status = "ABNORMAL" if abnormal_activities else "NORMAL"
    status_color = (0, 0, 255) if overall_status == "ABNORMAL" else (0, 255, 0)
    cv2.putText(img, overall_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show both the original frame and the foreground mask
    cv2.imshow('Detection', img)
    # cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the final heatmap after processing
save_heatmap(heatmap)

cap.release()
cv2.destroyAllWindows()