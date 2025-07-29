# realtime_har_fixed.py

import cv2
import torch
import numpy as np
from collections import deque
from pytorchvideo.models.hub import x3d_s
import requests

# 1) ดาวน์โหลดชื่อกิจกรรม (Kinetics-400 label map)
LABELS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
resp = requests.get(LABELS_URL)
label_list = [line.strip() for line in resp.text.splitlines() if line.strip()]

# 2) โหลดโมเดล X3D-S (pretrained บน Kinetics-400)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = x3d_s(pretrained=True).to(device).eval()

# 3) ตั้งค่า Sliding Window
T, STRIDE = 16, 8            # T=จำนวนเฟรมต่อคลิป, STRIDE=รัน inference ทุกกี่เฟรม
buffer = deque(maxlen=T)
frame_idx = 0
label = "N/A"

# 4) เปิดเว็บแคม
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("ไม่สามารถเปิดเว็บแคมได้")

# 5) ปรับ mean/std สำหรับ normalize
mean = torch.tensor([0.45,0.45,0.45], device=device).view(1,3,1,1,1)
std  = torch.tensor([0.225,0.225,0.225], device=device).view(1,3,1,1,1)

# 6) ลูปอ่านเฟรม + inference
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # **แก้**: ย่อขนาดเป็น 224×224 เพื่อให้ pooling kernel 5×5 ผ่านได้
    small = cv2.resize(frame, (224,224))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    buffer.append(rgb)
    frame_idx += 1

    # เมื่อ buffer เต็ม และถึงรอบ STRIDE ให้ inference
    if len(buffer) == T and frame_idx % STRIDE == 0:
        clip = np.stack(buffer, axis=0)                 # [T,H,W,3]
        clip = torch.from_numpy(clip).permute(3,0,1,2)   # [3,T,224,224]
        clip = clip.unsqueeze(0).float() / 255.0         # [1,3,T,224,224]
        clip = (clip.to(device) - mean) / std            # normalize

        # **แก้**: model คืน Tensor โดยตรง ไม่มี .logits
        with torch.no_grad():
            out = model(clip)                            # Tensor [1, num_classes]
        idx = out.argmax(-1).item()                      # top-1 class index
        label = label_list[idx]                          # lookup ชื่อกิจกรรม

    # วาดป้ายชื่อกิจกรรมบนภาพจริง
    cv2.putText(frame, f"Action: {label}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("Real-time HAR", frame)

    # กด Esc เพื่อออก
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
