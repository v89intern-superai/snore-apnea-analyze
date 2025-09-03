import os
from dataclasses import dataclass

# ===== Path โมเดล (แก้ทีเดียว) =====
CKPT_DIR = os.path.join("models", "wav2vec2_best")  # หรือใส่ path เต็ม r"D:\V89\...\wav2vec2_best"

# ===== Audio params =====
TARGET_SR = 16000
CHANNEL_CANDIDATES = ["mic", "audio", "snore", "tracheal", "mic1", "microphone"]

# ===== ค่าเริ่มต้น inference =====
@dataclass
class InferenceDefaults:
    window_s: float = 5.0
    stride_s: float = 1.0
    batch_size: int = 64
    enter: float = 0.6
    exit: float = 0.4
    min_consec: int = 2
    min_ep_sec: float = 10.0
    merge_gap_sec: float = 3.0

    # --- Snore Gate ---
    snore_high: float = 0.3   # ถ้า >= ส่งเข้า OSA (ลดลงจาก 0.4)
    snore_low: float = 0.15    # ถ้า < post-filter demote เป็น None (ลดลงจาก 0.2)


INF_DEF = InferenceDefaults()
