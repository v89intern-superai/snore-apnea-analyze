import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import List, Tuple
import pandas as pd
import random
import json


class OSAData:
    def __init__(self, data_path: str):
        # เก็บ paths ของโฟลเดอร์ผู้ป่วยทั้งหมด
        self.patient_dirs = [p for p in glob(os.path.join(data_path, "*")) if os.path.isdir(p)]
        # เก็บผลลัพธ์สำหรับ audio และ annotation
        self.patient_audio: List[Tuple[np.ndarray, int]] = []  # (waveform, sample_rate)
        self.patient_annotations: List[dict] = []

    @staticmethod
    def str2seconds(time_str: str) -> float:
        parts = time_str.split(":")
        h, m = map(int, parts[:2])
        s_part = parts[2]
        if '.' in s_part:
            s, ms = s_part.split('.')
            s = int(s)
            ms = float(f"0.{ms}")
        else:
            s = int(s_part)
            ms = 0.0
        if h < 12:
            h += 24
        return h * 3600 + m * 60 + s + ms

    @staticmethod
    def is_in_problem_interval(x: float, intervals: List[Tuple[float, float]]) -> bool:
        for interval in intervals:
            if interval[0] <= x <= interval[1]:
                return True
        return False

    @staticmethod
    def extract_event_annotations(file_path: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return {}

    # ฟังก์ชัน extract_spo2, extract_HR, extract_flow_dr, extract_sleep_stage 
    # ยังคงอยู่เพื่อใช้งานต่อ แต่ load_data จะไม่เรียกใช้งาน
    @staticmethod
    def extract_spo2(spo_path, r_start, awakes):
        # ... original implementation ...
        pass

    @staticmethod
    def extract_HR(HR_path, r_start):
        # ... original implementation ...
        pass

    @staticmethod
    def extract_flow_dr(flow_path, r_start):
        # ... original implementation ...
        pass

    @staticmethod
    def extract_sleep_stage(sleep_stage_path, r_start):
        # ... original implementation ...
        pass

    @staticmethod
    def load_single_data(folder: str) -> Tuple[Tuple[np.ndarray, int], dict]:
        """
        โหลดเฉพาะ audio จากมือถือ (*_phone.wav) และ annotation.json
        """
        phone_file = next(glob(os.path.join(folder, "*_phone.wav")), None)
        ann_file = next(glob(os.path.join(folder, "*annotation.json")), None)

        if not phone_file or not ann_file:
            raise FileNotFoundError(f"Missing phone or annotation in {folder}")

        # โหลดเสียงมือถือ (mono)
        wav_data, sr = librosa.load(phone_file, sr=None, mono=True)

        # โหลด annotation
        annotations = OSAData.extract_event_annotations(ann_file)

        return (wav_data, sr), annotations

    def load_data(self):
        """
        ลูปโหลดเฉพาะมือถือและ annotation เท่านั้น
        """
        total = len(self.patient_dirs)
        for idx, folder in enumerate(self.patient_dirs, start=1):
            print(f"[{idx}/{total}] Processing {folder}")
            try:
                audio, ann = self.load_single_data(folder)
                self.patient_audio.append(audio)
                self.patient_annotations.append(ann)
            except Exception as e:
                print(f"Warning: {e}")
        print("Data loading complete. Total patients:", len(self.patient_audio))

    # ฟังก์ชัน generate_boxplots, vis_data, vis_single_patient ยังคงอยู่
    def generate_boxplots(self, output_path: str):
        # เดิมมีการคำนวณจาก spo2, heart_rate, flow_dr
        print("generate_boxplots: Skipped because signals are not loaded.")

    def vis_data(self):
        print("vis_data: Skipped visualization as only phone and annotation are loaded.")

    def vis_single_patient(self, save_dir: str):
        print("vis_single_patient: Skipped visualization of other signals.")


if __name__ == "__main__":
    data_path = "/path/to/PSG-Audio"
    start = time.time()
    processor = OSAData(data_path)
    processor.load_data()
    elapsed = time.time() - start
    print(f"Loaded {len(processor.patient_audio)} patients in {elapsed:.2f} sec")
