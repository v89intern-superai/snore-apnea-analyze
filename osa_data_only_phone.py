import os
import time
import librosa
import numpy as np
from glob import glob
from typing import List, Tuple
import json


class OSAData:
    def __init__(self, data_path: str):
        # Find all *_phone.wav files directly in data_path
        self.phone_files = sorted(glob(os.path.join(data_path, "*_phone.wav")))
        self.ann_files = sorted(glob(os.path.join(data_path, "*_annotation.json")))
        print("Found phone files:", self.phone_files)
        print("Found annotation files:", self.ann_files)
        self.patient_audio: List[Tuple[np.ndarray, int]] = []
        self.patient_annotations: List[dict] = []

    @staticmethod
    def extract_event_annotations(file_path: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return {}

    def load_data(self):
        """Load all *_phone.wav and *_annotation.json pairs in the data folder."""
        total = len(self.phone_files)
        for idx, phone_file in enumerate(self.phone_files, start=1):
            # Match annotation file by prefix
            prefix = os.path.basename(phone_file).split('_')[0]
            ann_file = os.path.join(os.path.dirname(phone_file), f"{prefix}_annotation.json")
            print(f"[{idx}/{total}] Processing {phone_file} + {ann_file}")
            if not os.path.isfile(ann_file):
                print(f"Warning: Annotation file not found for {phone_file}")
                continue
            try:
                wav_data, sr = librosa.load(phone_file, sr=None, mono=True)
                annotations = self.extract_event_annotations(ann_file)
                self.patient_audio.append((wav_data, sr))
                self.patient_annotations.append(annotations)
            except Exception as e:
                print(f"Warning: {e}")
        print("Data loading complete. Total patients:", len(self.patient_audio))

    # คงฟังก์ชันอื่นๆ ไว้ตามเดิม ถ้าเรียกใช้จะใช้งานได้
    def generate_boxplots(self, output_path: str):
        print("generate_boxplots: No signal data loaded; skipped.")

    def vis_data(self):
        print("vis_data: No signal data loaded; skipped.")

    def vis_single_patient(self, save_dir: str):
        print("vis_single_patient: No signal data loaded; skipped.")


if __name__ == "__main__":
    # ระบุ path ของ dataset บน Windows (escaped backslash)
    data_path = r"C:\V89\data"
    start_time = time.time()

    processor = OSAData(data_path)
    processor.load_data()

    elapsed = time.time() - start_time
    print(f"Loaded {len(processor.patient_audio)} patients in {elapsed:.2f} sec")
