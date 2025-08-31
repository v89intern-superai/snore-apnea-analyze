import io, os, math, tempfile
import numpy as np
import soundfile as sf
import pyedflib
from scipy.signal import resample_poly
from .config import TARGET_SR, CHANNEL_CANDIDATES

def _resample_to_16k(x, sr):
    if sr == TARGET_SR:
        return np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x).astype(np.float32)
    g = math.gcd(sr, TARGET_SR)
    up, down = TARGET_SR // g, sr // g
    return resample_poly(x, up, down).astype(np.float32)

def load_wav_from_bytes(file_bytes: bytes):
    x, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
    if x.ndim == 2: x = x.mean(axis=1)
    return _resample_to_16k(x, sr), TARGET_SR

def load_npy_from_bytes(file_bytes: bytes):
    buf = io.BytesIO(file_bytes)
    x = np.load(buf, allow_pickle=False).astype(np.float32)
    return x, TARGET_SR

def _pick_mic_channel(reader: pyedflib.EdfReader):
    labels = [str(l).lower() for l in reader.getSignalLabels()]
    for idx, label in enumerate(labels):
        if any(c in label for c in CHANNEL_CANDIDATES):
            return idx, reader.getSignalLabels()[idx]
    return None, None

def load_edf_from_bytes(file_bytes: bytes):
    # เขียน temp แล้วอ่าน (หลบ memory spike)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    try:
        with pyedflib.EdfReader(tmp_path) as r:
            ch_idx, ch_name = _pick_mic_channel(r)
            if ch_idx is None:
                raise RuntimeError(f"No microphone-like channel found. Available: {r.getSignalLabels()}")
            sig = r.readSignal(ch_idx).astype(np.float32)
            sr = int(r.getSampleFrequency(ch_idx))
        x = _resample_to_16k(sig, sr)
        meta = {"channel": ch_name, "orig_sr": sr, "duration_s": len(sig)/sr}
        return x, TARGET_SR, meta
    finally:
        try: os.remove(tmp_path)
        except: pass
