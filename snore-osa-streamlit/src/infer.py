import numpy as np
import torch
from typing import Iterable, Tuple
from .config import TARGET_SR

def _predict_batch(wavs, processor, model, device):
    with torch.inference_mode():
        inputs = processor(wavs, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        if device.type == "cuda":
            for k in inputs: inputs[k] = inputs[k].to(device).half()
        else:
            for k in inputs: inputs[k] = inputs[k].to(device)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        return probs.detach().cpu().numpy()

def sliding_infer_streaming(x: np.ndarray, processor, model, device,
                            window_s: float, stride_s: float, batch_size: int,
                            progress_cb=None) -> Tuple[np.ndarray, np.ndarray]:
    N = len(x); win = int(window_s*TARGET_SR); hop = int(stride_s*TARGET_SR)
    # จำนวนหน้าต่างทั้งหมด (ประมาณค่าเพื่อ progress)
    total = (N + hop - 1) // hop
    times = []; p_all = []

    # stream เป็น batch: ทำหน้าต่างทีละ "batch_size" โดยไม่เก็บทั้งหมดไว้ก่อน
    i = 0; made = 0
    while True:
        wavs = []; t_batch = []
        for _ in range(batch_size):
            t = i*hop
            if t >= N: break
            seg = x[t:t+win]
            if len(seg) < win:
                pad = np.zeros(win, dtype=np.float32); pad[:len(seg)] = seg; seg = pad
            wavs.append(seg.astype(np.float32))
            t_batch.append(t / TARGET_SR)
            i += 1
        if not wavs: break

        probs = _predict_batch(wavs, processor, model, device)
        p_all.extend(probs.tolist()); times.extend(t_batch)

        made += len(wavs)
        if progress_cb: progress_cb(min(1.0, made/max(1,total)))

    return np.asarray(times), np.asarray(p_all)
