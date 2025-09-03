import numpy as np
import torch
from typing import Tuple
from .config import TARGET_SR, INF_DEF
from .snore_gate import snore_prob

def _predict_batch(wavs, processor, model, device):
    """Run OSA classifier (wav2vec2) on batch of wav segments."""
    with torch.inference_mode():
        inputs = processor(wavs, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        if device.type == "cuda":
            for k in inputs:
                inputs[k] = inputs[k].to(device).half()
        else:
            for k in inputs:
                inputs[k] = inputs[k].to(device)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # OSA prob
        return probs.detach().cpu().numpy()

def sliding_infer_streaming(
    x: np.ndarray, processor, model, device,
    window_s: float, stride_s: float, batch_size: int,
    progress_cb=None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Return:
      times: np.ndarray (window start times)
      p_osa: np.ndarray (raw OSA probs from wav2vec2, 0 if skipped)
      labels: list[str] ("None", "Snore", "OSA")
    """
    N = len(x)
    win = int(window_s * TARGET_SR)
    hop = int(stride_s * TARGET_SR)
    total = (N + hop - 1) // hop

    times, p_all, labels = [], [], []

    i, made = 0, 0
    while True:
        wavs, t_batch, segs = [], [], []
        for _ in range(batch_size):
            t = i * hop
            if t >= N:
                break
            seg = x[t:t + win]
            if len(seg) < win:
                pad = np.zeros(win, dtype=np.float32)
                pad[:len(seg)] = seg
                seg = pad
            seg = seg.astype(np.float32)
            wavs.append(seg)
            segs.append(seg)
            t_batch.append(t / TARGET_SR)
            i += 1
        if not wavs:
            break

        # Run snore gate on each seg
        for seg, t in zip(segs, t_batch):
            p_snore = snore_prob(seg)

            # Gate: ถ้า snore ต่ำกว่า threshold → None ทันที
            if p_snore < INF_DEF.snore_high:
                times.append(t)
                p_all.append(0.0)
                labels.append("None")
                continue

            # Otherwise → run OSA classifier
            osa_prob = float(_predict_batch([seg], processor, model, device)[0])

            # Post-filter: ถ้า OSA แต่ snore ต่ำกว่า low → None
            if osa_prob >= 0.5:
                if p_snore < INF_DEF.snore_low:
                    labels.append("None")
                else:
                    labels.append("OSA")
            else:
                labels.append("Snore")

            times.append(t)
            p_all.append(osa_prob)

        made += len(wavs)
        if progress_cb:
            progress_cb(min(1.0, made / max(1, total)))

    return np.asarray(times), np.asarray(p_all), labels
