import io, os, math, json, time
import numpy as np
import streamlit as st
import librosa, soundfile as sf
from scipy.signal import resample_poly
import pyedflib

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# =========================
# Config
# =========================
CKPT_DIR = os.path.join("models", "wav2vec2_best")   # โฟลเดอร์โมเดลของคุณ
TARGET_SR = 16000                                     # ต้องตรงกับที่เทรน
DEFAULTS = dict(window_s=5.0, stride_s=1.0, enter=0.6, exit=0.4,
                min_consec=2, min_ep_sec=10.0, merge_gap_sec=3.0,
                prob_threshold=0.5, batch_size=64)

CHANNEL_CANDIDATES = ["mic", "audio", "snore", "tracheal", "mic1", "microphone"]

# =========================
# Caching loaders
# =========================
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_processor():
    processor = Wav2Vec2Processor.from_pretrained(CKPT_DIR)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(CKPT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    id2label = model.config.id2label or {0:"Snore", 1:"OSA"}
    return processor, model, device, id2label

# =========================
# Audio helpers
# =========================
def resample_to_16k(x, sr):
    """ใช้ resample_poly เพื่อคุณภาพและความเร็ว"""
    if sr == TARGET_SR:
        return x.astype(np.float32)
    # ป้องกัน NaN/Inf
    x = np.nan_to_num(x).astype(np.float32)
    g = math.gcd(sr, TARGET_SR)
    up, down = TARGET_SR // g, sr // g
    y = resample_poly(x, up, down).astype(np.float32)
    return y

def load_wav(file_bytes):
    data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
    if data.ndim == 2:  # stereo -> mono
        data = data.mean(axis=1)
    data = resample_to_16k(data, sr)
    return data, TARGET_SR

def load_npy(file_bytes):
    buf = io.BytesIO(file_bytes)
    x = np.load(buf, allow_pickle=False).astype(np.float32)
    return x, TARGET_SR

def pick_mic_channel(edf_reader):
    labels = [str(l).lower() for l in edf_reader.getSignalLabels()]
    for idx, label in enumerate(labels):
        for cand in CHANNEL_CANDIDATES:
            if cand in label:
                return idx, edf_reader.getSignalLabels()[idx]
    return None, None

def load_edf(file_bytes):
    """อ่าน EDF → เลือก channel เสียง → แปลงเป็น 16k mono np.ndarray"""
    tmp = "tmp_upload.edf"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    try:
        with pyedflib.EdfReader(tmp) as r:
            ch_idx, ch_name = pick_mic_channel(r)
            if ch_idx is None:
                raise RuntimeError(f"No microphone-like channel found. Available: {r.getSignalLabels()}")
            sig = r.readSignal(ch_idx).astype(np.float32)
            sr = int(r.getSampleFrequency(ch_idx))
        x = resample_to_16k(sig, sr)
        meta = {"channel": ch_name, "orig_sr": sr, "duration_s": len(sig)/sr}
    finally:
        try: os.remove(tmp)
        except: pass
    return x, TARGET_SR, meta

# =========================
# Inference helpers
# =========================
def batched_predict(prob_inputs, processor, model, device, batch_size=64):
    """prob_inputs: list[np.ndarray], return list[float] p_osa"""
    out = []
    for i in range(0, len(prob_inputs), batch_size):
        batch = prob_inputs[i:i+batch_size]
        inputs = processor(batch, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**{k:v.to(device) for k,v in inputs.items()}).logits
        probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy().tolist()  # index 1 = OSA
        out.extend(probs)
    return out

def sliding_infer(x, processor, model, device, window_s, stride_s, batch_size=64, progress_cb=None):
    N = len(x); win = int(window_s * TARGET_SR); hop = int(stride_s * TARGET_SR)
    windows, times = [], []
    t = 0
    # เตรียมหน้าต่างทั้งหมด (padding ช่วงท้าย)
    while t < N:
        seg = x[t:t+win]
        if len(seg) < win:
            pad = np.zeros(win, dtype=np.float32)
            pad[:len(seg)] = seg
            seg = pad
        windows.append(seg.astype(np.float32))
        times.append(t / TARGET_SR)
        t += hop
    # รัน batch
    p_osa = []
    total = len(windows)
    for i in range(0, total, batch_size):
        batch = windows[i:i+batch_size]
        probs = batched_predict(batch, processor, model, device, batch_size=batch_size)
        p_osa.extend(probs)
        if progress_cb is not None:
            progress_cb(min(1.0, (i+len(batch)) / total))
    return np.array(times), np.array(p_osa)

def hysteresis_series(p, enter=0.6, exit=0.4, min_consec=2):
    """แปลง prob เป็นเฟรม OSA/Non-OSA (0/1) ด้วย hysteresis + ข้อกำหนดจำนวนเฟรมติดกัน"""
    state = 0
    y = np.zeros_like(p, dtype=int)
    consec = 0
    for i, v in enumerate(p):
        if state == 0:  # outside
            if v >= enter:
                consec += 1
                if consec >= min_consec:
                    state = 1
                    consec = 0
        else:  # inside
            if v <= exit:
                consec += 1
                if consec >= min_consec:
                    state = 0
                    consec = 0
        y[i] = state
        if (state==0 and v < enter): consec = 0
        if (state==1 and v > exit): consec = 0
    return y

def events_from_series(times, mask, min_ep_sec=10.0, merge_gap_sec=3.0, stride_s=1.0):
    """รวมเฟรมที่เป็น OSA ติดกันเป็น episode"""
    events = []
    i = 0; n = len(mask)
    while i < n:
        if mask[i] == 1:
            start = times[i]
            j = i+1
            while j < n and mask[j] == 1:
                j += 1
            end = times[j-1] + stride_s  # สิ้นสุดตอน
            dur = end - start
            if dur >= min_ep_sec:
                # merge กับ event ก่อนหน้าถ้าห่างน้อย
                if events and start - events[-1]["end_s"] <= merge_gap_sec:
                    events[-1]["end_s"] = end
                    events[-1]["duration_s"] = events[-1]["end_s"] - events[-1]["start_s"]
                else:
                    events.append({"start_s": start, "end_s": end, "duration_s": dur})
            i = j
        else:
            i += 1
    return events

# =========================
# UI
# =========================
st.set_page_config(page_title="Snore vs OSA (Wav2Vec2)", layout="wide")
st.title("Snore vs OSA Detector (Wav2Vec2)")

with st.sidebar:
    st.subheader("Model & Inference Settings")
    window_s = st.number_input("Window (sec)", min_value=2.0, max_value=15.0, value=DEFAULTS["window_s"], step=0.5)
    stride_s = st.number_input("Stride (sec)", min_value=0.25, max_value=5.0, value=DEFAULTS["stride_s"], step=0.25)
    batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=DEFAULTS["batch_size"], step=8)
    st.markdown("---")
    st.subheader("Smoothing (Hysteresis)")
    enter = st.slider("Enter threshold", 0.1, 0.9, DEFAULTS["enter"], 0.05)
    exit_ = st.slider("Exit threshold", 0.1, 0.9, DEFAULTS["exit"], 0.05)
    min_consec = st.number_input("Min consecutive frames", min_value=1, max_value=10, value=DEFAULTS["min_consec"])
    st.markdown("---")
    st.subheader("Eventization")
    min_ep_sec = st.number_input("Min episode length (sec)", min_value=3.0, max_value=60.0, value=DEFAULTS["min_ep_sec"], step=1.0)
    merge_gap_sec = st.number_input("Merge gap (sec)", min_value=0.0, max_value=10.0, value=DEFAULTS["merge_gap_sec"], step=0.5)

processor, model, device, id2label = load_model_and_processor()
st.success(f"Model loaded on {device}. Labels: {id2label}")

uploaded = st.file_uploader("Upload sleep audio file (.edf / .wav / .npy)", type=["edf","wav","npy"])
if uploaded:
    st.write(f"**File:** {uploaded.name} | Size: {uploaded.size/1e6:.2f} MB")

    # Preview audio if WAV
    if uploaded.type in ("audio/wav", "audio/x-wav"):
        st.audio(uploaded)

    # Load audio
    with st.spinner("Reading & resampling audio..."):
        if uploaded.name.lower().endswith(".edf"):
            x, sr, meta = load_edf(uploaded.read())
            st.info(f"EDF channel used: {meta['channel']} | orig_sr={meta['orig_sr']} | duration={meta['duration_s']:.1f}s")
        elif uploaded.name.lower().endswith(".wav"):
            x, sr = load_wav(uploaded.read())
        else:  # npy
            x, sr = load_npy(uploaded.read())
        duration_h = len(x)/sr/3600
        st.write(f"Signal length: **{len(x)/sr:.1f} s** (~{duration_h:.2f} h) @ {sr} Hz")

    # Analyze button
    if st.button("Analyze"):
        prog = st.progress(0.0, text="Running sliding window inference...")
        t0 = time.time()
        times, p_osa = sliding_infer(
            x, processor, model, device,
            window_s=window_s, stride_s=stride_s, batch_size=int(batch_size),
            progress_cb=lambda p: prog.progress(p, text=f"Running... {int(100*p)}%"),
        )
        prog.progress(1.0, text="Done.")
        st.success(f"Inference completed in {time.time()-t0:.1f} s. Windows: {len(times)}")

        # Hysteresis → episodes
        mask = hysteresis_series(p_osa, enter=enter, exit=exit_, min_consec=int(min_consec))
        events = events_from_series(times, mask, min_ep_sec=min_ep_sec, merge_gap_sec=merge_gap_sec, stride_s=stride_s)

        # Segment-level counts (threshold 0.5)
        seg_pred = (p_osa >= 0.5).astype(int)
        snore_segments = int((seg_pred==0).sum())
        osa_segments   = int((seg_pred==1).sum())
        osa_ratio = osa_segments / max(1, (snore_segments + osa_segments))

        # Summary text
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Snore segments", snore_segments)
        col2.metric("OSA segments", osa_segments)
        col3.metric("OSA ratio", f"{osa_ratio*100:.1f}%")

        # Timeline prob chart
        st.line_chart({"p_OSA": p_osa}, height=220)

        # Events table
        st.subheader("Detected OSA Episodes")
        if events:
            import pandas as pd
            ev_df = pd.DataFrame(events)
            # เพิ่ม mean_prob / max_prob สำหรับแต่ละตอน
            mean_probs, max_probs = [], []
            for _, row in ev_df.iterrows():
                idx = (times >= row.start_s) & (times <= row.end_s)
                mean_probs.append(float(p_osa[idx].mean()) if idx.any() else 0.0)
                max_probs.append(float(p_osa[idx].max()) if idx.any() else 0.0)
            ev_df["mean_prob"] = mean_probs
            ev_df["max_prob"] = max_probs
            st.dataframe(ev_df, use_container_width=True)

            # Download CSV
            csv_bytes = ev_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download events CSV", data=csv_bytes, file_name="osa_events.csv", mime="text/csv")
        else:
            st.info("No OSA episode detected with current settings.")

        st.caption("Note: This tool is for research/demo only, not a medical diagnosis.")
else:
    st.info("Upload a .edf / .wav / .npy file to start.")
