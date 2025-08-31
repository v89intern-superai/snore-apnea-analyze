import os, time, streamlit as st, numpy as np

# Note: Upload size limits should be set via command line or config.toml
# st.set_option("server.maxUploadSize", 2048)  # Cannot be set on the fly
# st.set_option("server.maxMessageSize", 2048)  # Cannot be set on the fly

from src.config import INF_DEF, TARGET_SR
from src.model_loader import load_model_and_processor
from src.audio_io import load_edf_from_bytes, load_wav_from_bytes, load_npy_from_bytes
from src.infer import sliding_infer_streaming
from src.postprocess import hysteresis_series, events_from_series
from src.ui import sidebar, show_summary, show_events_table

st.set_page_config(page_title="Snore vs OSA (Wav2Vec2)", layout="wide")
st.title("Snore vs OSA Detector (Wav2Vec2)")

# Sidebar (UI-only)
window_s, stride_s, batch_size, enter, exit_, min_consec, min_ep_sec, merge_gap_sec = sidebar(INF_DEF)

@st.cache_resource(show_spinner="Loading model...")
def _cached():
    return load_model_and_processor()

processor, model, device, id2label = _cached()
st.success(f"Model loaded on {device}. Labels: {id2label}")

# อัปโหลดไฟล์ (หรือเลือกพาธด้านล่าง)
uploaded = st.file_uploader("Upload .edf / .wav / .npy (large files allowed)", type=["edf","wav","npy"])
st.markdown("**Or analyze a local path (no browser upload):**")
local_path = st.text_input("Local path", "")

def _load_from_input():
    if uploaded is not None:
        data = uploaded.read()
        name = uploaded.name.lower()
        if name.endswith(".edf"):
            x, sr, meta = load_edf_from_bytes(data)
            st.info(f"EDF channel: {meta['channel']} | orig_sr={meta['orig_sr']} | dur={meta['duration_s']:.1f}s")
        elif name.endswith(".wav"):
            x, sr = load_wav_from_bytes(data)
        else:
            x, sr = load_npy_from_bytes(data)
        return x, sr
    elif local_path:
        ext = os.path.splitext(local_path)[1].lower()
        with open(local_path, "rb") as f: data = f.read()
        if ext == ".edf":
            x, sr, meta = load_edf_from_bytes(data)
            st.info(f"EDF channel: {meta['channel']} | orig_sr={meta['orig_sr']} | dur={meta['duration_s']:.1f}s")
        elif ext == ".wav":
            x, sr = load_wav_from_bytes(data)
        elif ext == ".npy":
            x, sr = load_npy_from_bytes(data)
        else:
            st.error("Unsupported extension"); return None, None
        return x, sr
    else:
        return None, None

x, sr = _load_from_input()
if x is None:
    st.info("Upload a file or provide a local path to start.")
    st.stop()

st.write(f"Signal length: **{len(x)/sr:.1f} s** @ {sr} Hz")

if st.button("Analyze"):
    prog = st.progress(0.0, text="Running inference...")
    t0 = time.time()
    times, p_osa = sliding_infer_streaming(
        x, processor, model, device,
        window_s=float(window_s), stride_s=float(stride_s), batch_size=int(batch_size),
        progress_cb=lambda p: prog.progress(p, text=f"Running... {int(100*p)}%")
    )
    prog.progress(1.0, text="Done.")
    st.success(f"Inference finished in {time.time()-t0:.1f}s | windows={len(times)}")

    # รายงานนับ segment ที่ threshold 0.5 (สำหรับ count view)
    seg_pred = (p_osa >= 0.5).astype(int)
    show_summary(int((seg_pred==0).sum()), int((seg_pred==1).sum()))

    st.line_chart({"p_OSA": p_osa}, height=220)

    mask = hysteresis_series(p_osa, enter=float(enter), exit=float(exit_), min_consec=int(min_consec))
    events = events_from_series(times, mask, min_ep_sec=float(min_ep_sec),
                                merge_gap_sec=float(merge_gap_sec), stride_s=float(stride_s))
    show_events_table(times, p_osa, events)

st.caption("Research/PoC only. Not a medical device.")
