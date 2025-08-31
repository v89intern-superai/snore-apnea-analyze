import pandas as pd
import streamlit as st

def sidebar(INF_DEF):
    with st.sidebar:
        st.subheader("Inference Settings")
        window_s = st.number_input("Window (sec)", 2.0, 15.0, INF_DEF.window_s, 0.5)
        stride_s = st.number_input("Stride (sec)", 0.25, 5.0, INF_DEF.stride_s, 0.25)
        batch_size = st.number_input("Batch size", 8, 256, INF_DEF.batch_size, 8)
        st.markdown("---")
        st.subheader("Smoothing (Hysteresis)")
        enter = st.slider("Enter threshold", 0.1, 0.9, INF_DEF.enter, 0.05)
        exit_ = st.slider("Exit threshold", 0.1, 0.9, INF_DEF.exit, 0.05)
        min_consec = st.number_input("Min consecutive frames", 1, 10, INF_DEF.min_consec)
        st.markdown("---")
        st.subheader("Eventization")
        min_ep_sec = st.number_input("Min episode length (sec)", 3.0, 60.0, INF_DEF.min_ep_sec, 1.0)
        merge_gap_sec = st.number_input("Merge gap (sec)", 0.0, 10.0, INF_DEF.merge_gap_sec, 0.5)
    return window_s, stride_s, batch_size, enter, exit_, min_consec, min_ep_sec, merge_gap_sec

def show_summary(snore_segments, osa_segments):
    total = max(1, snore_segments + osa_segments)
    ratio = osa_segments/total
    c1, c2, c3 = st.columns(3)
    c1.metric("Snore segments", snore_segments)
    c2.metric("OSA segments", osa_segments)
    c3.metric("OSA ratio", f"{ratio*100:.1f}%")

def show_events_table(times, p_osa, events):
    st.subheader("Detected OSA Episodes")
    if not events:
        st.info("No OSA episode detected with current settings.")
        return
    rows = []
    for ev in events:
        m = (times >= ev["start_s"]) & (times <= ev["end_s"])
        mean_prob = float(p_osa[m].mean()) if m.any() else 0.0
        max_prob  = float(p_osa[m].max()) if m.any() else 0.0
        rows.append({**ev, "mean_prob": mean_prob, "max_prob": max_prob})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download events CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="osa_events.csv", mime="text/csv")
