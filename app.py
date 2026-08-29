"""
app.py
TrajGuard demo frontend.

Shows a video next to its risk-score graph, revealing the graph
progressively in sync with playback, and raises an alert when the
risk score crosses a threshold.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import time
import os

VIDEOS_DIR = "videos"
RISK_DIR = "data/risk_scores"
FPS = 30
THRESHOLD = 0.5

st.set_page_config(page_title="TrajGuard", layout="wide")
st.title("TrajGuard — Failure Prediction Demo")

# ---- Step 1: pick a video ----
available = [f.replace(".csv", "") for f in os.listdir(RISK_DIR) if f.endswith(".csv")] if os.path.exists(RISK_DIR) else []

if not available:
    st.warning(f"No risk score CSVs found in {RISK_DIR}. Run compute_risk.py first, or add a test CSV there.")
    st.stop()

video_id = st.selectbox("Choose a clip to analyze", available)

# ---- Step 2: load data ----
risk_df = pd.read_csv(os.path.join(RISK_DIR, f"{video_id}.csv"))
video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Video Feed")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info(f"No video file found at {video_path} — showing risk graph only.")

with col2:
    st.subheader("Risk Score Over Time")
    chart_placeholder = st.empty()

st.subheader("Status")
status_placeholder = st.empty()

# ---- Step 3: play button triggers the synced reveal ----
if st.button("▶ Run Analysis"):
    revealed_rows = []
    alerted = False

    for i, row in risk_df.iterrows():
        revealed_rows.append(row)
        partial_df = pd.DataFrame(revealed_rows)

        # update the graph with data revealed so far
        chart_placeholder.line_chart(partial_df.set_index("frame")["R_t"])

        # check threshold crossing
        if row["R_t"] >= THRESHOLD and not alerted:
            alerted = True
            status_placeholder.error(
                f"⚠️ Risk threshold crossed at frame {int(row['frame'])} "
                f"(R_t = {row['R_t']:.2f}). Failure likely imminent."
            )
        elif not alerted:
            status_placeholder.success(f"Normal — frame {int(row['frame'])}, R_t = {row['R_t']:.2f}")

        time.sleep(1 / FPS)  # paces the reveal to roughly match video playback speed

    if not alerted:
        status_placeholder.success("Clip completed — no risk threshold crossed.")
else:
    # show the full static graph before playback starts
    chart_placeholder.line_chart(risk_df.set_index("frame")["R_t"])
    status_placeholder.info("Press ▶ Run Analysis to replay the risk score in sync with the video.")

