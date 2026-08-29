"""
replay_view.py

The Validation Replay tab. Lets a judge pick a labeled clip (REFERENCE /
VALIDATION / HELD-OUT), replays its risk score synced to a playhead,
marks the labeled drop_frame, and shows the headline proof metrics:
lead time, false-alarm count, naive-vs-ours comparison, compute savings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

import data_loader
from filter_logic import DeviationFilter, naive_score_full_trajectory, DEFAULT_THRESHOLD
import reka_client


def render():
    st.subheader("Validation Replay")

    labels = data_loader.load_labels()
    if not labels:
        st.warning("No labels found.")
        return

    # Group by split so judges see the honest train/test separation
    split_options = sorted(set(row["split"] for row in labels))
    selected_split = st.selectbox("Filter by split", ["ALL"] + split_options)

    filtered = labels if selected_split == "ALL" else [r for r in labels if r["split"] == selected_split]
    video_ids = [r["video_id"] for r in filtered]

    if not video_ids:
        st.info("No clips in this split.")
        return

    video_id = st.selectbox("Choose a clip", video_ids)
    row = next(r for r in filtered if r["video_id"] == video_id)

    st.caption(f"Type: **{row['type'].upper()}**  |  Split: **{row['split']}**")

    # Load reference envelope + trajectory (with fallback flags so we can warn if synthetic)
    ref_mean, ref_std, ref_is_real = data_loader.load_reference_envelope()
    trajectory, traj_is_real = data_loader.load_trajectory(video_id, num_frames=len(ref_mean))

    if not ref_is_real:
        st.info("Using a synthetic reference envelope — real reference_runs/ data not found yet.")
    if not traj_is_real:
        st.info(f"Using synthetic data for '{video_id}' — real trajectory not found yet.")

    # Score using the real filter, and using the naive baseline for comparison
    scorer = DeviationFilter(ref_mean, ref_std)
    results = scorer.score_full_trajectory(trajectory)
    df = pd.DataFrame(results)

    naive_results = naive_score_full_trajectory(trajectory, ref_mean, ref_std, threshold=DEFAULT_THRESHOLD)
    naive_df = pd.DataFrame(naive_results)

    # --- Video + chart side by side ---
    col1, col2 = st.columns(2)

    with col1:
        video_path = f"videos/{video_id}.mp4"
        try:
            st.video(video_path)
        except Exception:
            st.info("No video file available for this clip — showing risk graph only.")

    with col2:
        chart_df = df.set_index("frame")[["R_t"]]
        st.line_chart(chart_df)

        drop_frame = row["drop_frame"]
        if drop_frame:
            st.caption(f"Labeled drop frame: **{drop_frame}**")

    # --- Headline metrics ---
    st.markdown("### Proof Metrics")
    m1, m2, m3, m4 = st.columns(4)

    # Lead time (only meaningful for failure clips with a drop_frame)
    with m1:
        if row["type"] == "failure" and drop_frame:
            drop_frame_int = int(drop_frame)
            flagged = df[df["status"] == "RISK CONFIRMED"]
            if not flagged.empty:
                first_flag = int(flagged.iloc[0]["frame"])
                fps = data_loader.load_fps(video_id)
                lead_frames = drop_frame_int - first_flag
                lead_seconds = lead_frames / fps
                st.metric("Lead Time", f"{lead_seconds:.2f}s", f"{lead_frames} frames early")
            else:
                st.metric("Lead Time", "Not flagged", "threshold never crossed")
        else:
            st.metric("Lead Time", "N/A", "not a failure clip")

    # False alarm count (meaningful for normal/borderline clips)
    with m2:
        if row["type"] in ("normal", "borderline"):
            alarms = (df["status"] == "RISK CONFIRMED").sum()
            st.metric("False Alarms", int(alarms))
        else:
            st.metric("False Alarms", "N/A")

    # Naive vs ours comparison
    with m3:
        naive_flags = int(naive_df["flagged"].sum())
        our_flags = int((df["status"] == "RISK CONFIRMED").sum())
        st.metric("Naive Alerts", naive_flags, f"Ours: {our_flags}")

    # Compute savings
    with m4:
        total_frames = len(df)
        triggered_frames = int((df["status"] == "RISK CONFIRMED").sum())
        pct = 100 * triggered_frames / total_frames if total_frames else 0
        st.metric("Layer 2 Calls", f"{pct:.1f}%", "of total frames")

    # --- Layer 2 explanation, only if risk was confirmed ---
    risky_rows = df[df["status"] == "RISK CONFIRMED"]
    if not risky_rows.empty:
        st.markdown("### Layer 2 Explanation")
        trigger_row = risky_rows.iloc[0].to_dict()
        with st.spinner("Calling explainer..."):
            result = reka_client.explain_risk(trigger_row)
        st.markdown(
            f"""<div class="tg-card">
            <div class="tg-metric-label">EXPLANATION (source: {result['source']})</div>
            <div style="margin-top:8px;">{result['explanation']}</div>
            <div class="tg-metric-label" style="margin-top:12px;">LATENCY</div>
            <div class="tg-metric-value">{result['latency_ms']:.0f} ms</div>
            </div>""",
            unsafe_allow_html=True,
        )