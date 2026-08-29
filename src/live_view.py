"""
live_view.py

The Live Demo tab. Judges press "Start Monitoring", perform a motion in
front of the webcam, and watch the risk score chart + status bar react
in real time.

Note: Streamlit doesn't support true continuous video natively. This
achieves a near-real-time effect by looping and re-rendering an image
placeholder + chart placeholder on every captured frame, while a
"Stop" flag in session_state controls the loop.
"""

import streamlit as st
import pandas as pd

from live_engine import LiveEngine
import reka_client


def render():
    st.subheader("Live Demo")
    st.caption("Press Start, then try a sudden or unstable motion in front of the camera.")

    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "live_history" not in st.session_state:
        st.session_state.live_history = []
    if "live_explained" not in st.session_state:
        st.session_state.live_explained = False

    col_btn1, col_btn2 = st.columns(2)
    start_clicked = col_btn1.button("▶ Start Monitoring")
    stop_clicked = col_btn2.button("■ Stop")

    if start_clicked:
        st.session_state.live_running = True
        st.session_state.live_history = []
        st.session_state.live_explained = False

    if stop_clicked:
        st.session_state.live_running = False

    frame_placeholder = st.empty()
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    explanation_placeholder = st.empty()

    if not st.session_state.live_running:
        status_placeholder.info("Monitoring stopped. Press Start Monitoring to begin.")
        return

    engine = LiveEngine()
    if not engine.reference_is_real:
        st.info("Using a synthetic reference envelope — real reference_runs/ data not found yet.")

    engine.start()

    try:
        # Run a bounded number of frames per Streamlit rerun to avoid
        # locking up the browser tab indefinitely
        max_frames_per_run = 200

        for _ in range(max_frames_per_run):
            if not st.session_state.live_running:
                break

            frame, score = engine.read_and_score()
            if frame is None:
                status_placeholder.warning("Could not read from webcam.")
                break

            frame_placeholder.image(frame, channels="RGB")

            if score is not None:
                st.session_state.live_history.append(score)
                history_df = pd.DataFrame(st.session_state.live_history)
                chart_placeholder.line_chart(history_df.set_index("frame")[["R_t"]])

                if score["status"] == "RISK CONFIRMED":
                    status_placeholder.markdown(
                        '<div class="tg-handoff-bar risk">RISK CONFIRMED</div>',
                        unsafe_allow_html=True,
                    )
                    if not st.session_state.live_explained:
                        st.session_state.live_explained = True
                        with explanation_placeholder.container():
                            with st.spinner("Calling Layer 2 explainer..."):
                                result = reka_client.explain_risk(score)
                            st.markdown(
                                f"""<div class="tg-card">
                                <div class="tg-metric-label">EXPLANATION</div>
                                <div style="margin-top:8px;">{result['explanation']}</div>
                                <div class="tg-metric-label" style="margin-top:12px;">LATENCY</div>
                                <div class="tg-metric-value">{result['latency_ms']:.0f} ms</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                else:
                    status_placeholder.markdown(
                        '<div class="tg-handoff-bar nominal">NOMINAL</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.live_explained = False
            else:
                status_placeholder.markdown(
                    '<div class="tg-handoff-bar nominal">NO HAND DETECTED</div>',
                    unsafe_allow_html=True,
                )
    finally:
        engine.stop()