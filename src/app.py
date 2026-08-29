"""
app.py
The app shell: loads the design system CSS, renders the problem-statement
panel and architecture diagram, and hosts the Live Demo + Validation
Replay tabs.

Usage (run from inside src/, or adjust paths accordingly):
    streamlit run src/app.py
"""

import streamlit as st
import os

import live_view
import replay_view

st.set_page_config(page_title="TrajGuard", layout="wide")


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_problem_statement():
    st.markdown(
        """<div class="tg-card">
        <div class="tg-metric-label">THE PROBLEM</div>
        <div style="margin-top:6px; font-size:14px; line-height:1.5;">
        Vision-language models are documented to struggle with fine-grained temporal reasoning —
        distinguishing subtle frame-to-frame motion differences. In deployed systems, this shows up
        as <b>alert fatigue</b>: motion-triggered monitoring floods operators with non-actionable noise,
        while genuinely early warning signs get missed.
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_architecture():
    col1, col2, col3, col4, col5 = st.columns([2, 0.5, 2, 0.5, 2])
    with col1:
        st.markdown('<div class="tg-architecture-box">Layer 1<br>Kinematic Pre-Filter<br><span style="opacity:0.6;">(lightweight, physics-grounded)</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="tg-architecture-arrow">→</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="tg-architecture-box">Sustained &amp; Rising<br>Risk Gate<br><span style="opacity:0.6;">(suppresses false alarms)</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="tg-architecture-arrow">→</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="tg-architecture-box">Layer 2<br>VLM Explainer<br><span style="opacity:0.6;">(called only on confirmed risk)</span></div>', unsafe_allow_html=True)


def main():
    load_css()
    st.title("TrajGuard")
    st.caption("A Fine-Grained Kinematic Pre-Filter for Physical AI Systems")

    render_problem_statement()
    render_architecture()

    st.markdown("---")

    tab1, tab2 = st.tabs(["🟢 Live Demo", "📼 Validation Replay"])

    with tab1:
        live_view.render()

    with tab2:
        replay_view.render()


if __name__ == "__main__":
    main()