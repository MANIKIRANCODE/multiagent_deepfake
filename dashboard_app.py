import streamlit as st

st.set_page_config(page_title="Multi-Agent Anti-Spoofing", layout="wide")
st.title("Real-Time Multi-Agent Anti-Spoofing")
st.caption("Research prototype dashboard — live camera processing runs in run_webcam.py.")

st.warning(
    "The included starter does not yet contain trained anti-spoofing or deepfake models. "
    "Those agents report NOT_CONFIGURED rather than generating fake predictions."
)

left, right = st.columns(2)
with left:
    st.subheader("Agent architecture")
    st.markdown("""
    1. Face Detection Agent — OpenCV
    2. Face Anti-Spoofing Agent — model integration pending
    3. Lip-Synchronization Agent — MediaPipe + exploratory signal correlation
    4. Audio Analysis Agent — microphone energy / activity
    5. Deepfake Detection Agent — model integration pending
    6. Decision Agent — evidence sufficiency checks
    """)
with right:
    st.subheader("Current prototype status")
    st.metric("Trained PAD model", "Not configured")
    st.metric("Trained deepfake model", "Not configured")
    st.metric("Lip/audio analysis", "Experimental")
    st.metric("Decision output", "Insufficient evidence by design")

st.info("Run `python run_webcam.py` in a terminal to start live webcam analysis.")
st.subheader("Planned next milestones")
st.markdown("""
- Integrate a validated presentation-attack detection model.
- Integrate SyncNet or another validated audio-visual synchronization model.
- Integrate a validated deepfake detector.
- Add synchronized event logging and report export.
- Evaluate false-positive rate, precision, recall, latency, and FPS.
""")
