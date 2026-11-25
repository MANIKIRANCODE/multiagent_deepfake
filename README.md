# Real-Time Multi-Agent Face Anti-Spoofing & Audio-Visual Verification

A modular **research prototype starter** for analyzing an authorized webcam stream.
It includes:
- OpenCV face detection and tracking-by-frame
- Optional MediaPipe Face Mesh mouth-landmark extraction
- Optional microphone speech-activity energy signal
- A simple audio/mouth-motion synchronization analysis
- A central evidence-fusion decision agent
- Explicit `INSUFFICIENT_EVIDENCE` outputs when a real trained detector is not installed

## Important limitations

This is a research prototype, **not a production identity-verification or fraud-detection system**.
The included baseline does not contain a trained presentation-attack detector or a trained deepfake detector. Those agents deliberately report `NOT_CONFIGURED` rather than inventing predictions. Face detection only means that a face-like region was detected; it does not prove that the face is live or genuine.

Lip/audio correlation is only a rough signal. Network delay, microphone placement, silence, multiple speakers, dubbing, and poor lighting can cause mismatches. A mismatch is not proof of spoofing. Use informed consent and only analyze streams you are authorized to process.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 recommended
- Webcam
- Microphone (optional)
- VS Code or another Python IDE

## Setup (Windows Anaconda Prompt)

```bash
conda create -n antispoof python=3.10 -y
conda activate antispoof
cd anti_spoof_multi_agent
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If MediaPipe installation fails on your machine, the app still runs face detection without mouth landmarks. Install a compatible MediaPipe build for your Python/OS combination.

## Run

```bash
python run_webcam.py
```

Keyboard:
- `q`: quit
- `m`: toggle microphone capture (microphone starts enabled if available)
- `r`: reset recent signal history

To run the dashboard:

```bash
streamlit run dashboard_app.py
```

The Streamlit dashboard is a status/architecture dashboard. The OpenCV window is the live webcam application.

## Project structure

```text
anti_spoof_multi_agent/
├── run_webcam.py
├── dashboard_app.py
├── requirements.txt
├── agents/
│   ├── base_agent.py
│   ├── face_detection_agent.py
│   ├── anti_spoof_agent.py
│   ├── lip_sync_agent.py
│   ├── audio_agent.py
│   ├── deepfake_agent.py
│   └── decision_agent.py
├── core/
│   ├── audio_stream.py
│   ├── video_stream.py
│   └── orchestrator.py
├── tests/
│   └── test_decision_agent.py
├── models/README.md
└── logs/
```

## Agent outputs

Each agent returns a dictionary with a consistent structure:
- `agent`: agent name
- `status`: `OK`, `WARNING`, `NOT_CONFIGURED`, or `ERROR`
- `score`: optional numeric score
- `details`: explanation

Scores are not interchangeable across models. The decision agent only combines signals explicitly marked as usable. It returns `INSUFFICIENT_EVIDENCE` if the required trained models are unavailable.

## Next development milestones

1. Integrate and validate a trained face presentation-attack detection model.
2. Integrate a validated audio-visual synchronization model (e.g. SyncNet) and calibrate for stream delay.
3. Integrate a validated deepfake detector.
4. Evaluate on held-out datasets and report APCER/BPCER, precision, recall, F1, false-positive rate, latency, and FPS.
5. Only then consider meeting-platform integration. Do not bypass platform permissions or record participants without consent.
