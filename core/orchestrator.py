
from agents.face_detection_agent import FaceDetectionAgent
from agents.anti_spoof_agent import AntiSpoofAgent
from agents.lip_sync_agent import LipSyncAgent
from agents.audio_agent import AudioAgent
from agents.deepfake_agent import DeepfakeAgent
from agents.decision_agent import DecisionAgent


class MultiAgentOrchestrator:

    def __init__(self):
        self.face_agent = FaceDetectionAgent()
        self.anti_spoof_agent = AntiSpoofAgent()
        self.lip_agent = LipSyncAgent()
        self.audio_agent = AudioAgent()
        self.deepfake_agent = DeepfakeAgent()
        self.decision_agent = DecisionAgent()

    def analyze_frame(
        self,
        frame,
        audio_level=None,
        audio_available=False,
        audio_samples=None
    ):

        face = self.face_agent.analyze(frame)

        first_face = (
            face["faces"][0]
            if face["faces"]
            else None
        )

        anti = self.anti_spoof_agent.analyze(
            frame=frame,
            face=first_face
        )

        lip = self.lip_agent.analyze(
            frame=frame,
            audio_level=audio_level or 0.0,
            audio_samples=audio_samples
        )

        audio = self.audio_agent.analyze(
            audio_level=audio_level,
            available=audio_available
        )

        deepfake = self.deepfake_agent.analyze(
            frame=frame,
            face=first_face
        )

        results = [
            face,
            anti,
            lip,
            audio,
            deepfake
        ]

        decision = self.decision_agent.analyze(results)

        return {
            "agents": results,
            "decision": decision
        }
