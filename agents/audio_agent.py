from .base_agent import BaseAgent

class AudioAgent(BaseAgent):
    name = "audio_analysis"

    def analyze(self, audio_level=None, available=False):
        if not available or audio_level is None:
            return self.result(status="NOT_CONFIGURED", score=None,
                               details="Microphone signal unavailable; audio analysis skipped.")
        return self.result(
            status="OK",
            score=None,
            details="Microphone energy measured. Energy indicates activity, not speaker identity or authenticity.",
            audio_level=float(audio_level),
            speech_activity=bool(audio_level > 0.015),
        )
