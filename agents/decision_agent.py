
from .base_agent import BaseAgent


class DecisionAgent(BaseAgent):
    name = "decision_making"

    def analyze(self, agent_results):

        # Find results from the two visual classification models.
        agents = {
            item.get("agent"): item
            for item in agent_results
        }

        anti = agents.get("anti_spoofing", {})
        deepfake = agents.get("deepfake_detection", {})
        lip = agents.get("lip_sync", {})
        audio = agents.get("audio_analysis", {})

        # Both models must provide usable predictions.
        if anti.get("status") != "OK" or deepfake.get("status") != "OK":
            return self.result(
                status="INSUFFICIENT_EVIDENCE",
                score=None,
                details="One or both visual classification models are unavailable.",
            )

        anti_fake = anti.get("fake_score")
        deep_fake = deepfake.get("fake_score")

        if anti_fake is None or deep_fake is None:
            return self.result(
                status="INSUFFICIENT_EVIDENCE",
                score=None,
                details="Required visual model scores are missing.",
            )

        anti_fake = float(anti_fake)
        deep_fake = float(deep_fake)

        # Reject invalid model outputs.
        if not (
            0.0 <= anti_fake <= 1.0
            and 0.0 <= deep_fake <= 1.0
        ):
            return self.result(
                status="INSUFFICIENT_EVIDENCE",
                score=None,
                details="Invalid model probability values.",
            )

        # Provisional fusion score, not a calibrated probability.
        risk_score = (anti_fake + deep_fake) / 2.0

        # Conservative provisional decision rules.
        if anti_fake >= 0.85 and deep_fake >= 0.85:
            assessment = "HIGH RISK"
            explanation = (
                "Both visual models show strong indications of "
                "a possible presentation attack or manipulation."
            )

        elif anti_fake >= 0.60 or deep_fake >= 0.60:
            assessment = "SUSPICIOUS"
            explanation = (
                "At least one visual model reports an elevated "
                "manipulation or spoofing score."
            )

        elif anti_fake <= 0.20 and deep_fake <= 0.20:
            assessment = "LOW RISK"
            explanation = (
                "Both visual models report relatively low "
                "manipulation scores. This does not prove authenticity."
            )

        else:
            assessment = "INSUFFICIENT_EVIDENCE"
            explanation = (
                "The visual model results are inconclusive."
            )

        return self.result(
            status=assessment,
            score=round(risk_score, 4),
            details=explanation,
            anti_spoof_fake_score=anti_fake,
            deepfake_score=deep_fake,
            lip_sync_status=lip.get("status"),
            audio_status=audio.get("status"),
            evidence_count=2,
            calibration="PROVISIONAL_UNVALIDATED"
        )
