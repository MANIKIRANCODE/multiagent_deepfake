from agents.decision_agent import DecisionAgent

def test_decision_agent_does_not_invent_verdict():
    agent = DecisionAgent()
    result = agent.analyze([
        {"agent": "anti_spoofing", "status": "NOT_CONFIGURED", "score": None},
        {"agent": "deepfake_detection", "status": "NOT_CONFIGURED", "score": None},
    ])
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["score"] is None
