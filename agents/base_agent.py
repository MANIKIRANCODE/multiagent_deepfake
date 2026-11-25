from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Common interface for independent analysis agents."""
    name = "base"

    @abstractmethod
    def analyze(self, **kwargs) -> dict:
        raise NotImplementedError

    def result(self, status="OK", score=None, details="", **extra):
        return {
            "agent": self.name,
            "status": status,
            "score": score,
            "details": details,
            **extra,
        }
