from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class HistoryStep:
    action: str
    observation: str


@dataclass
class EpisodeContext:
    initial_observation: str
    history: list[HistoryStep] = field(default_factory=list)

    def append(self, action: str, observation: str) -> None:
        self.history.append(HistoryStep(action=action, observation=observation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_observation": self.initial_observation,
            "history": [asdict(step) for step in self.history],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeContext:
        return cls(
            initial_observation=str(value["initial_observation"]),
            history=[HistoryStep(**step) for step in value.get("history", [])],
        )


@dataclass(frozen=True)
class EnvironmentState:
    observation: str
    admissible_actions: tuple[str, ...]
    done: bool = False
    won: bool = False
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["admissible_actions"] = list(self.admissible_actions)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EnvironmentState:
        return cls(
            observation=str(value["observation"]),
            admissible_actions=tuple(value.get("admissible_actions", [])),
            done=bool(value.get("done", False)),
            won=bool(value.get("won", False)),
            score=float(value.get("score", 0.0)),
            metadata=dict(value.get("metadata", {})),
        )


class Verdict(str, Enum):
    EQUIVALENT = "EQUIVALENT"
    DIFFERENT = "DIFFERENT"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class JudgeResult:
    verdict: Verdict
    confidence: float = 1.0
    rationale: str = ""

    @property
    def equivalent(self) -> bool | None:
        if self.verdict is Verdict.EQUIVALENT:
            return True
        if self.verdict is Verdict.DIFFERENT:
            return False
        return None


@dataclass(frozen=True)
class SemanticCluster:
    representative: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class ActionPrediction:
    action: str
    samples: tuple[str, ...]
    clusters: tuple[SemanticCluster, ...]
    entropy: float
    generated_tokens: int | None = None
    hit_token_limit: bool | None = None


@dataclass(frozen=True)
class TokenEntropyPrediction:
    observation: str
    mean_token_entropy: float
    generated_tokens: int
    hit_token_limit: bool


@dataclass(frozen=True)
class PolicyDecision:
    raw_response: str
    action: str | None


@dataclass(frozen=True)
class AcquisitionDecision:
    action: str
    score: float
    predictions: tuple[ActionPrediction, ...]


@dataclass(frozen=True)
class Experience:
    context: EpisodeContext
    action: str
    predicted_observation: str
    actual_observation: str
    step: int
    episode: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "action": self.action,
            "predicted_observation": self.predicted_observation,
            "actual_observation": self.actual_observation,
            "step": self.step,
            "episode": self.episode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Experience:
        return cls(
            context=EpisodeContext.from_dict(value["context"]),
            action=str(value["action"]),
            predicted_observation=str(value.get("predicted_observation", "")),
            actual_observation=str(value["actual_observation"]),
            step=int(value.get("step", 0)),
            episode=int(value.get("episode", 0)),
        )
