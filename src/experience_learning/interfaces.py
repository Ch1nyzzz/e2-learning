from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from experience_learning.types import (
    EnvironmentState,
    EpisodeContext,
    Experience,
    JudgeResult,
    TokenEntropyPrediction,
)


class Environment(Protocol):
    def reset(self) -> EnvironmentState: ...
    def step(self, action: str) -> EnvironmentState: ...
    def close(self) -> None: ...


class WorldModel(Protocol):
    def predict(
        self,
        requests: Sequence[tuple[EpisodeContext, str]],
        *,
        samples_per_request: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> list[list[str]]: ...
    def predict_with_token_entropy(
        self,
        requests: Sequence[tuple[EpisodeContext, str]],
        *,
        seed: int | None = None,
    ) -> list[TokenEntropyPrediction]: ...
    def learn(self, experiences: Sequence[Experience]) -> dict[str, float]: ...
    def score(self, experiences: Sequence[Experience]) -> dict[str, float]: ...
    def save(self, output_dir: str) -> None: ...


class SemanticJudge(Protocol):
    def compare(
        self,
        *,
        context: EpisodeContext,
        action: str,
        left: str,
        right: str,
    ) -> JudgeResult: ...
