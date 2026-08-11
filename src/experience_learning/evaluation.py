from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experience_learning.config import AppConfig, EnvironmentConfig
from experience_learning.distributed import broadcast_object
from experience_learning.environment import ALFWorldTextEnvironment, filter_actions
from experience_learning.interfaces import SemanticJudge, WorldModel
from experience_learning.logging import JsonlEventLogger
from experience_learning.types import EpisodeContext, Experience, Verdict


@dataclass(frozen=True)
class ProbeTransition:
    split: str
    gamefile: str
    episode: int
    step: int
    context: EpisodeContext
    action: str
    actual_observation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "gamefile": self.gamefile,
            "episode": self.episode,
            "step": self.step,
            "context": self.context.to_dict(),
            "action": self.action,
            "actual_observation": self.actual_observation,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProbeTransition:
        return cls(
            split=str(value["split"]),
            gamefile=str(value["gamefile"]),
            episode=int(value["episode"]),
            step=int(value["step"]),
            context=EpisodeContext.from_dict(value["context"]),
            action=str(value["action"]),
            actual_observation=str(value["actual_observation"]),
        )


def collect_random_probes(
    config: AppConfig,
    *,
    split: str,
    episodes: int,
    output_path: str | Path,
) -> dict[str, int]:
    env_config = EnvironmentConfig(**vars(config.environment))
    env_config.split = split
    environment = ALFWorldTextEnvironment(env_config, config.experiment.seed)
    rng = random.Random(config.experiment.seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    transitions = 0
    completed = 0
    with output.open("w", encoding="utf-8") as handle:
        try:
            for episode in range(episodes):
                state = environment.reset()
                gamefile = str(state.metadata.get("gamefile", ""))
                context = EpisodeContext(initial_observation=state.observation)
                step = 0
                while not state.done:
                    actions = filter_actions(
                        state.admissible_actions,
                        excluded=config.environment.excluded_actions,
                        maximum=config.environment.max_candidate_actions,
                    )
                    if not actions:
                        break
                    action = rng.choice(actions)
                    next_state = environment.step(action)
                    record = ProbeTransition(
                        split=split,
                        gamefile=gamefile,
                        episode=episode,
                        step=step,
                        context=EpisodeContext.from_dict(context.to_dict()),
                        action=action,
                        actual_observation=next_state.observation,
                    )
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    transitions += 1
                    context.append(action, next_state.observation)
                    state = next_state
                    step += 1
                    limit = config.evaluation.max_transitions
                    if limit > 0 and transitions >= limit:
                        return {"episodes": completed + 1, "transitions": transitions}
                completed += 1
        finally:
            environment.close()
    return {"episodes": completed, "transitions": transitions}


def load_probes(path: str | Path, maximum: int = 0) -> list[ProbeTransition]:
    records: list[ProbeTransition] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(ProbeTransition.from_dict(json.loads(line)))
                if maximum > 0 and len(records) >= maximum:
                    break
    return records


def evaluate_probes(
    config: AppConfig,
    *,
    model: WorldModel,
    judge: SemanticJudge | None,
    is_main_process: bool,
    probes_path: str | Path,
    output_path: str | Path,
) -> dict[str, float]:
    records = load_probes(probes_path, config.evaluation.max_transitions)
    correct = 0
    incorrect = 0
    uncertain = 0
    nll_sum = 0.0
    target_tokens = 0.0
    logger = None
    logger_startup: dict[str, Any] | None = None
    if is_main_process:
        try:
            logger = JsonlEventLogger(output_path)
            logger_startup = {"phase": "OK"}
        except Exception as exc:
            logger_startup = {
                "phase": "STOP",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    logger_startup = broadcast_object(logger_startup)
    if logger_startup["phase"] == "STOP":
        raise RuntimeError(
            f"evaluation logger failed: {logger_startup['error_type']}: "
            f"{logger_startup['error']}"
        )
    batch_size = config.evaluation.batch_size

    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            predictions = model.predict(
                [(record.context, record.action) for record in batch],
                samples_per_request=1,
                do_sample=False,
                seed=config.experiment.seed + start * 1009,
            )
            packet: dict[str, Any] | None = None
            if is_main_process:
                assert judge is not None
                try:
                    judgments = []
                    for record, samples in zip(batch, predictions, strict=True):
                        point_prediction = samples[0]
                        result = judge.compare(
                            context=record.context,
                            action=record.action,
                            left=point_prediction,
                            right=record.actual_observation,
                        )
                        judgments.append(
                            {
                                "prediction": point_prediction,
                                "verdict": result.verdict.value,
                                "confidence": result.confidence,
                            }
                        )
                    packet = {"phase": "OK", "judgments": judgments}
                except Exception as exc:
                    packet = {
                        "phase": "STOP",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            packet = broadcast_object(packet)
            if packet["phase"] == "STOP":
                raise RuntimeError(
                    f"evaluation judge failed: {packet['error_type']}: {packet['error']}"
                )

            experiences = [
                Experience(
                    context=record.context,
                    action=record.action,
                    predicted_observation="",
                    actual_observation=record.actual_observation,
                    step=record.step,
                    episode=record.episode,
                )
                for record in batch
            ]
            scores = model.score(experiences)
            nll_sum += scores["target_nll_sum"]
            target_tokens += scores["target_tokens"]
            log_packet: dict[str, Any] | None = None
            if is_main_process:
                try:
                    assert logger is not None
                    for record, judgment in zip(batch, packet["judgments"], strict=True):
                        logger.write("probe", probe=record.to_dict(), judgment=judgment)
                    log_packet = {"phase": "OK"}
                except Exception as exc:
                    log_packet = {
                        "phase": "STOP",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            log_packet = broadcast_object(log_packet)
            if log_packet["phase"] == "STOP":
                raise RuntimeError(
                    f"evaluation logging failed: {log_packet['error_type']}: "
                    f"{log_packet['error']}"
                )

            for judgment in packet["judgments"]:
                verdict = Verdict(judgment["verdict"])
                correct += int(verdict is Verdict.EQUIVALENT)
                incorrect += int(verdict is Verdict.DIFFERENT)
                uncertain += int(verdict is Verdict.UNCERTAIN)
    finally:
        if logger is not None:
            logger.close()

    definite = correct + incorrect
    total = definite + uncertain
    return {
        "transitions": float(total),
        "semantic_correct": float(correct),
        "semantic_incorrect": float(incorrect),
        "semantic_uncertain": float(uncertain),
        "semantic_accuracy_definite": correct / definite if definite else 0.0,
        "semantic_accuracy_conservative": correct / total if total else 0.0,
        "semantic_accuracy_upper_bound": (correct + uncertain) / total if total else 0.0,
        "judge_coverage": definite / total if total else 0.0,
        "target_token_nll": nll_sum / target_tokens if target_tokens else 0.0,
        "target_tokens": target_tokens,
    }
