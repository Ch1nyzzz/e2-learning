from __future__ import annotations

import ast
import json
import random
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experience_learning.acquisition import select_by_semantic_entropy
from experience_learning.config import AppConfig
from experience_learning.distributed import broadcast_object
from experience_learning.environment import filter_actions
from experience_learning.interfaces import Environment, SemanticJudge, WorldModel
from experience_learning.logging import JsonlEventLogger
from experience_learning.types import (
    EnvironmentState,
    EpisodeContext,
    Experience,
    Verdict,
)


class OnlineExperienceExperiment:
    """Synchronous distributed controller for acquisition, reality checks, and gated updates."""

    def __init__(
        self,
        *,
        config: AppConfig,
        model: WorldModel,
        is_main_process: bool,
        environment: Environment | None,
        judge: SemanticJudge | None,
    ):
        if is_main_process and (environment is None or judge is None):
            raise ValueError("rank 0 requires both environment and judge")
        self.config = config
        self.model = model
        self.is_main_process = is_main_process
        self.environment = environment
        self.judge = judge
        self.rng = random.Random(config.experiment.seed)
        self.output_dir = Path(config.experiment.output_dir)
        self.logger: JsonlEventLogger | None = None
        logger_startup: dict[str, Any] | None = None
        if is_main_process:
            try:
                self.logger = JsonlEventLogger(self.output_dir / "events.jsonl")
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
                "event logger startup failed: "
                f"{logger_startup['error_type']}: {logger_startup['error']}"
            )
        self.recent_errors: deque[int] = deque(
            maxlen=config.experiment.stop_error_rate_window
        )
        self.optimizer_step = 0
        self.last_checkpoint_environment_step = 0

    def _controller_packet(self, operation: Any) -> Any:
        packet: dict[str, Any] | None = None
        if self.is_main_process:
            try:
                packet = {"phase": "OK", "payload": operation()}
            except Exception as exc:  # broadcast failure instead of stranding NCCL peers
                packet = {
                    "phase": "STOP",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        packet = broadcast_object(packet)
        if packet["phase"] == "STOP":
            raise RuntimeError(
                f"rank-0 controller failed: {packet['error_type']}: {packet['error']}"
            )
        return packet["payload"]

    def _log_event(self, event_type: str, **payload: Any) -> None:
        def operation() -> dict[str, Any]:
            assert self.logger is not None
            self.logger.write(event_type, **payload)
            return {}

        self._controller_packet(operation)

    def _reset(self) -> EnvironmentState:
        payload = self._controller_packet(
            lambda: self.environment.reset().to_dict()  # type: ignore[union-attr]
        )
        return EnvironmentState.from_dict(payload)

    def _select_action(
        self,
        *,
        context: EpisodeContext,
        actions: tuple[str, ...],
        predictions: list[list[str]],
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            assert self.judge is not None
            predictions_by_action = dict(zip(actions, predictions, strict=True))
            if self.config.acquisition.strategy == "random":
                action = self.rng.choice(list(actions))
                return {
                    "action": action,
                    "score": None,
                    "predictions": [
                        {
                            "action": candidate,
                            "samples": samples,
                            "clusters": [],
                            "entropy": None,
                        }
                        for candidate, samples in predictions_by_action.items()
                    ],
                }
            decision = select_by_semantic_entropy(
                predictions_by_action,
                lambda action, left, right: self.judge.compare(
                    context=context,
                    action=action,
                    left=left,
                    right=right,
                ),
                rng=self.rng,
                uncertain_is_distinct=(
                    self.config.acquisition.uncertain_comparisons_are_distinct
                ),
                normalize=self.config.acquisition.normalize_entropy,
            )
            return {
                "action": decision.action,
                "score": decision.score,
                "predictions": [asdict(item) for item in decision.predictions],
            }

        return self._controller_packet(operation)

    def _step_and_judge(
        self,
        *,
        context: EpisodeContext,
        action: str,
        point_prediction: str,
        episode: int,
        global_step: int,
        acquisition: dict[str, Any],
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            assert self.judge is not None
            assert self.environment is not None
            next_state = self.environment.step(action)
            verdict = self.judge.compare(
                context=context,
                action=action,
                left=point_prediction,
                right=next_state.observation,
            )
            experience = Experience(
                context=EpisodeContext.from_dict(context.to_dict()),
                action=action,
                predicted_observation=point_prediction,
                actual_observation=next_state.observation,
                step=global_step,
                episode=episode,
            )
            return {
                "next_state": next_state.to_dict(),
                "experience": experience.to_dict(),
                "verdict": verdict.verdict.value,
                "judge_confidence": verdict.confidence,
                "judge_rationale": verdict.rationale,
                "acquisition": acquisition,
            }

        return self._controller_packet(operation)

    def _should_stop_from_error_rate(self) -> bool:
        threshold = self.config.experiment.stop_error_rate_threshold
        window = self.config.experiment.stop_error_rate_window
        if threshold is None or len(self.recent_errors) < window:
            return False
        return sum(self.recent_errors) / len(self.recent_errors) < threshold

    def _checkpoint(
        self,
        name: str,
        *,
        next_episode: int,
        global_step: int,
        mistakes: int,
        uncertain: int,
    ) -> None:
        path = self.output_dir / "checkpoints" / name
        self.model.save(str(path))
        def operation() -> dict[str, Any]:
            assert self.logger is not None
            controller_state = {
                "next_episode": next_episode,
                "global_step": global_step,
                "optimizer_step": self.optimizer_step,
                "mistakes": mistakes,
                "uncertain": uncertain,
                "recent_errors": list(self.recent_errors),
                "controller_rng_state": repr(self.rng.getstate()),
                "events_byte_offset": self.logger.byte_offset(),
            }
            (path / "controller_state.json").write_text(
                json.dumps(controller_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.logger.write("checkpoint", path=str(path), optimizer_step=self.optimizer_step)
            return {}

        self._controller_packet(operation)

    def _resume_state(self) -> dict[str, Any]:
        default = {
            "next_episode": 0,
            "global_step": 0,
            "optimizer_step": 0,
            "mistakes": 0,
            "uncertain": 0,
            "recent_errors": [],
            "controller_rng_state": None,
        }
        resume_from = self.config.training.resume_from
        if not resume_from:
            return default

        def operation() -> dict[str, Any]:
            path = Path(resume_from) / "controller_state.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"training resume requires controller metadata at {path}; "
                    "evaluation may load model-only checkpoints"
                )
            state = json.loads(path.read_text(encoding="utf-8"))
            if self.logger is not None and "events_byte_offset" in state:
                self.logger.truncate_to(int(state["events_byte_offset"]))
            return state

        state = self._controller_packet(operation)
        if state.get("controller_rng_state"):
            self.rng.setstate(ast.literal_eval(state["controller_rng_state"]))
        self.recent_errors.extend(int(value) for value in state.get("recent_errors", []))
        self.optimizer_step = int(state.get("optimizer_step", 0))
        self.last_checkpoint_environment_step = int(state.get("global_step", 0))
        return state

    def run(self) -> dict[str, int]:
        resume = self._resume_state()
        global_step = int(resume["global_step"])
        completed_episodes = int(resume["next_episode"])
        mistakes = int(resume["mistakes"])
        uncertain = int(resume["uncertain"])
        checkpoint_due = False
        try:
            # Recreate ALFWorld's deterministic game iterator. Checkpoints are controller-complete
            # only at episode boundaries, so no mid-episode environment state needs to be pickled.
            for _ in range(completed_episodes):
                self._reset()
            for episode in range(completed_episodes, self.config.experiment.max_episodes):
                if global_step >= self.config.experiment.max_environment_steps:
                    break
                state = self._reset()
                context = EpisodeContext(initial_observation=state.observation)
                self._log_event(
                    "episode_start",
                    episode=episode,
                    global_step=global_step,
                    gamefile=state.metadata.get("gamefile", ""),
                    initial_observation=state.observation,
                )
                while not state.done:
                    if global_step >= self.config.experiment.max_environment_steps:
                        break
                    actions = filter_actions(
                        state.admissible_actions,
                        excluded=self.config.environment.excluded_actions,
                        maximum=self.config.environment.max_candidate_actions,
                    )
                    if not actions:
                        break
                    generation_seed = self.config.experiment.seed + global_step * 1009
                    predictions = self.model.predict(
                        [(context, action) for action in actions],
                        samples_per_request=self.config.generation.samples_per_action,
                        do_sample=True,
                        seed=generation_seed,
                    )
                    acquisition = self._select_action(
                        context=context,
                        actions=actions,
                        predictions=predictions,
                    )
                    point_generation_seed = generation_seed + 1
                    point_prediction = self.model.predict(
                        [(context, str(acquisition["action"]))],
                        samples_per_request=1,
                        do_sample=False,
                        seed=point_generation_seed,
                    )[0][0]
                    packet = self._step_and_judge(
                        context=context,
                        action=str(acquisition["action"]),
                        point_prediction=point_prediction,
                        episode=episode,
                        global_step=global_step,
                        acquisition=acquisition,
                    )
                    experience = Experience.from_dict(packet["experience"])
                    verdict = Verdict(packet["verdict"])
                    if verdict is Verdict.DIFFERENT:
                        mistakes += 1
                        self.recent_errors.append(1)
                    elif verdict is Verdict.EQUIVALENT:
                        self.recent_errors.append(0)
                    else:
                        uncertain += 1

                    should_update = self.config.training.update_gate == "all_transitions" or (
                        verdict is Verdict.DIFFERENT
                    )
                    metrics: dict[str, float] = {}
                    model_version_before = self.optimizer_step
                    if should_update:
                        for _ in range(self.config.training.updates_per_mistake):
                            metrics = self.model.learn([experience])
                        self.optimizer_step = int(metrics["optimizer_step"])

                    global_step += 1
                    next_state = EnvironmentState.from_dict(packet["next_state"])
                    self._log_event(
                        "transition",
                        episode=episode,
                        global_step=global_step,
                        optimizer_step=self.optimizer_step,
                        model_version_before=model_version_before,
                        model_version_after=self.optimizer_step,
                        generation_seed=generation_seed,
                        point_generation_seed=point_generation_seed,
                        context=experience.context.to_dict(),
                        action=experience.action,
                        prediction=experience.predicted_observation,
                        observation=experience.actual_observation,
                        verdict=verdict.value,
                        judge_confidence=packet["judge_confidence"],
                        judge_rationale=packet["judge_rationale"],
                        acquisition=packet["acquisition"],
                        won=next_state.won,
                        score=next_state.score,
                        update_metrics=metrics,
                    )
                    context.append(experience.action, experience.actual_observation)
                    state = next_state

                    checkpoint_every = (
                        self.config.training.checkpoint_every_environment_steps
                    )
                    if (
                        checkpoint_every > 0
                        and global_step > 0
                        and global_step % checkpoint_every == 0
                        and global_step != self.last_checkpoint_environment_step
                    ):
                        checkpoint_due = True
                    if self._should_stop_from_error_rate():
                        state = EnvironmentState(
                            observation=state.observation,
                            admissible_actions=state.admissible_actions,
                            done=True,
                            won=state.won,
                            score=state.score,
                            metadata={**state.metadata, "error_rate_stop": True},
                        )
                completed_episodes += 1
                self._log_event(
                    "episode_end",
                    episode=episode,
                    global_step=global_step,
                    won=state.won,
                    score=state.score,
                )
                if checkpoint_due:
                    self._checkpoint(
                        f"env_step_{global_step:06d}",
                        next_episode=completed_episodes,
                        global_step=global_step,
                        mistakes=mistakes,
                        uncertain=uncertain,
                    )
                    self.last_checkpoint_environment_step = global_step
                    checkpoint_due = False
                if self._should_stop_from_error_rate():
                    break
            self._checkpoint(
                "final",
                next_episode=completed_episodes,
                global_step=global_step,
                mistakes=mistakes,
                uncertain=uncertain,
            )
            return {
                "environment_steps": global_step,
                "optimizer_steps": self.optimizer_step,
                "episodes": completed_episodes,
                "mistakes": mistakes,
                "uncertain_judgments": uncertain,
            }
        finally:
            if self.is_main_process and self.environment is not None:
                self.environment.close()
            if self.logger is not None:
                self.logger.close()
