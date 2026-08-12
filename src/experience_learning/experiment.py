from __future__ import annotations

import ast
import json
import random
import shutil
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experience_learning.acquisition import select_by_semantic_entropy, select_by_token_entropy
from experience_learning.config import AppConfig
from experience_learning.distributed import broadcast_object
from experience_learning.environment import filter_actions
from experience_learning.interfaces import Environment, SemanticJudge, WorldModel
from experience_learning.logging import JsonlEventLogger
from experience_learning.types import (
    EnvironmentState,
    EpisodeContext,
    Experience,
    TokenEntropyPrediction,
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
        environment: Environment | Sequence[Environment] | None,
        judge: SemanticJudge | None,
    ):
        if environment is None:
            environments: list[Environment] = []
        elif isinstance(environment, Sequence):
            environments = list(environment)
        else:
            environments = [environment]
        if is_main_process and (not environments or judge is None):
            raise ValueError("rank 0 requires both environment and judge")
        if is_main_process and len(environments) != config.experiment.parallel_environments:
            raise ValueError("rank 0 environment count must match experiment.parallel_environments")
        self.config = config
        self.model = model
        self.is_main_process = is_main_process
        self.environments = environments
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
        self.recent_errors: deque[int] = deque(maxlen=config.experiment.stop_error_rate_window)
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

    def _log_events(self, records: Sequence[tuple[str, dict[str, Any]]]) -> None:
        def operation() -> dict[str, Any]:
            assert self.logger is not None
            for event_type, payload in records:
                self.logger.write(event_type, **payload)
            return {}

        self._controller_packet(operation)

    def _reset_many(self, environment_indices: Sequence[int]) -> list[EnvironmentState]:
        def operation() -> list[dict[str, Any]]:
            return [self.environments[index].reset().to_dict() for index in environment_indices]

        payload = self._controller_packet(operation)
        return [EnvironmentState.from_dict(item) for item in payload]

    def _select_actions(
        self,
        items: Sequence[
            tuple[
                EpisodeContext,
                tuple[str, ...],
                list[list[str]],
                list[TokenEntropyPrediction] | None,
            ]
        ],
    ) -> list[dict[str, Any]]:
        def operation() -> list[dict[str, Any]]:
            assert self.judge is not None
            decisions: list[dict[str, Any]] = []
            for context, actions, predictions, token_predictions in items:
                predictions_by_action = dict(zip(actions, predictions, strict=True))
                if self.config.acquisition.strategy == "token_entropy":
                    if token_predictions is None or len(token_predictions) != len(actions):
                        raise ValueError("token entropy acquisition requires one score per action")
                    decision = select_by_token_entropy(
                        dict(zip(actions, token_predictions, strict=True)),
                        rng=self.rng,
                    )
                    decisions.append(
                        {
                            "action": decision.action,
                            "score": decision.score,
                            "predictions": [asdict(item) for item in decision.predictions],
                        }
                    )
                    continue
                if self.config.acquisition.strategy == "random":
                    action = self.rng.choice(list(actions))
                    decisions.append(
                        {
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
                    )
                    continue
                decision = select_by_semantic_entropy(
                    predictions_by_action,
                    lambda action, left, right, context=context: self.judge.compare(
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
                decisions.append(
                    {
                        "action": decision.action,
                        "score": decision.score,
                        "predictions": [asdict(item) for item in decision.predictions],
                    }
                )
            return decisions

        return self._controller_packet(operation)

    def _step_and_judge_many(
        self,
        jobs: Sequence[tuple[int, EpisodeContext, str, str, int, int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        def operation() -> list[dict[str, Any]]:
            assert self.judge is not None

            # TextWorld's PDDL parser has process-global parser state and is not thread-safe.
            # Environment transitions are cheap, so execute them serially and only parallelize
            # the remote semantic-judge requests.
            next_states = [
                self.environments[job[0]].step(job[2])
                for job in jobs
            ]

            def judge_transition(
                item: tuple[
                    tuple[int, EpisodeContext, str, str, int, int, dict[str, Any]],
                    EnvironmentState,
                ],
            ) -> dict[str, Any]:
                job, next_state = item
                (
                    _,
                    context,
                    action,
                    point_prediction,
                    episode,
                    step,
                    acquisition,
                ) = job
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
                    step=step,
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

            workers = min(self.config.judge.max_concurrency, len(jobs))
            items = list(zip(jobs, next_states, strict=True))
            if workers <= 1:
                return [judge_transition(item) for item in items]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(judge_transition, items))

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
            pruned = self._prune_periodic_checkpoints()
            if pruned:
                self.logger.write("checkpoint_pruned", paths=pruned)
            return {}

        self._controller_packet(operation)

    def _prune_periodic_checkpoints(self) -> list[str]:
        """Keep only the newest configured number of env-step checkpoints."""
        keep = self.config.training.max_periodic_checkpoints_to_keep
        if keep == 0:
            return []
        checkpoint_root = self.output_dir / "checkpoints"
        periodic = sorted(
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir()
            and path.name.startswith("env_step_")
            and path.name.removeprefix("env_step_").isdigit()
        )
        removed: list[str] = []
        for path in periodic[:-keep]:
            shutil.rmtree(path)
            removed.append(str(path))
        return removed

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
        pending_updates: list[Experience] = []

        def apply_pending_updates(*, force: bool) -> dict[str, float]:
            metrics: dict[str, float] = {}
            batch_size = self.config.training.update_batch_size
            while len(pending_updates) >= batch_size or (force and pending_updates):
                take = min(batch_size, len(pending_updates))
                batch = pending_updates[:take]
                del pending_updates[:take]
                for _ in range(self.config.training.updates_per_mistake):
                    metrics = self.model.learn(batch)
                self.optimizer_step = int(metrics["optimizer_step"])
            return metrics

        try:
            # Each environment owns a disjoint game partition. Replaying the resets assigned to
            # each slot reconstructs its deterministic iterator at an episode-boundary checkpoint.
            for episode in range(completed_episodes):
                self._reset_many([episode % self.config.experiment.parallel_environments])

            stop_requested = False
            while (
                completed_episodes < self.config.experiment.max_episodes
                and global_step < self.config.experiment.max_environment_steps
                and not stop_requested
            ):
                wave_end = min(
                    completed_episodes + self.config.experiment.parallel_environments,
                    self.config.experiment.max_episodes,
                )
                episode_ids = list(range(completed_episodes, wave_end))
                environment_indices = [
                    episode % self.config.experiment.parallel_environments
                    for episode in episode_ids
                ]
                states = self._reset_many(environment_indices)
                contexts = [
                    EpisodeContext(initial_observation=state.observation) for state in states
                ]
                self._log_events(
                    [
                        (
                            "episode_start",
                            {
                                "episode": episode,
                                "environment_slot": environment_index,
                                "global_step": global_step,
                                "gamefile": state.metadata.get("gamefile", ""),
                                "initial_observation": state.observation,
                            },
                        )
                        for episode, environment_index, state in zip(
                            episode_ids, environment_indices, states, strict=True
                        )
                    ]
                )
                active = [index for index, state in enumerate(states) if not state.done]

                while active and global_step < self.config.experiment.max_environment_steps:
                    actions_by_slot: dict[int, tuple[str, ...]] = {}
                    for slot in active:
                        actions = filter_actions(
                            states[slot].admissible_actions,
                            excluded=self.config.environment.excluded_actions,
                            maximum=self.config.environment.max_candidate_actions,
                        )
                        if actions:
                            actions_by_slot[slot] = actions
                    active = [slot for slot in active if slot in actions_by_slot]
                    if not active:
                        break
                    remaining = self.config.experiment.max_environment_steps - global_step
                    round_slots = active[:remaining]
                    generation_seed = self.config.experiment.seed + global_step * 1009
                    requests: list[tuple[EpisodeContext, str]] = []
                    spans: list[tuple[int, int]] = []
                    for slot in round_slots:
                        start = len(requests)
                        requests.extend(
                            (contexts[slot], action) for action in actions_by_slot[slot]
                        )
                        spans.append((start, len(requests)))

                    flat_token_predictions: list[TokenEntropyPrediction] | None = None
                    if self.config.acquisition.strategy == "token_entropy":
                        flat_token_predictions = self.model.predict_with_token_entropy(
                            requests,
                            seed=generation_seed,
                        )
                        flat_predictions = [
                            [prediction.observation] for prediction in flat_token_predictions
                        ]
                    else:
                        flat_predictions = self.model.predict(
                            requests,
                            samples_per_request=self.config.generation.samples_per_action,
                            do_sample=True,
                            seed=generation_seed,
                        )

                    selection_items = []
                    for slot, (start, end) in zip(round_slots, spans, strict=True):
                        token_slice = (
                            None
                            if flat_token_predictions is None
                            else flat_token_predictions[start:end]
                        )
                        selection_items.append(
                            (
                                contexts[slot],
                                actions_by_slot[slot],
                                flat_predictions[start:end],
                                token_slice,
                            )
                        )
                    acquisitions = self._select_actions(selection_items)

                    point_generation_seed = generation_seed
                    point_prediction_reused = self.config.acquisition.strategy == "token_entropy"
                    if point_prediction_reused:
                        point_predictions = []
                        for slot, (start, _), acquisition in zip(
                            round_slots, spans, acquisitions, strict=True
                        ):
                            chosen_index = actions_by_slot[slot].index(str(acquisition["action"]))
                            point_predictions.append(flat_predictions[start + chosen_index][0])
                    else:
                        point_generation_seed = generation_seed + 1
                        point_predictions = [
                            outcomes[0]
                            for outcomes in self.model.predict(
                                [
                                    (contexts[slot], str(acquisition["action"]))
                                    for slot, acquisition in zip(
                                        round_slots, acquisitions, strict=True
                                    )
                                ],
                                samples_per_request=1,
                                do_sample=False,
                                seed=point_generation_seed,
                            )
                        ]

                    model_version_before = self.optimizer_step
                    jobs = [
                        (
                            environment_indices[slot],
                            contexts[slot],
                            str(acquisition["action"]),
                            point_prediction,
                            episode_ids[slot],
                            global_step + index,
                            acquisition,
                        )
                        for index, (slot, acquisition, point_prediction) in enumerate(
                            zip(
                                round_slots,
                                acquisitions,
                                point_predictions,
                                strict=True,
                            )
                        )
                    ]
                    packets = self._step_and_judge_many(jobs)

                    round_results: list[
                        tuple[int, Experience, Verdict, EnvironmentState, dict[str, Any]]
                    ] = []
                    for slot, packet in zip(round_slots, packets, strict=True):
                        experience = Experience.from_dict(packet["experience"])
                        verdict = Verdict(packet["verdict"])
                        next_state = EnvironmentState.from_dict(packet["next_state"])
                        if verdict is Verdict.DIFFERENT:
                            mistakes += 1
                            self.recent_errors.append(1)
                        elif verdict is Verdict.EQUIVALENT:
                            self.recent_errors.append(0)
                        else:
                            uncertain += 1
                        should_update = (
                            self.config.training.update_gate == "all_transitions"
                            or verdict is Verdict.DIFFERENT
                        )
                        if should_update:
                            pending_updates.append(experience)
                        round_results.append((slot, experience, verdict, next_state, packet))

                    metrics = apply_pending_updates(force=False)
                    transition_records: list[tuple[str, dict[str, Any]]] = []
                    for index, (
                        slot,
                        experience,
                        verdict,
                        next_state,
                        packet,
                    ) in enumerate(round_results):
                        event_global_step = global_step + index + 1
                        transition_records.append(
                            (
                                "transition",
                                {
                                    "episode": episode_ids[slot],
                                    "environment_slot": environment_indices[slot],
                                    "global_step": event_global_step,
                                    "optimizer_step": self.optimizer_step,
                                    "model_version_before": model_version_before,
                                    "model_version_after": self.optimizer_step,
                                    "generation_seed": generation_seed,
                                    "point_generation_seed": point_generation_seed,
                                    "point_prediction_reused": point_prediction_reused,
                                    "context": experience.context.to_dict(),
                                    "action": experience.action,
                                    "prediction": experience.predicted_observation,
                                    "observation": experience.actual_observation,
                                    "verdict": verdict.value,
                                    "judge_confidence": packet["judge_confidence"],
                                    "judge_rationale": packet["judge_rationale"],
                                    "acquisition": packet["acquisition"],
                                    "won": next_state.won,
                                    "score": next_state.score,
                                    "update_metrics": metrics,
                                    "pending_update_examples": len(pending_updates),
                                },
                            )
                        )
                        contexts[slot].append(experience.action, experience.actual_observation)
                        states[slot] = next_state
                    global_step += len(round_results)
                    self._log_events(transition_records)
                    active = [slot for slot in active if not states[slot].done]

                    checkpoint_every = self.config.training.checkpoint_every_environment_steps
                    if (
                        checkpoint_every > 0
                        and global_step // checkpoint_every
                        > self.last_checkpoint_environment_step // checkpoint_every
                    ):
                        checkpoint_due = True
                    if self._should_stop_from_error_rate():
                        stop_requested = True
                        break

                completed_episodes = wave_end
                self._log_events(
                    [
                        (
                            "episode_end",
                            {
                                "episode": episode,
                                "environment_slot": environment_index,
                                "global_step": global_step,
                                "won": state.won,
                                "score": state.score,
                            },
                        )
                        for episode, environment_index, state in zip(
                            episode_ids, environment_indices, states, strict=True
                        )
                    ]
                )
                if checkpoint_due:
                    apply_pending_updates(force=True)
                    self._checkpoint(
                        f"env_step_{global_step:06d}",
                        next_episode=completed_episodes,
                        global_step=global_step,
                        mistakes=mistakes,
                        uncertain=uncertain,
                    )
                    self.last_checkpoint_environment_step = global_step
                    checkpoint_due = False
            apply_pending_updates(force=True)
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
            if self.is_main_process:
                for environment in self.environments:
                    environment.close()
            if self.logger is not None:
                self.logger.close()
