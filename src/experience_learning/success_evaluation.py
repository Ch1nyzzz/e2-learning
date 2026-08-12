from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from experience_learning.distributed import broadcast_object
from experience_learning.environment import filter_actions
from experience_learning.interfaces import Environment, WorldModel
from experience_learning.types import EnvironmentState, EpisodeContext

T = TypeVar("T")


def _rank_zero_call(is_main_process: bool, operation: Callable[[], T]) -> T:
    packet: dict[str, Any] | None = None
    if is_main_process:
        try:
            packet = {"ok": True, "value": operation()}
        except Exception as exc:
            packet = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    packet = broadcast_object(packet)
    if not packet["ok"]:
        raise RuntimeError(
            f"rank-0 SR evaluation failed: {packet['error_type']}: {packet['error']}"
        )
    return packet["value"]


def evaluate_success_rate(
    *,
    model: WorldModel,
    environments: Sequence[Environment],
    is_main_process: bool,
    split: str,
    episodes: int,
    parallelism: int,
    max_steps: int,
    report_step: int,
    max_action_tokens: int,
    seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    if episodes < 1 or parallelism < 1:
        raise ValueError("episodes and parallelism must be positive")
    if not 0 < report_step <= max_steps:
        raise ValueError("report_step must be between one and max_steps")
    if is_main_process and len(environments) != parallelism:
        raise ValueError("rank 0 requires one environment per parallel slot")

    output = Path(output_path)
    handle = None
    if is_main_process:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", encoding="utf-8")
    completed = 0
    wins_at_report = 0
    wins_at_max = 0
    invalid_actions = 0
    total_steps = 0
    success_steps: list[int] = []
    task_results: dict[str, dict[str, int]] = {}

    try:
        while completed < episodes:
            width = min(parallelism, episodes - completed)
            episode_ids = list(range(completed, completed + width))

            def reset_wave(wave_width: int = width) -> list[dict[str, Any]]:
                return [
                    environments[slot].reset().to_dict() for slot in range(wave_width)
                ]

            states = [
                EnvironmentState.from_dict(value)
                for value in _rank_zero_call(is_main_process, reset_wave)
            ]
            contexts = [EpisodeContext(state.observation) for state in states]
            gamefiles = [str(state.metadata.get("gamefile", "")) for state in states]
            task_types = [
                Path(gamefile).parents[1].name.split("-", 1)[0]
                if len(Path(gamefile).parents) > 1
                else "unknown"
                for gamefile in gamefiles
            ]
            won_steps: list[int | None] = [None] * width
            active = [slot for slot, state in enumerate(states) if not state.done]

            if is_main_process:
                assert handle is not None
                for slot, episode, state in zip(
                    range(width), episode_ids, states, strict=True
                ):
                    handle.write(
                        json.dumps(
                            {
                                "event": "episode_start",
                                "split": split,
                                "episode": episode,
                                "slot": slot,
                                "gamefile": gamefiles[slot],
                                "initial_observation": state.observation,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            for step_index in range(1, max_steps + 1):
                if not active:
                    break
                actions_by_slot = {
                    slot: filter_actions(
                        states[slot].admissible_actions, excluded=[], maximum=0
                    )
                    for slot in active
                }
                active = [slot for slot in active if actions_by_slot[slot]]
                if not active:
                    break
                requests = [(contexts[slot], actions_by_slot[slot]) for slot in active]
                decisions = model.choose_actions(
                    requests,
                    max_new_tokens=max_action_tokens,
                    do_sample=False,
                    seed=seed + total_steps * 1009,
                )

                def execute_wave(
                    wave_slots: list[int] = active,
                    wave_decisions: list[Any] = decisions,
                ) -> list[dict[str, Any]]:
                    results = []
                    for slot, decision in zip(wave_slots, wave_decisions, strict=True):
                        executed = decision.action or "__invalid_action__"
                        next_state = environments[slot].step(executed)
                        results.append(
                            {
                                "slot": slot,
                                "executed_action": executed,
                                "invalid": decision.action is None,
                                "state": next_state.to_dict(),
                            }
                        )
                    return results

                results = _rank_zero_call(is_main_process, execute_wave)
                for result, decision in zip(results, decisions, strict=True):
                    slot = int(result["slot"])
                    next_state = EnvironmentState.from_dict(result["state"])
                    total_steps += 1
                    invalid_actions += int(result["invalid"])
                    contexts[slot].append(result["executed_action"], next_state.observation)
                    states[slot] = next_state
                    if next_state.won and won_steps[slot] is None:
                        won_steps[slot] = step_index
                    if is_main_process:
                        assert handle is not None
                        handle.write(
                            json.dumps(
                                {
                                    "event": "transition",
                                    "split": split,
                                    "episode": episode_ids[slot],
                                    "step": step_index,
                                    "raw_response": decision.raw_response,
                                    "action": decision.action,
                                    "executed_action": result["executed_action"],
                                    "invalid_action": result["invalid"],
                                    "observation": next_state.observation,
                                    "won": next_state.won,
                                    "score": next_state.score,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                active = [slot for slot in active if not states[slot].done]

            for slot, episode in enumerate(episode_ids):
                won_step = won_steps[slot]
                wins_at_report += int(won_step is not None and won_step <= report_step)
                wins_at_max += int(won_step is not None)
                if won_step is not None:
                    success_steps.append(won_step)
                task = task_results.setdefault(
                    task_types[slot], {"episodes": 0, "wins_at_report": 0, "wins_at_max": 0}
                )
                task["episodes"] += 1
                task["wins_at_report"] += int(
                    won_step is not None and won_step <= report_step
                )
                task["wins_at_max"] += int(won_step is not None)
                if is_main_process:
                    assert handle is not None
                    handle.write(
                        json.dumps(
                            {
                                "event": "episode_end",
                                "split": split,
                                "episode": episode,
                                "gamefile": gamefiles[slot],
                                "won": won_step is not None,
                                "success_step": won_step,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if handle is not None:
                handle.flush()
            completed += width
    finally:
        if handle is not None:
            handle.close()
        if is_main_process:
            for environment in environments:
                environment.close()

    summary: dict[str, Any] = {
        "episodes": float(episodes),
        f"sr_at_{report_step}": wins_at_report / episodes,
        f"successes_at_{report_step}": float(wins_at_report),
        f"sr_at_{max_steps}": wins_at_max / episodes,
        f"successes_at_{max_steps}": float(wins_at_max),
        "environment_steps": float(total_steps),
        "invalid_actions": float(invalid_actions),
        "invalid_action_rate": invalid_actions / total_steps if total_steps else 0.0,
        "mean_success_steps": (
            sum(success_steps) / len(success_steps) if success_steps else 0.0
        ),
        "by_task_type": {
            task_type: {
                "episodes": values["episodes"],
                f"sr_at_{report_step}": values["wins_at_report"] / values["episodes"],
                f"sr_at_{max_steps}": values["wins_at_max"] / values["episodes"],
            }
            for task_type, values in sorted(task_results.items())
        },
    }
    if is_main_process:
        summary_path = output.with_suffix(output.suffix + ".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary
