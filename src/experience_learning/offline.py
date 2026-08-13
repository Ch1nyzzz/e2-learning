from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from experience_learning.environment import filter_actions
from experience_learning.interfaces import Environment, WorldModel
from experience_learning.types import EnvironmentState, EpisodeContext, Experience

T = TypeVar("T")


def _rank_zero_call(is_main_process: bool, operation: Callable[[], T]) -> T:
    from experience_learning.distributed import broadcast_object

    packet: dict[str, Any] | None = None
    if is_main_process:
        try:
            packet = {"ok": True, "value": operation()}
        except Exception as exc:
            packet = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    packet = broadcast_object(packet)
    if not packet["ok"]:
        raise RuntimeError(
            f"rank-0 offline operation failed: {packet['error_type']}: {packet['error']}"
        )
    return packet["value"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_rwml_transitions(
    *,
    model: WorldModel,
    environment_factory: Callable[[int, int], Environment],
    is_main_process: bool,
    tasks: int,
    rollouts_per_task: int,
    parallelism: int,
    max_steps: int,
    max_action_tokens: int,
    excluded_actions: list[str],
    seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Collect a fixed RWML-style on-policy transition corpus.

    Environments are recreated for every rollout pass. Their sorted, strided game partitions make
    task index i refer to the same ALFWorld game in every pass.
    """
    if min(tasks, rollouts_per_task, parallelism, max_steps, max_action_tokens) < 1:
        raise ValueError("collection counts and token limits must be positive")
    output = Path(output_path)
    _rank_zero_call(is_main_process, lambda: _reserve_fixed_dataset(output))
    handle = output.open("a", encoding="utf-8") if is_main_process else None

    expected_games: dict[int, str] = {}
    transitions = 0
    valid_transitions = 0
    invalid_actions = 0
    try:
        for rollout in range(rollouts_per_task):
            environments = (
                [environment_factory(slot, parallelism) for slot in range(parallelism)]
                if is_main_process
                else []
            )
            try:
                completed = 0
                while completed < tasks:
                    width = min(parallelism, tasks - completed)
                    task_ids = list(range(completed, completed + width))

                    def reset_wave(
                        wave_environments: Sequence[Environment] = environments,
                        wave_width: int = width,
                    ) -> list[dict[str, Any]]:
                        return [
                            wave_environments[slot].reset().to_dict()
                            for slot in range(wave_width)
                        ]

                    states = [
                        EnvironmentState.from_dict(value)
                        for value in _rank_zero_call(is_main_process, reset_wave)
                    ]
                    contexts = [EpisodeContext(state.observation) for state in states]
                    gamefiles = [str(state.metadata.get("gamefile", "")) for state in states]
                    for task_id, gamefile in zip(task_ids, gamefiles, strict=True):
                        previous = expected_games.setdefault(task_id, gamefile)
                        if previous != gamefile:
                            raise RuntimeError(
                                f"task {task_id} changed game across rollout passes: "
                                f"{previous!r} != {gamefile!r}"
                            )
                    active = [slot for slot, state in enumerate(states) if not state.done]

                    for step in range(max_steps):
                        if not active:
                            break
                        actions_by_slot = {
                            slot: filter_actions(
                                states[slot].admissible_actions,
                                excluded=excluded_actions,
                                maximum=0,
                            )
                            for slot in active
                        }
                        active = [slot for slot in active if actions_by_slot[slot]]
                        if not active:
                            break
                        decisions = model.choose_actions(
                            [(contexts[slot], actions_by_slot[slot]) for slot in active],
                            max_new_tokens=max_action_tokens,
                            do_sample=True,
                            seed=seed + rollout * 10_000_019 + transitions * 1009,
                        )

                        def execute_wave(
                            wave_active: list[int] = active,
                            wave_decisions: list[Any] = decisions,
                            wave_environments: Sequence[Environment] = environments,
                        ) -> list[dict[str, Any]]:
                            results = []
                            for slot, decision in zip(
                                wave_active, wave_decisions, strict=True
                            ):
                                executed = decision.action or "__invalid_action__"
                                results.append(
                                    {
                                        "slot": slot,
                                        "executed_action": executed,
                                        "valid_action": decision.action is not None,
                                        "state": wave_environments[slot]
                                        .step(executed)
                                        .to_dict(),
                                    }
                                )
                            return results

                        results = _rank_zero_call(is_main_process, execute_wave)
                        for result, decision in zip(results, decisions, strict=True):
                            slot = int(result["slot"])
                            next_state = EnvironmentState.from_dict(result["state"])
                            record = {
                                "record_id": (
                                    f"task_{task_ids[slot]:04d}/rollout_{rollout:02d}/step_{step:02d}"
                                ),
                                "task_id": task_ids[slot],
                                "rollout": rollout,
                                "step": step,
                                "gamefile": gamefiles[slot],
                                "context": contexts[slot].to_dict(),
                                "action": result["executed_action"],
                                "actual_observation": next_state.observation,
                                "predicted_observation": "",
                                "valid_action": bool(result["valid_action"]),
                                "raw_policy_response": decision.raw_response,
                                "won": next_state.won,
                                "done": next_state.done,
                            }
                            if is_main_process:
                                assert handle is not None
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            transitions += 1
                            valid_transitions += int(result["valid_action"])
                            invalid_actions += int(not result["valid_action"])
                            contexts[slot].append(
                                result["executed_action"], next_state.observation
                            )
                            states[slot] = next_state
                        active = [slot for slot in active if not states[slot].done]
                    completed += width
                    if handle is not None:
                        handle.flush()
            finally:
                if is_main_process:
                    for environment in environments:
                        environment.close()
    finally:
        if handle is not None:
            handle.close()

    summary = {
        "tasks": tasks,
        "rollouts_per_task": rollouts_per_task,
        "parallelism": parallelism,
        "max_steps": max_steps,
        "transitions": transitions,
        "valid_transitions": valid_transitions,
        "invalid_actions": invalid_actions,
        "output": str(output),
    }
    if is_main_process:
        summary["sha256"] = sha256_file(output)
        output.with_suffix(output.suffix + ".manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


def _reserve_fixed_dataset(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite fixed dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=False)


def split_rwml_transitions(
    *,
    input_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    validation_fraction: float,
    seed: int,
    exclude_invalid: bool = True,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    source = Path(input_path)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if exclude_invalid:
        records = [record for record in records if record.get("valid_action", True)]
    if len(records) < 2:
        raise ValueError("need at least two valid transitions to split")
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['record_id']}".encode()
        ).hexdigest(),
    )
    validation_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    validation_ids = {record["record_id"] for record in ordered[:validation_count]}
    train_records = [record for record in records if record["record_id"] not in validation_ids]
    validation_records = [record for record in records if record["record_id"] in validation_ids]

    outputs = ((Path(train_path), train_records), (Path(validation_path), validation_records))
    for path, values in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite fixed split: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
            encoding="utf-8",
        )
    manifest = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "exclude_invalid": exclude_invalid,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "filtering": "invalid-action removal only; RWML difficulty filtering not yet applied",
    }
    manifest_path = Path(train_path).with_suffix(".split-manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def make_deterministic_subset(
    *,
    input_path: str | Path,
    output_path: str | Path,
    records: int,
    seed: int,
) -> dict[str, Any]:
    if records < 1:
        raise ValueError("subset record count must be positive")
    source = Path(input_path)
    values = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if records > len(values):
        raise ValueError(f"requested {records} records from a dataset with only {len(values)}")
    selected = sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value['record_id']}".encode()
        ).hexdigest(),
    )[:records]
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fixed subset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in selected),
        encoding="utf-8",
    )
    manifest = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "records": records,
        "seed": seed,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "purpose": "environment-transition-count-matched baseline",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_experiences(path: str | Path) -> list[Experience]:
    return [
        Experience.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _evaluate_nll(
    model: WorldModel, experiences: Sequence[Experience], batch_size: int
) -> dict[str, float]:
    nll_sum = 0.0
    tokens = 0.0
    for start in range(0, len(experiences), batch_size):
        metrics = model.score(experiences[start : start + batch_size])
        nll_sum += metrics["target_nll_sum"]
        tokens += metrics["target_tokens"]
    return {"target_token_nll": nll_sum / tokens if tokens else 0.0, "target_tokens": tokens}


def train_offline_wm_sft(
    *,
    model: WorldModel,
    is_main_process: bool,
    train_path: str | Path,
    validation_path: str | Path | None,
    output_dir: str | Path,
    epochs: int,
    effective_batch_size: int,
    seed: int,
    evaluation_batch_size: int,
) -> dict[str, Any]:
    if min(epochs, effective_batch_size, evaluation_batch_size) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    train = load_experiences(train_path)
    validation = load_experiences(validation_path) if validation_path else []
    if not train:
        raise ValueError("offline SFT training dataset is empty")
    output = Path(output_dir)
    _rank_zero_call(is_main_process, lambda: _prepare_offline_output(output))
    rng = random.Random(seed)
    optimizer_steps = 0
    sample_exposures = 0
    training_target_tokens = 0.0
    last_metrics: dict[str, float] = {}
    validation_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for start in range(0, len(order), effective_batch_size):
            batch = [train[index] for index in order[start : start + effective_batch_size]]
            last_metrics = model.learn(batch)
            optimizer_steps = int(last_metrics["optimizer_step"])
            sample_exposures += len(batch)
            training_target_tokens += float(last_metrics.get("target_tokens", 0.0))
        if validation:
            validation_metrics = _evaluate_nll(model, validation, evaluation_batch_size)
        checkpoint = output / "checkpoints" / f"epoch_{epoch + 1:02d}"
        model.save(str(checkpoint))
        if is_main_process:
            with (output / "offline_sft_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "optimizer_step": optimizer_steps,
                            "train_loss": last_metrics.get("loss"),
                            "validation": validation_metrics,
                            "checkpoint": str(checkpoint),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    model.save(str(output / "checkpoints" / "final"))
    summary = {
        "train_records": len(train),
        "validation_records": len(validation),
        "epochs": epochs,
        "effective_batch_size": effective_batch_size,
        "updates_per_epoch": math.ceil(len(train) / effective_batch_size),
        "optimizer_steps": optimizer_steps,
        "sample_exposures": sample_exposures,
        "training_target_tokens": training_target_tokens,
        "last_train_metrics": last_metrics,
        "validation": validation_metrics,
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path) if validation_path else None,
    }
    if is_main_process:
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


def _prepare_offline_output(output: Path) -> None:
    if (output / "offline_sft_events.jsonl").exists() or (output / "checkpoints").exists():
        raise FileExistsError(f"refusing to overwrite existing offline run: {output}")
    output.mkdir(parents=True, exist_ok=True)
