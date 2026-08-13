import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from experience_learning.offline import (
    collect_rwml_transitions,
    load_experiences,
    make_deterministic_subset,
    split_rwml_transitions,
    train_offline_wm_sft,
)
from experience_learning.types import (
    EnvironmentState,
    EpisodeContext,
    Experience,
    PolicyDecision,
)


class CollectionEnvironment:
    def __init__(self, slot: int, stride: int):
        self.slot = slot
        self.stride = stride
        self.episode = -1
        self.closed = False

    def reset(self) -> EnvironmentState:
        self.episode += 1
        game = self.slot + self.episode * self.stride
        return EnvironmentState(
            f"start {game}",
            ("act",),
            metadata={"gamefile": f"game-{game}"},
        )

    def step(self, action: str) -> EnvironmentState:
        assert action == "act"
        return EnvironmentState("result", (), done=True, metadata={"gamefile": "ignored"})

    def close(self) -> None:
        self.closed = True


class OfflineModel:
    def __init__(self):
        self.optimizer_step = 0
        self.saved: list[str] = []

    def choose_actions(
        self,
        requests: Sequence[tuple[EpisodeContext, tuple[str, ...]]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> list[PolicyDecision]:
        del max_new_tokens, seed
        assert do_sample
        return [PolicyDecision("<action>act</action>", actions[0]) for _, actions in requests]

    def learn(self, experiences: Sequence[Experience]) -> dict[str, float]:
        assert experiences
        self.optimizer_step += 1
        return {
            "loss": 0.5,
            "learning_rate": 2e-6,
            "optimizer_step": float(self.optimizer_step),
        }

    def score(self, experiences: Sequence[Experience]) -> dict[str, float]:
        return {"target_nll_sum": 2.0 * len(experiences), "target_tokens": len(experiences)}

    def save(self, output_dir: str) -> None:
        self.saved.append(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def test_collect_rwml_data_repeats_the_same_games_across_rollouts(tmp_path: Path) -> None:
    output = tmp_path / "raw.jsonl"
    summary = collect_rwml_transitions(
        model=OfflineModel(),
        environment_factory=CollectionEnvironment,
        is_main_process=True,
        tasks=3,
        rollouts_per_task=2,
        parallelism=2,
        max_steps=2,
        max_action_tokens=16,
        excluded_actions=[],
        seed=42,
        output_path=output,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["transitions"] == 6
    assert {record["gamefile"] for record in records if record["task_id"] == 2} == {
        "game-2"
    }
    assert len({record["record_id"] for record in records}) == 6
    assert output.with_suffix(".jsonl.manifest.json").exists()


def test_split_is_deterministic_and_removes_invalid_actions(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    records = []
    for index in range(10):
        records.append(
            {
                "record_id": f"r{index}",
                "context": {"initial_observation": "start", "history": []},
                "action": "act",
                "actual_observation": "result",
                "valid_action": index != 9,
            }
        )
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"

    manifest = split_rwml_transitions(
        input_path=source,
        train_path=train,
        validation_path=validation,
        validation_fraction=0.2,
        seed=42,
    )

    assert manifest["train_records"] == 7
    assert manifest["validation_records"] == 2
    assert len(load_experiences(train)) == 7
    with pytest.raises(FileExistsError):
        split_rwml_transitions(
            input_path=source,
            train_path=train,
            validation_path=validation,
            validation_fraction=0.2,
            seed=42,
        )


def test_offline_sft_trains_fixed_epochs_and_saves_final(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    record = {
        "context": {"initial_observation": "start", "history": []},
        "action": "act",
        "actual_observation": "result",
    }
    train.write_text("".join(json.dumps(record) + "\n" for _ in range(5)))
    validation.write_text(json.dumps(record) + "\n")
    model = OfflineModel()

    summary = train_offline_wm_sft(
        model=model,
        is_main_process=True,
        train_path=train,
        validation_path=validation,
        output_dir=tmp_path / "run",
        epochs=2,
        effective_batch_size=2,
        seed=42,
        evaluation_batch_size=1,
    )

    assert summary["optimizer_steps"] == 6
    assert summary["sample_exposures"] == 10
    assert summary["validation"]["target_token_nll"] == 2.0
    assert model.saved[-1].endswith("checkpoints/final")


def test_deterministic_subset_has_requested_size_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        "".join(json.dumps({"record_id": f"r{index}"}) + "\n" for index in range(5))
    )
    output = tmp_path / "matched.jsonl"

    manifest = make_deterministic_subset(
        input_path=source, output_path=output, records=3, seed=42
    )

    assert manifest["records"] == 3
    assert len(output.read_text().splitlines()) == 3
    assert output.with_suffix(".jsonl.manifest.json").exists()
