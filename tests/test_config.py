from pathlib import Path

import pytest

from experience_learning.config import AppConfig, load_config


def test_config_expands_environment_and_applies_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_MODEL", "judge-test")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "judge:\n  model: '${JUDGE_MODEL:-fallback}'\nexperiment:\n  max_environment_steps: 5\n",
        encoding="utf-8",
    )
    config = load_config(config_file, ["experiment.max_environment_steps=7"])
    assert config.judge.model == "judge-test"
    assert config.experiment.max_environment_steps == 7


def test_openai_compatible_judge_requires_explicit_model() -> None:
    config = AppConfig()
    config.judge.model = ""
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        config.validate()


def test_exact_match_smoke_judge_does_not_require_api_model() -> None:
    config = AppConfig()
    config.judge.provider = "exact_match"
    config.judge.model = ""
    config.validate()


def test_token_entropy_is_the_default_acquisition() -> None:
    assert AppConfig().acquisition.strategy == "token_entropy"


def test_checkpoint_retention_cannot_be_negative() -> None:
    config = AppConfig()
    config.training.max_periodic_checkpoints_to_keep = -1
    with pytest.raises(ValueError, match="checkpoint retention"):
        config.validate()


def test_parallel_training_counts_must_be_positive() -> None:
    config = AppConfig()
    config.experiment.parallel_environments = 0
    with pytest.raises(ValueError, match="parallel_environments"):
        config.validate()

    config.experiment.parallel_environments = 1
    config.training.update_batch_size = 0
    with pytest.raises(ValueError, match="update_batch_size"):
        config.validate()
