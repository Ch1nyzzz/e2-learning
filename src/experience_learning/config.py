from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"
    max_context_tokens: int = 4096
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    distributed_timeout_seconds: int = 21600


@dataclass
class GenerationConfig:
    samples_per_action: int = 4
    micro_batch_size: int = 1
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class JudgeConfig:
    provider: str = "openai_compatible"
    model: str = "${JUDGE_MODEL:-}"
    api_key_env: str = "JUDGE_API_KEY"
    base_url: str | None = "${JUDGE_BASE_URL:-}"
    timeout_seconds: float = 60.0
    max_retries: int = 3
    cache_path: str = "outputs/judge_cache.sqlite3"


@dataclass
class EnvironmentConfig:
    config_path: str = "configs/alfworld/base_config.yaml"
    split: str = "train"
    max_steps_per_episode: int = 50
    excluded_actions: list[str] = field(default_factory=list)
    max_candidate_actions: int = 0


@dataclass
class AcquisitionConfig:
    strategy: str = "token_entropy"
    uncertain_comparisons_are_distinct: bool = True
    normalize_entropy: bool = True


@dataclass
class TrainingConfig:
    learning_rate: float = 1.0e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    updates_per_mistake: int = 1
    update_gate: str = "mistake_only"
    warmup_updates: int = 20
    checkpoint_every_environment_steps: int = 100
    resume_from: str | None = None


@dataclass
class ExperimentConfig:
    seed: int = 42
    output_dir: str = "outputs/alfworld_qwen3_8b"
    max_episodes: int = 100
    max_environment_steps: int = 2000
    stop_error_rate_threshold: float | None = None
    stop_error_rate_window: int = 200


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    max_transitions: int = 0


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.generation.samples_per_action < 2:
            raise ValueError("samples_per_action must be at least 2")
        if self.generation.micro_batch_size < 1:
            raise ValueError("generation micro_batch_size must be positive")
        if self.model.distributed_timeout_seconds < 600:
            raise ValueError("distributed_timeout_seconds must be at least 600")
        if self.environment.split not in {
            "train",
            "eval_in_distribution",
            "eval_out_of_distribution",
        }:
            raise ValueError(f"unsupported ALFWorld split: {self.environment.split}")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training.updates_per_mistake < 1:
            raise ValueError("updates_per_mistake must be positive")
        if self.training.checkpoint_every_environment_steps < 0:
            raise ValueError("checkpoint interval cannot be negative")
        if self.training.update_gate not in {"mistake_only", "all_transitions"}:
            raise ValueError("update_gate must be mistake_only or all_transitions")
        if self.acquisition.strategy not in {"token_entropy", "semantic_entropy", "random"}:
            raise ValueError(
                "acquisition strategy must be token_entropy, semantic_entropy, or random"
            )
        if self.judge.max_retries < 1:
            raise ValueError("judge max_retries must be at least 1")
        if self.judge.provider == "openai_compatible" and not self.judge.model.strip():
            raise ValueError("JUDGE_MODEL must name the semantic judge model")
        if self.experiment.max_environment_steps < 1:
            raise ValueError("max_environment_steps must be positive")
        if self.evaluation.batch_size < 1:
            raise ValueError("evaluation batch size must be positive")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"override must be key=value, got: {raw}")
    key, value = raw.split("=", 1)
    return key.split("."), yaml.safe_load(value)


def _apply_override(data: dict[str, Any], raw: str) -> None:
    keys, value = _parse_override(raw)
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override through non-mapping key: {key}")
        cursor = child
    cursor[keys[-1]] = value


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"config section {name!r} must be a mapping")
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> AppConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("top-level configuration must be a mapping")
    for override in overrides or []:
        _apply_override(raw, override)
    raw = _expand_env(raw)
    cfg = AppConfig(
        model=ModelConfig(**_section(raw, "model")),
        generation=GenerationConfig(**_section(raw, "generation")),
        judge=JudgeConfig(**_section(raw, "judge")),
        environment=EnvironmentConfig(**_section(raw, "environment")),
        acquisition=AcquisitionConfig(**_section(raw, "acquisition")),
        training=TrainingConfig(**_section(raw, "training")),
        experiment=ExperimentConfig(**_section(raw, "experiment")),
        evaluation=EvaluationConfig(**_section(raw, "evaluation")),
    )
    cfg.validate()
    return cfg
