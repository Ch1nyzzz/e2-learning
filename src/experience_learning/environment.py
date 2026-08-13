from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experience_learning.config import EnvironmentConfig
from experience_learning.types import EnvironmentState


class ALFWorldTextEnvironment:
    """Batch-size-one adapter over ALFWorld's TextWorld API."""

    def __init__(
        self,
        config: EnvironmentConfig,
        seed: int,
        *,
        game_offset: int = 0,
        game_stride: int = 1,
    ):
        try:
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise RuntimeError("install the project with the 'alfworld' extra") from exc
        with Path(config.config_path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["env"]["type"] = "AlfredTWEnv"
        raw["general"]["training_method"] = "dqn"
        raw["env"]["domain_randomization"] = False
        raw["rl"]["training"]["max_nb_steps_per_episode"] = config.max_steps_per_episode
        random.seed(seed)
        np.random.seed(seed)
        builder = get_environment("AlfredTWEnv")(raw, train_eval=config.split)
        builder.game_files = sorted(builder.game_files)
        if game_stride < 1 or not 0 <= game_offset < game_stride:
            raise ValueError("game partition requires 0 <= offset < stride")
        builder.game_files = builder.game_files[game_offset::game_stride]
        builder.num_games = len(builder.game_files)
        if builder.num_games == 0:
            raise RuntimeError(f"ALFWorld game partition {game_offset}/{game_stride} is empty")
        self._env = builder.init_env(batch_size=1)
        self._env.seed(seed)
        self._episode_steps = 0
        self._max_steps = config.max_steps_per_episode

    @staticmethod
    def _value(info: dict[str, Any], key: str, default: Any) -> Any:
        value = info.get(key, [default])
        if isinstance(value, (list, tuple, np.ndarray)):
            return value[0]
        return value

    @classmethod
    def _state(
        cls,
        observations: list[str],
        scores: Any,
        dones: Any,
        info: dict[str, Any],
    ) -> EnvironmentState:
        actions = tuple(str(item) for item in cls._value(info, "admissible_commands", []))
        return EnvironmentState(
            observation=str(observations[0]),
            admissible_actions=actions,
            done=bool(dones[0]) if dones is not None else False,
            won=bool(cls._value(info, "won", False)),
            score=float(scores[0]) if scores is not None else 0.0,
            metadata={"gamefile": cls._value(info, "extra.gamefile", "")},
        )

    def reset(self) -> EnvironmentState:
        observations, info = self._env.reset()
        self._episode_steps = 0
        return self._state(observations, None, None, info)

    def step(self, action: str) -> EnvironmentState:
        observations, scores, dones, info = self._env.step([action])
        self._episode_steps += 1
        state = self._state(observations, scores, dones, info)
        if self._episode_steps >= self._max_steps and not state.done:
            return EnvironmentState(
                observation=state.observation,
                admissible_actions=state.admissible_actions,
                done=True,
                won=state.won,
                score=state.score,
                metadata={**state.metadata, "locally_truncated": True},
            )
        return state

    def close(self) -> None:
        self._env.close()


def filter_actions(
    actions: tuple[str, ...], *, excluded: list[str], maximum: int
) -> tuple[str, ...]:
    excluded_values = {item.strip().lower() for item in excluded}
    filtered = tuple(
        sorted(
            {a for a in actions if a.strip().lower() not in excluded_values},
            key=lambda value: value.lower(),
        )
    )
    if not filtered:
        filtered = tuple(sorted(set(actions), key=lambda value: value.lower()))
    return filtered[:maximum] if maximum > 0 else filtered
