from collections.abc import Sequence
from pathlib import Path

from experience_learning.prompts import extract_policy_action
from experience_learning.success_evaluation import evaluate_success_rate
from experience_learning.types import (
    EnvironmentState,
    EpisodeContext,
    PolicyDecision,
)


class SuccessEnvironment:
    def __init__(self, win_step: int):
        self.win_step = win_step
        self.steps = 0
        self.closed = False

    def reset(self) -> EnvironmentState:
        self.steps = 0
        return EnvironmentState(
            "goal and room", ("go to fridge 1",), metadata={"gamefile": "game"}
        )

    def step(self, action: str) -> EnvironmentState:
        assert action == "go to fridge 1"
        self.steps += 1
        won = self.steps >= self.win_step
        return EnvironmentState(
            f"step {self.steps}",
            ("go to fridge 1",),
            done=won,
            won=won,
            score=float(won),
            metadata={"gamefile": "game"},
        )

    def close(self) -> None:
        self.closed = True


class SuccessModel:
    def choose_actions(
        self,
        requests: Sequence[tuple[EpisodeContext, tuple[str, ...]]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> list[PolicyDecision]:
        del max_new_tokens, seed
        assert do_sample is False
        return [
            PolicyDecision(
                raw_response="<think>move</think><action>go to fridge 1</action>",
                action=actions[0],
            )
            for _, actions in requests
        ]


def test_policy_action_parser_requires_an_admissible_exact_match() -> None:
    actions = ("go to fridge 1", "open fridge 1")
    assert (
        extract_policy_action("<action>Go To Fridge 1.</action>", actions)
        == "go to fridge 1"
    )
    assert extract_policy_action("<action>go to cabinet 1</action>", actions) is None
    assert extract_policy_action("reasoning only", actions) is None


def test_success_evaluation_reports_both_step_budgets(tmp_path: Path) -> None:
    environments = [SuccessEnvironment(1), SuccessEnvironment(3)]
    result = evaluate_success_rate(
        model=SuccessModel(),
        environments=environments,
        is_main_process=True,
        split="eval_out_of_distribution",
        episodes=2,
        parallelism=2,
        max_steps=3,
        report_step=2,
        max_action_tokens=64,
        seed=42,
        output_path=tmp_path / "sr.jsonl",
    )
    assert result["sr_at_2"] == 0.5
    assert result["sr_at_3"] == 1.0
    assert result["environment_steps"] == 4.0
    assert result["invalid_action_rate"] == 0.0
    assert all(environment.closed for environment in environments)
