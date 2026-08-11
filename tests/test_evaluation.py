import json
from collections.abc import Sequence
from pathlib import Path

from experience_learning.config import AppConfig
from experience_learning.evaluation import ProbeTransition, evaluate_probes
from experience_learning.judge import ExactMatchJudge
from experience_learning.types import EpisodeContext, Experience


class EvaluationModel:
    def predict(
        self,
        requests: Sequence[tuple[EpisodeContext, str]],
        *,
        samples_per_request: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> list[list[str]]:
        del do_sample, seed
        return [["actual"] * samples_per_request for _ in requests]

    def score(self, experiences: Sequence[Experience]) -> dict[str, float]:
        return {"target_nll_sum": 2.0 * len(experiences), "target_tokens": 2.0 * len(experiences)}

    def learn(self, experiences: Sequence[Experience]) -> dict[str, float]:
        raise AssertionError("evaluation must not train")

    def save(self, output_dir: str) -> None:
        raise AssertionError("evaluation must not save")


def test_probe_evaluation_reports_semantic_accuracy_and_nll(tmp_path: Path) -> None:
    probe = ProbeTransition(
        split="eval_out_of_distribution",
        gamefile="game.tw-pddl",
        episode=0,
        step=0,
        context=EpisodeContext("initial"),
        action="open fridge",
        actual_observation="actual",
    )
    probes_path = tmp_path / "probes.jsonl"
    probes_path.write_text(json.dumps(probe.to_dict()) + "\n", encoding="utf-8")
    config = AppConfig()
    result = evaluate_probes(
        config,
        model=EvaluationModel(),
        judge=ExactMatchJudge(),
        is_main_process=True,
        probes_path=probes_path,
        output_path=tmp_path / "evaluation.jsonl",
    )
    assert result["semantic_accuracy_definite"] == 1.0
    assert result["judge_coverage"] == 1.0
    assert result["target_token_nll"] == 1.0
