import math
import random

from experience_learning.acquisition import select_by_semantic_entropy
from experience_learning.types import JudgeResult, Verdict


def _compare(_action: str, left: str, right: str) -> JudgeResult:
    return JudgeResult(Verdict.EQUIVALENT if left == right else Verdict.DIFFERENT)


def test_semantic_entropy_selects_disagreement() -> None:
    decision = select_by_semantic_entropy(
        {"certain": ["same"] * 4, "uncertain": ["yes", "yes", "no", "no"]},
        _compare,
        rng=random.Random(1),
        normalize=True,
    )
    assert decision.action == "uncertain"
    assert math.isclose(decision.score, 0.5)


def test_all_distinct_has_normalized_entropy_one() -> None:
    decision = select_by_semantic_entropy(
        {"a": ["one", "two", "three", "four"]},
        _compare,
        rng=random.Random(1),
        normalize=True,
    )
    assert math.isclose(decision.score, 1.0)


def test_prediction_order_does_not_change_entropy_or_clusters() -> None:
    first = select_by_semantic_entropy(
        {"a": ["yes", "no", "yes", "maybe"]},
        _compare,
        rng=random.Random(1),
        normalize=True,
    )
    second = select_by_semantic_entropy(
        {"a": ["maybe", "yes", "no", "yes"]},
        _compare,
        rng=random.Random(1),
        normalize=True,
    )
    assert first.score == second.score
    assert first.predictions[0].clusters == second.predictions[0].clusters
