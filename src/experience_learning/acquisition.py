from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from functools import partial

from experience_learning.types import (
    AcquisitionDecision,
    ActionPrediction,
    JudgeResult,
    SemanticCluster,
    TokenEntropyPrediction,
)

CompareFn = Callable[[str, str], JudgeResult]


def cluster_semantic_predictions(
    predictions: Sequence[str],
    compare: CompareFn,
    *,
    uncertain_is_distinct: bool = True,
) -> tuple[SemanticCluster, ...]:
    """Cluster outcomes with an order-invariant pairwise equivalence graph."""
    values = list(predictions)
    parents = list(range(len(values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            result = compare(values[left_index], values[right_index])
            if result.equivalent is True or (
                result.equivalent is None and not uncertain_is_distinct
            ):
                union(left_index, right_index)

    grouped: dict[int, list[str]] = {}
    for index, value in enumerate(values):
        grouped.setdefault(find(index), []).append(value)
    clusters = []
    for members in grouped.values():
        ordered_members = tuple(sorted(members, key=lambda item: (item.strip().lower(), item)))
        clusters.append(
            SemanticCluster(
                representative=ordered_members[0],
                members=ordered_members,
            )
        )
    return tuple(sorted(clusters, key=lambda cluster: cluster.representative.lower()))


def semantic_entropy(clusters: Sequence[SemanticCluster], total: int) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    entropy = 0.0
    for cluster in clusters:
        probability = len(cluster.members) / total
        if probability > 0:
            entropy -= probability * math.log(probability)
    return entropy


def select_by_token_entropy(
    predictions_by_action: Mapping[str, TokenEntropyPrediction],
    *,
    rng: random.Random,
) -> AcquisitionDecision:
    """Select the action with maximum mean next-token entropy."""
    if not predictions_by_action:
        raise ValueError("cannot acquire from an empty action set")
    action_predictions = [
        ActionPrediction(
            action=action,
            samples=(prediction.observation,),
            clusters=(),
            entropy=prediction.mean_token_entropy,
            generated_tokens=prediction.generated_tokens,
            hit_token_limit=prediction.hit_token_limit,
        )
        for action, prediction in predictions_by_action.items()
    ]
    best_score = max(item.entropy for item in action_predictions)
    tied = [item for item in action_predictions if math.isclose(item.entropy, best_score)]
    chosen = rng.choice(tied)
    return AcquisitionDecision(
        action=chosen.action,
        score=chosen.entropy,
        predictions=tuple(action_predictions),
    )


def select_by_semantic_entropy(
    predictions_by_action: Mapping[str, Sequence[str]],
    compare_for_action: Callable[[str, str, str], JudgeResult],
    *,
    rng: random.Random,
    uncertain_is_distinct: bool = True,
    normalize: bool = False,
) -> AcquisitionDecision:
    if not predictions_by_action:
        raise ValueError("cannot acquire from an empty action set")
    action_predictions: list[ActionPrediction] = []
    for action, predictions in predictions_by_action.items():
        if not predictions:
            raise ValueError(f"action {action!r} has no predictions")
        clusters = cluster_semantic_predictions(
            predictions,
            partial(compare_for_action, action),
            uncertain_is_distinct=uncertain_is_distinct,
        )
        score = semantic_entropy(clusters, len(predictions))
        if normalize and len(predictions) > 1:
            score /= math.log(len(predictions))
        action_predictions.append(
            ActionPrediction(
                action=action,
                samples=tuple(predictions),
                clusters=clusters,
                entropy=score,
            )
        )
    best_score = max(item.entropy for item in action_predictions)
    tied = [item for item in action_predictions if math.isclose(item.entropy, best_score)]
    chosen = rng.choice(tied)
    return AcquisitionDecision(
        action=chosen.action,
        score=chosen.entropy,
        predictions=tuple(action_predictions),
    )


def modal_prediction(decision: AcquisitionDecision, action: str) -> str:
    prediction = next(item for item in decision.predictions if item.action == action)
    largest = min(
        prediction.clusters,
        key=lambda cluster: (-len(cluster.members), cluster.representative.lower()),
    )
    return largest.representative
