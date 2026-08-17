"""Sweep the data-construction tau_d to hit the paper's ~30% easy operating point.

Table A2: tau_d is a heuristic "set such that 'too easy' samples correspond to
~30% of the original dataset". Our corpus at tau_d=0.1 gives 88% easy, so we
sweep tau_d over the achievable range. A sample is easy when it has >=1 hit in
K attempts; exact string matches hit at any tau_d, which sets a floor on the
reachable easy fraction.

Usage (from repo root, .venv-verl, one free GPU):
    PYTHONPATH=src .venv-verl/bin/python scripts/rwml/sweep_filter_tau.py \
        --train-data data/rwml_alfworld_qwen25_7b_train_merged10k.jsonl \
        --predictions data/rwml_filter_predictions_merged10k.jsonl \
        --distances-out data/rwml_filter_distances.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_scorer import DEFAULT_MODEL, EmbeddingScorer

from experience_learning.offline import load_experiences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--distances-out", required=True)
    parser.add_argument("--embed-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    experiences = load_experiences(args.train_data)
    predictions = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line
    ]

    cache_path = Path(args.distances_out)
    if cache_path.exists():
        per_sample = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        scorer = EmbeddingScorer(args.embed_model, device=args.device)
        pending_idx: list[tuple[int, int]] = []
        pending_pred: list[str] = []
        pending_gold: list[str] = []
        for index, (experience, row) in enumerate(zip(experiences, predictions, strict=True)):
            gold = experience.actual_observation.strip()
            for attempt, pred in enumerate(row["predictions"]):
                if pred and pred != gold:
                    pending_idx.append((index, attempt))
                    pending_pred.append(pred)
                    pending_gold.append(gold)
        distances = scorer.cosine_distances(pending_pred, pending_gold)
        # per sample: list of attempt outcomes; exact match -> distance 0.0,
        # empty prediction -> None (never a hit)
        per_sample = []
        for _index, (experience, row) in enumerate(zip(experiences, predictions, strict=True)):
            gold = experience.actual_observation.strip()
            per_sample.append(
                [0.0 if pred == gold and pred else None for pred in row["predictions"]]
            )
        for (index, attempt), dist in zip(pending_idx, distances, strict=True):
            per_sample[index][attempt] = dist
        cache_path.write_text(json.dumps(per_sample), encoding="utf-8")

    exact_floor = sum(
        1 for sample in per_sample if any(d == 0.0 for d in sample if d is not None)
    ) / len(per_sample)
    print(f"samples: {len(per_sample)}  exact-hit floor: {exact_floor:.3f}")
    print(f"{'tau_d':>8} {'easy_frac':>10} {'kept_approx':>12}")
    for tau in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]:
        easy = sum(
            1
            for sample in per_sample
            if any(d is not None and d < tau for d in sample)
        ) / len(per_sample)
        kept = round(len(per_sample) * ((1 - easy) + easy * 0.1))
        print(f"{tau:>8} {easy:>10.3f} {kept:>12}")


if __name__ == "__main__":
    main()
