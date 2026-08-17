"""RWML stage-2b: score filter-model predictions and subsample easy triplets.

Reads the predictions from generate_filter_predictions.py, computes the binary
embedding reward per attempt (tau_d=0.1 for data construction), flags a triplet
easy when its mean reward exceeds tau_easy (0.0 -> any hit within K), keeps easy
triplets with probability p=0.1, and writes the filtered training jsonl.

Usage (from repo root, .venv-verl):
    PYTHONPATH=src .venv-verl/bin/python scripts/rwml/score_and_filter.py \
        --train-data data/rwml_alfworld_qwen25_7b_train_merged10k.jsonl \
        --predictions data/rwml_filter_predictions.jsonl \
        --output data/rwml_alfworld_qwen25_7b_train_merged10k_filtered.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_scorer import DEFAULT_MODEL, EmbeddingScorer

from experience_learning.offline import load_experiences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tau-d", type=float, default=0.1)
    parser.add_argument("--tau-easy", type=float, default=0.0)
    parser.add_argument("--subsample-p", type=float, default=0.1)
    parser.add_argument("--embed-model", default=DEFAULT_MODEL)
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")

    experiences = load_experiences(args.train_data)
    raw_lines = [
        line for line in Path(args.train_data).read_text(encoding="utf-8").splitlines() if line
    ]
    predictions = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not (len(experiences) == len(raw_lines) == len(predictions)):
        raise SystemExit("train data and predictions are misaligned")

    scorer = EmbeddingScorer(args.embed_model, device=args.embed_device)

    # Score all non-exact, non-empty predictions in one batched pass.
    pending_pairs: list[tuple[int, int]] = []
    pending_pred: list[str] = []
    pending_gold: list[str] = []
    for index, (experience, row) in enumerate(zip(experiences, predictions, strict=True)):
        gold = experience.actual_observation.strip()
        for attempt, pred in enumerate(row["predictions"]):
            if pred and pred != gold:
                pending_pairs.append((index, attempt))
                pending_pred.append(pred)
                pending_gold.append(gold)
    print(f"embedding-scoring {len(pending_pred)} prediction/gold pairs", flush=True)
    distances = scorer.cosine_distances(pending_pred, pending_gold)
    distance_map = {pair: dist for pair, dist in zip(pending_pairs, distances, strict=True)}

    rng = random.Random(args.seed)
    kept, easy_count, mean_rewards = [], 0, []
    for index, (experience, row) in enumerate(zip(experiences, predictions, strict=True)):
        gold = experience.actual_observation.strip()
        rewards = []
        for attempt, pred in enumerate(row["predictions"]):
            if not pred:
                rewards.append(0.0)
            elif pred == gold:
                rewards.append(1.0)
            else:
                rewards.append(1.0 if distance_map[(index, attempt)] < args.tau_d else 0.0)
        mean_reward = sum(rewards) / len(rewards)
        mean_rewards.append(mean_reward)
        if mean_reward > args.tau_easy:
            easy_count += 1
            if rng.random() >= args.subsample_p:
                continue
        kept.append(raw_lines[index])

    output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    manifest = {
        "source": args.train_data,
        "predictions": args.predictions,
        "tau_d": args.tau_d,
        "tau_easy": args.tau_easy,
        "subsample_p": args.subsample_p,
        "total": len(experiences),
        "easy": easy_count,
        "easy_fraction": easy_count / len(experiences),
        "kept": len(kept),
        "mean_reward_overall": sum(mean_rewards) / len(mean_rewards),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    Path(str(output) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
