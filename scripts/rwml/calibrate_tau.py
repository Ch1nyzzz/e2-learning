"""Calibrate the small embedding model's tau_d against Qwen3-Embedding-8B.

GRPO-time rewards run on Qwen3-Embedding-0.6B (the 8B does not fit next to the
trainer). The paper's tau_d=0.2 is defined under the 8B embedding, so we map it
across models by quantile matching on real filter-model predictions: find the
0.6B distance threshold that awards reward at the same rate the 8B does at 0.2.

Usage (from repo root, .venv-verl, one free GPU):
    PYTHONPATH=src .venv-verl/bin/python scripts/rwml/calibrate_tau.py \
        --train-data data/rwml_alfworld_qwen25_7b_train_merged10k.jsonl \
        --predictions data/rwml_filter_predictions_merged10k.jsonl \
        --reference-tau 0.2 --sample 2000 --output data/rwml_tau_calibration.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_scorer import EmbeddingScorer

from experience_learning.offline import load_experiences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reference-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--small-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--reference-tau", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiences = load_experiences(args.train_data)
    predictions = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line
    ]
    pairs = []
    for experience, row in zip(experiences, predictions, strict=True):
        gold = experience.actual_observation.strip()
        for pred in row["predictions"]:
            if pred and pred != gold:
                pairs.append((pred, gold))
    rng = random.Random(args.seed)
    if len(pairs) > args.sample:
        pairs = rng.sample(pairs, args.sample)
    preds = [pair[0] for pair in pairs]
    golds = [pair[1] for pair in pairs]
    print(f"calibrating on {len(pairs)} prediction/gold pairs", flush=True)

    reference = EmbeddingScorer(args.reference_model, device=args.device)
    ref_distances = reference.cosine_distances(preds, golds)
    del reference

    import torch

    torch.cuda.empty_cache()
    small = EmbeddingScorer(args.small_model, device=args.device)
    small_distances = small.cosine_distances(preds, golds)

    positive_rate = sum(1 for d in ref_distances if d < args.reference_tau) / len(ref_distances)
    ordered = sorted(small_distances)
    index = max(0, min(len(ordered) - 1, round(positive_rate * len(ordered)) - 1))
    calibrated_tau = ordered[index] if positive_rate > 0 else 0.0

    # Agreement of the calibrated small-model decision with the 8B decision.
    agree = sum(
        1
        for ref_d, small_d in zip(ref_distances, small_distances, strict=True)
        if (ref_d < args.reference_tau) == (small_d < calibrated_tau)
    ) / len(ref_distances)

    result = {
        "reference_model": args.reference_model,
        "small_model": args.small_model,
        "reference_tau": args.reference_tau,
        "reference_positive_rate": positive_rate,
        "calibrated_tau": calibrated_tau,
        "decision_agreement": agree,
        "pairs": len(pairs),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
