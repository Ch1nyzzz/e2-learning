"""RWML stage-2a: K predictions per training triplet from the SFT filter model.

Writes one json line per triplet: {"record_id", "predictions": [...]}. Scoring
and subsampling happen in score_and_filter.py (separate process so vLLM memory
is fully released before the embedding model loads).

Usage (from repo root, .venv-verl):
    PYTHONPATH=src .venv-verl/bin/python scripts/rwml/generate_filter_predictions.py \
        --train-data data/rwml_alfworld_qwen25_7b_train_merged10k.jsonl \
        --filter-model outputs/rwml_filter_model/checkpoints/final \
        --output data/rwml_filter_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experience_learning.offline import load_experiences
from experience_learning.prompts import extract_next_state, rwml_wm_sft_prediction_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--filter-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")

    from vllm import LLM, SamplingParams

    experiences = load_experiences(args.train_data)
    raw_records = [
        json.loads(line)
        for line in Path(args.train_data).read_text(encoding="utf-8").splitlines()
        if line
    ]
    print(f"loaded {len(experiences)} triplets", flush=True)

    llm = LLM(
        model=args.filter_model,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        # PCIe H100s here lack P2P; vLLM's custom all-reduce fails with
        # "Cuda error 'invalid argument'", so force the NCCL path.
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    conversations = [
        rwml_wm_sft_prediction_messages(exp.context, exp.action) for exp in experiences
    ]
    generations = llm.chat(conversations, sampling, use_tqdm=True)

    with output.open("w", encoding="utf-8") as handle:
        for record, result in zip(raw_records, generations, strict=True):
            handle.write(
                json.dumps(
                    {
                        "record_id": record.get("record_id", ""),
                        "predictions": [
                            extract_next_state(out.text) or "" for out in result.outputs
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote predictions for {len(experiences)} triplets to {output}")


if __name__ == "__main__":
    main()
