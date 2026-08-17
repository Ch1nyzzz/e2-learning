"""Convert RWML transition jsonl into veRL RLHFDataset parquet.

Each row: chat-format prompt (paper Table A4 RWML prompt), rule-style
reward_model with the gold next observation as ground_truth, and record_id in
extra_info for traceability.

Usage (from repo root, .venv-verl):
    PYTHONPATH=src .venv-verl/bin/python scripts/rwml/prepare_grpo_parquet.py \
        --input data/rwml_alfworld_qwen25_7b_train_merged10k_filtered.jsonl \
        --output data/rwml_grpo/train.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experience_learning.offline import load_experiences
from experience_learning.prompts import rwml_grpo_prediction_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-source", default="alfworld_rwml")
    args = parser.parse_args()

    experiences = load_experiences(args.input)
    raw_records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = []
    for experience, record in zip(experiences, raw_records, strict=True):
        rows.append(
            {
                "data_source": args.data_source,
                "prompt": rwml_grpo_prediction_messages(experience.context, experience.action),
                "reward_model": {
                    "style": "rule",
                    "ground_truth": experience.actual_observation.strip(),
                },
                "extra_info": {"record_id": record.get("record_id", "")},
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
