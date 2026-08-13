#!/usr/bin/env bash
set -euo pipefail

raw_path="${1:-data/rwml_alfworld_qwen25_7b_raw.jsonl}"
train_path="${2:-data/rwml_alfworld_qwen25_7b_train.jsonl}"
validation_path="${3:-data/rwml_alfworld_qwen25_7b_validation.jsonl}"
matched_path="${4:-data/rwml_alfworld_qwen25_7b_train_matched2000.jsonl}"

uv run --extra train --extra alfworld accelerate launch \
  --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
  -m experience_learning.cli collect-rwml-data \
  --config configs/rwml_qwen25_7b.yaml \
  --tasks 256 \
  --rollouts-per-task 1 \
  --parallel-environments 8 \
  --max-steps 30 \
  --max-action-tokens 512 \
  --output "${raw_path}"

uv run --extra alfworld experience-learning split-rwml-data \
  --config configs/rwml_qwen25_7b.yaml \
  --input "${raw_path}" \
  --train-output "${train_path}" \
  --validation-output "${validation_path}" \
  --validation-fraction 0.1

uv run --extra alfworld experience-learning subset-rwml-data \
  --config configs/rwml_qwen25_7b.yaml \
  --input "${train_path}" \
  --output "${matched_path}" \
  --records 2000
