#!/usr/bin/env bash
set -euo pipefail

train_path="${1:-data/rwml_alfworld_qwen25_7b_train_matched2000.jsonl}"
validation_path="${2:-data/rwml_alfworld_qwen25_7b_validation.jsonl}"
epochs="${RWML_SFT_EPOCHS:-1}"

uv run --extra train accelerate launch \
  --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
  -m experience_learning.cli train-wm-sft \
  --config configs/rwml_qwen25_7b.yaml \
  --train-data "${train_path}" \
  --validation-data "${validation_path}" \
  --epochs "${epochs}" \
  --effective-batch-size 32 \
  "${@:3}"
