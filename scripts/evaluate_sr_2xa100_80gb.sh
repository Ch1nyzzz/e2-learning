#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/alfworld_qwen3_8b.yaml}"
checkpoint_path="${2:-outputs/alfworld_qwen3_8b_parallel8/checkpoints/final}"
output_root="${3:-outputs/alfworld_qwen3_8b_parallel8/sr_eval}"

mkdir -p "${output_root}"

run_evaluation() {
  local label="$1"
  local split="$2"
  shift 2
  uv run --extra train --extra alfworld accelerate launch \
    --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
    -m experience_learning.cli evaluate-sr \
    --config "${config_path}" \
    --set experiment.parallel_environments=64 \
    --set generation.micro_batch_size=32 \
    --split "${split}" \
    --max-steps 30 \
    --report-step 20 \
    --max-action-tokens 512 \
    --seed 42 \
    --output "${output_root}/${label}_${split}.jsonl" \
    "$@"
}

run_evaluation base eval_in_distribution
run_evaluation base eval_out_of_distribution
run_evaluation final eval_in_distribution --checkpoint "${checkpoint_path}"
run_evaluation final eval_out_of_distribution --checkpoint "${checkpoint_path}"
