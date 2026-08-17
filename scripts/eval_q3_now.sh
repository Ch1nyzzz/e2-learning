#!/usr/bin/env bash
# Evaluate Qwen3-8B base + every part1 checkpoint on both SR splits, GPU 2/3.
# (Training was stopped intentionally at cumulative step 7180; no final ckpt.)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_sr() {
  local label="$1" split="$2"
  shift 2
  local out="outputs/sr_eval_10k/${label}_${split}.jsonl"
  [ -f "${out}.summary.json" ] && return 0
  echo "=== EVAL ${label} ${split} ==="
  CUDA_VISIBLE_DEVICES=2,3 uv run --extra train --extra alfworld accelerate launch \
    --config_file configs/accelerate/fsdp_2xa100_80gb.yaml --main_process_port 29501 \
    -m experience_learning.cli evaluate-sr --config configs/alfworld_qwen3_8b.yaml \
    --set experiment.parallel_environments=64 --set generation.micro_batch_size=32 \
    --split "${split}" --max-steps 30 --report-step 20 --max-action-tokens 512 --seed 42 \
    --output "${out}" "$@"
}

for split in eval_in_distribution eval_out_of_distribution; do
  run_sr online_q3_base "${split}"
done
for ckpt in outputs/alfworld_qwen3_8b_active_10000_hf/checkpoints/*/; do
  name=$(basename "${ckpt}")
  [ -f "${ckpt}/controller_state.json" ] || continue
  for split in eval_in_distribution eval_out_of_distribution; do
    run_sr "online_q3_${name}" "${split}" --checkpoint "${ckpt%/}"
  done
done
echo "Q3_EVAL_ALL_DONE"
