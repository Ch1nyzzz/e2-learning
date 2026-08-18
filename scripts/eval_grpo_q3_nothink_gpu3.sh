#!/usr/bin/env bash
# ID/OOD SR eval for grpo_q3_nothink_base checkpoints on GPU 3 only, while
# GPUs 0-2 keep training. Evaluates the rotation-safe HF snapshots
# (step_125_hf, step_55_hf) copied out of the verl checkpoint dir.
# Protocol: T=0 greedy (do_sample=False), full splits, 30 steps, SR@20+SR@30.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT_ROOT=/helios-storage/helios4-data/yuhan/e2l_policy_grpo/grpo_q3_nothink_base

run_sr() {
  local label="$1" split="$2"
  shift 2
  local out="outputs/sr_eval_grpo/${label}_${split}.jsonl"
  [ -f "${out}.summary.json" ] && return 0
  echo "=== EVAL-GRPO-GPU3 ${label} ${split} ==="
  CUDA_VISIBLE_DEVICES=3 uv run --extra train --extra alfworld accelerate launch \
    --config_file configs/accelerate/fsdp_1xh100_80gb.yaml --main_process_port 29503 \
    -m experience_learning.cli evaluate-sr --config configs/alfworld_qwen3_8b.yaml \
    --set experiment.parallel_environments=64 --set generation.micro_batch_size=32 \
    --set generation.policy_enable_thinking=false \
    --split "${split}" --max-steps 30 --report-step 20 --seed 42 \
    --output "${out}" "$@"
}

mkdir -p outputs/sr_eval_grpo
for ckpt in step_125_hf; do
  label="grpo_q3_nothink_base_${ckpt%_hf}"
  for split in eval_in_distribution eval_out_of_distribution; do
    run_sr "${label}" "${split}" --checkpoint "${CKPT_ROOT}/${ckpt}"
  done
done
echo "GRPO_Q3_NOTHINK_GPU3_EVAL_DONE"
