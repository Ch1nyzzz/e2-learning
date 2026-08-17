#!/usr/bin/env bash
# Non-thinking SR eval worker on GPU 0/1, ascending order: base first, then
# checkpoints from earliest to latest so the nothink curve fills in
# chronologically. Dedup against the GPU 2/3 nothink worker (which is also
# ascending) relies on the output-jsonl existence check in run_sr.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_sr() {
  local label="$1" split="$2"
  shift 2
  local out="outputs/sr_eval_10k/${label}_${split}.jsonl"
  [ -f "${out}" ] && return 0
  echo "=== EVAL-NOTHINK01 ${label} ${split} ==="
  CUDA_VISIBLE_DEVICES=0,1 uv run --extra train --extra alfworld accelerate launch \
    --config_file configs/accelerate/fsdp_2xa100_80gb.yaml --main_process_port 29500 \
    -m experience_learning.cli evaluate-sr --config configs/alfworld_qwen3_8b.yaml \
    --set experiment.parallel_environments=64 --set generation.micro_batch_size=32 \
    --set generation.policy_enable_thinking=false \
    --split "${split}" --max-steps 30 --report-step 20 --seed 42 \
    --output "${out}" "$@"
}

for split in eval_in_distribution eval_out_of_distribution; do
  run_sr online_q3_nothink_base "${split}"
done
for ckpt in outputs/alfworld_qwen3_8b_active_10000_hf/checkpoints/*/; do
  name=$(basename "${ckpt}")
  [ -f "${ckpt}/controller_state.json" ] || continue
  for split in eval_in_distribution eval_out_of_distribution; do
    run_sr "online_q3_nothink_${name}" "${split}" --checkpoint "${ckpt%/}"
  done
done
echo "Q3_NOTHINK_EVAL01_ASC_DONE"
