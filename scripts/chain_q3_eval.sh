#!/usr/bin/env bash
# Qwen3-8B post-training evaluation chain (v2, warm-restart aware).
# Waits for the part2 continuation run to finish, then runs the base eval and
# evaluates every checkpoint from part1 (labels online_q3_*) and part2
# (labels online_q3p2_*, steps offset by +7180 in analysis) once on GPU 2/3.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while pgrep -f "experience_learning.cli train --config configs/alfworld_qwen3_8b" >/dev/null 2>&1; do
  sleep 300
done
sleep 60
if [ ! -d outputs/alfworld_qwen3_8b_active_10000_hf_part2/checkpoints/final ]; then
  echo "QWEN3_PART2_EXITED_WITHOUT_FINAL"
  ls outputs/alfworld_qwen3_8b_active_10000_hf_part2/checkpoints/ 2>/dev/null | tail -3
  exit 1
fi

run_sr() {
  local label="$1" split="$2"
  shift 2
  local out="outputs/sr_eval_10k/${label}_${split}.jsonl"
  [ -f "${out}.summary.json" ] && return 0
  CUDA_VISIBLE_DEVICES=2,3 uv run --extra train --extra alfworld accelerate launch \
    --config_file configs/accelerate/fsdp_2xa100_80gb.yaml --main_process_port 29501 \
    -m experience_learning.cli evaluate-sr --config configs/alfworld_qwen3_8b.yaml \
    --set experiment.parallel_environments=64 --set generation.micro_batch_size=32 \
    --split "${split}" --max-steps 30 --report-step 20 --max-action-tokens 512 --seed 42 \
    --output "${out}" "$@"
}

echo "QWEN3_PART2_DONE — base eval first"
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
for ckpt in outputs/alfworld_qwen3_8b_active_10000_hf_part2/checkpoints/*/; do
  name=$(basename "${ckpt}")
  [ -f "${ckpt}/controller_state.json" ] || continue
  for split in eval_in_distribution eval_out_of_distribution; do
    run_sr "online_q3p2_${name}" "${split}" --checkpoint "${ckpt%/}"
  done
done
echo "QWEN3_EVAL_DONE"
