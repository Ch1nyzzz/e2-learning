#!/usr/bin/env bash
# SR-eval daemon on GPU 2/3: evaluates every completed online checkpoint
# (periodic env_step_* and final) on both eval splits, oldest first.
# A checkpoint counts as complete once controller_state.json exists
# (it is written last during _checkpoint). Exits after 'final' is evaluated.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${EVAL_GPUS:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ckpt_root=outputs/alfworld_qwen25_7b_active_10000_hf/checkpoints
sr_out=outputs/sr_eval_10k
mkdir -p "${sr_out}"

run_sr() {
  local label="$1"
  local split="$2"
  local ckpt="$3"
  uv run --extra train --extra alfworld accelerate launch \
    --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
    --main_process_port "${EVAL_PORT:-29500}" \
    -m experience_learning.cli evaluate-sr \
    --config configs/alfworld_qwen25_7b.yaml \
    --set experiment.parallel_environments=64 \
    --set generation.micro_batch_size=32 \
    --split "${split}" \
    --max-steps 30 \
    --report-step 20 \
    --max-action-tokens 512 \
    --seed 42 \
    --checkpoint "${ckpt}" \
    --output "${sr_out}/online_${label}_${split}.jsonl"
}

while true; do
  did_work=0
  for ckpt in "${ckpt_root}"/*/; do
    [ -d "${ckpt}" ] || continue
    name=$(basename "${ckpt}")
    [ -f "${ckpt}/controller_state.json" ] || continue
    for split in eval_in_distribution eval_out_of_distribution; do
      out="${sr_out}/online_${name}_${split}.jsonl"
      [ -f "${out}.summary.json" ] && continue
      echo "=== EVAL ${name} ${split} ==="
      if run_sr "${name}" "${split}" "${ckpt%/}"; then
        did_work=1
      else
        echo "=== EVAL_FAILED ${name} ${split} ==="
        sleep 60
      fi
    done
  done
  if [ -f "${sr_out}/online_final_eval_out_of_distribution.jsonl.summary.json" ] \
     && [ -f "${sr_out}/online_final_eval_in_distribution.jsonl.summary.json" ]; then
    echo "=== ONLINE_CKPT_EVAL_DAEMON_DONE ==="
    break
  fi
  [ "${did_work}" -eq 0 ] && sleep 300
done
