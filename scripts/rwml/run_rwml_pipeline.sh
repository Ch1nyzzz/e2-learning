#!/usr/bin/env bash
# Full RWML pipeline (paper Appendix B.1, budget-matched merged10k corpus).
# Intended to start after the online training run releases GPU 0/1.
#
#   [1/6] filter-model WM SFT on the validation split (.venv, GPU 0/1)
#   [2/6] K=10 filter-model predictions over the train split (vLLM, GPU 0/1)
#   [3/6] embedding scoring + easy-subsample (tau_d=0.1, p=0.1, GPU 0)
#        + 0.6B-vs-8B tau calibration for the GRPO-time reward
#   [4/6] parquet conversion for veRL
#   [5/6] RWML GRPO (GPU 0/1) + colocated 0.6B embedding reward server (GPU 0)
#   [6/6] merge final checkpoint to HF + SR eval on both splits
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

train_jsonl=data/rwml_alfworld_qwen25_7b_train_merged10k.jsonl
val_jsonl=data/rwml_alfworld_qwen25_7b_validation_merged10k.jsonl
filter_model_dir=outputs/rwml_filter_model
predictions=data/rwml_filter_predictions_merged10k.jsonl
filtered=data/rwml_alfworld_qwen25_7b_train_merged10k_filtered.jsonl
grpo_dir=outputs/rwml_grpo_merged10k
sr_out=outputs/sr_eval_10k

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/6] filter-model WM SFT on validation split ==="
if [ ! -d "${filter_model_dir}/checkpoints/final" ]; then
  CUDA_VISIBLE_DEVICES=0,1 uv run --extra train accelerate launch \
    --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
    --main_process_port 29500 \
    -m experience_learning.cli train-wm-sft \
    --config configs/rwml_qwen25_7b.yaml \
    --train-data "${val_jsonl}" \
    --epochs 2 \
    --effective-batch-size 32 \
    --set experiment.output_dir="${filter_model_dir}"
fi

echo "=== [2/6] K=10 filter-model predictions (vLLM) ==="
if [ ! -f "${predictions}" ]; then
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src .venv-verl/bin/python \
    scripts/rwml/generate_filter_predictions.py \
    --train-data "${train_jsonl}" \
    --filter-model "${filter_model_dir}/checkpoints/final" \
    --output "${predictions}" \
    --k 10 --temperature 1.0 --tensor-parallel 2
fi

echo "=== [3/6] embedding scoring + easy subsample ==="
if [ ! -f "${filtered}" ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-verl/bin/python \
    scripts/rwml/score_and_filter.py \
    --train-data "${train_jsonl}" \
    --predictions "${predictions}" \
    --output "${filtered}" \
    --tau-d 0.1 --tau-easy 0.0 --subsample-p 0.1
fi

tau_json=data/rwml_tau_calibration.json
if [ ! -f "${tau_json}" ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-verl/bin/python \
    scripts/rwml/calibrate_tau.py \
    --train-data "${train_jsonl}" \
    --predictions "${predictions}" \
    --reference-tau 0.2 --sample 2000 \
    --output "${tau_json}"
fi
calibrated_tau=$(python3 -c "import json; print(json.load(open('${tau_json}'))['calibrated_tau'])")
echo "calibrated 0.6B tau_d: ${calibrated_tau}"

echo "=== [4/6] parquet conversion ==="
PYTHONPATH=src .venv-verl/bin/python scripts/rwml/prepare_grpo_parquet.py \
  --input "${filtered}" --output data/rwml_grpo/train.parquet
PYTHONPATH=src .venv-verl/bin/python scripts/rwml/prepare_grpo_parquet.py \
  --input "${val_jsonl}" --output data/rwml_grpo/val.parquet

echo "=== [5/6] RWML GRPO (waiting for all 4 GPUs) ==="
# GRPO on 2 GPUs is memory-marginal (vLLM rollout + FSDP actor + ref); wait
# until the Qwen3-8B training and its eval daemon release GPU 2/3, then train
# on all four cards.
while pgrep -f "alfworld_qwen3_8b" >/dev/null 2>&1 \
   || pgrep -f "chain_q3" >/dev/null 2>&1 \
   || pgrep -f "run_ckpt_eval_daemon_q" >/dev/null 2>&1; do
  sleep 300
done
sleep 60
RWML_TRAIN_GPUS=0,1,2,3 RWML_EMBED_GPU=0 \
  RWML_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B \
  RWML_TAU_D="${calibrated_tau}" \
  RWML_OUTPUT_DIR="${grpo_dir}" \
  bash scripts/rwml/run_rwml_grpo.sh

echo "=== [6/6] merge + SR eval ==="
last_step_dir=$(ls -d "${grpo_dir}"/global_step_* | sort -V | tail -1)
merged_hf="${grpo_dir}/hf_final"
.venv-verl/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${last_step_dir}/actor" \
  --target_dir "${merged_hf}"

run_sr() {
  local label="$1" split="$2"
  CUDA_VISIBLE_DEVICES=0,1 uv run --extra train --extra alfworld accelerate launch \
    --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
    --main_process_port 29500 \
    -m experience_learning.cli evaluate-sr \
    --config configs/alfworld_qwen25_7b.yaml \
    --set experiment.parallel_environments=64 \
    --set generation.micro_batch_size=32 \
    --split "${split}" \
    --max-steps 30 --report-step 20 --max-action-tokens 512 --seed 42 \
    --checkpoint "${merged_hf}" \
    --output "${sr_out}/${label}_${split}.jsonl"
}
run_sr rwml_grpo_merged10k eval_in_distribution
run_sr rwml_grpo_merged10k eval_out_of_distribution

echo "=== RWML_PIPELINE_DONE ==="
