#!/usr/bin/env bash
# RWML GRPO training (paper Appendix B.1 ALFWorld settings, scaled to 2xH100):
# GRPO, lr 1e-6, train batch 32, group size 8, 2 epochs, binary embedding reward
# with tau_d=0.2 served by embed_server.py on a third GPU.
#
# Env overrides: RWML_TRAIN_PARQUET, RWML_VAL_PARQUET, RWML_MODEL_PATH,
# RWML_OUTPUT_DIR, RWML_TRAIN_GPUS (default "0,1"), RWML_EMBED_GPU (default "0",
# colocated with the trainer -- pair with the small embed model), RWML_EMBED_MODEL
# (default Qwen3-Embedding-0.6B; tau must come from calibrate_tau.py), RWML_EPOCHS,
# RWML_TAU_D.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Interpreter of the RWML veRL env (container images set this to the baked-in
# venv; locally it is the repo's .venv-verl).
VERL_VENV_PYTHON=${VERL_VENV_PYTHON:-.venv-verl/bin/python}

train_parquet="${RWML_TRAIN_PARQUET:-data/rwml_grpo/train.parquet}"
val_parquet="${RWML_VAL_PARQUET:-data/rwml_grpo/val.parquet}"
model_path="${RWML_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
output_dir="${RWML_OUTPUT_DIR:-outputs/rwml_grpo_merged10k}"
train_gpus="${RWML_TRAIN_GPUS:-0,1}"
embed_gpu="${RWML_EMBED_GPU:-0}"
embed_model="${RWML_EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
epochs="${RWML_EPOCHS:-2}"
export RWML_TAU_D="${RWML_TAU_D:-0.2}"
export RWML_EMBED_URL="http://127.0.0.1:8901"
# Shared hosts may have another user's long-lived Ray cluster registered under
# /tmp/ray; force an isolated local instance so we neither join nor clobber it.
export RAY_ADDRESS=local
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_rwml_grpo_$USER}"

n_gpus=$(awk -F',' '{print NF}' <<< "${train_gpus}")

echo "=== starting embedding reward server (${embed_model}) on GPU ${embed_gpu} ==="
CUDA_VISIBLE_DEVICES="${embed_gpu}" nohup "$VERL_VENV_PYTHON" scripts/rwml/embed_server.py \
  --model "${embed_model}" \
  --device cuda:0 --port 8901 > runs/rwml_embed_server.log 2>&1 &
embed_pid=$!
trap 'kill ${embed_pid} 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8901/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:8901/health >/dev/null || { echo "embed server failed to start"; exit 1; }
echo "embed server ready (pid ${embed_pid})"

echo "=== launching veRL GRPO on GPUs ${train_gpus} ==="
CUDA_VISIBLE_DEVICES="${train_gpus}" "$VERL_VENV_PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${train_parquet}" \
  data.val_files="${val_parquet}" \
  data.train_batch_size=32 \
  data.val_batch_size=64 \
  data.max_prompt_length=3072 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${model_path}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${n_gpus}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=naive \
  custom_reward_function.path=scripts/rwml/rwml_reward.py \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name=e2l_rwml \
  trainer.experiment_name=rwml_grpo_merged10k \
  trainer.n_gpus_per_node="${n_gpus}" \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.total_epochs="${epochs}" \
  trainer.val_before_train=False \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.default_local_dir="${output_dir}" \
  "$@"

echo "=== RWML_GRPO_DONE ==="
