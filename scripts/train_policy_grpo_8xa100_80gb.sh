#!/usr/bin/env bash
set -euo pipefail

# Run from a verl-agent checkout. This wrapper performs full-parameter GRPO
# (lora_rank=0) for ALFWorld with the same settings for Base and WM-SFT.
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model id or local HF checkpoint}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?Set EXPERIMENT_NAME}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
VERL_PYTHON=${VERL_PYTHON:-.venv/bin/python}
VERL_AGENT_DIR=${VERL_AGENT_DIR:-/workspace/verl-agent}
ALFWORLD_DATA=${ALFWORLD_DATA:-/workspace/data/alfworld}
TRAIN_STEPS=${TRAIN_STEPS:-300}
TRAIN_TASKS_PER_UPDATE=${TRAIN_TASKS_PER_UPDATE:-16}
GROUP_SIZE=${GROUP_SIZE:-8}
VAL_TASKS=${VAL_TASKS:-64}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-20}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-50}
EVAL_DATASET=${EVAL_DATASET:-eval_in_distribution}

export ALFWORLD_DATA
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

cd "$VERL_AGENT_DIR"

"$VERL_PYTHON" -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size "$TRAIN_TASKS_PER_UPDATE" \
  --val_data_size "$VAL_TASKS"

"$VERL_PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=False \
  data.train_files="${VERL_DATA_DIR:-$HOME/data/verl-agent}/text/train.parquet" \
  data.val_files="${VERL_DATA_DIR:-$HOME/data/verl-agent}/text/test.parquet" \
  data.train_batch_size="$TRAIN_TASKS_PER_UPDATE" \
  data.val_batch_size="$VAL_TASKS" \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.tokenizer="$TOKENIZER_PATH" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps="$MAX_EPISODE_STEPS" \
  env.history_length=2 \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus=0.1 \
  env.alfworld.eval_dataset="$EVAL_DATASET" \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name=e2l_policy_grpo \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TRAIN_STEPS" \
  trainer.total_training_steps="$TRAIN_STEPS" \
  trainer.val_before_train=False \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.default_local_dir="${CKPT_ROOT:-/workspace/checkpoints/e2l_policy_grpo}/$EXPERIMENT_NAME" \
  ray_init.num_cpus=64 \
  "$@"
