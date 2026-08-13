# Experience Learning

This repository implements the first controlled version of **environment-centric,
mistake-driven experience learning** in the deterministic ALFWorld TextWorld environment.

The training loop is intentionally not GRPO/PPO:

1. enumerate ALFWorld's admissible actions;
2. greedily predict the next observation for every action while measuring the model's mean
   next-token entropy;
3. choose the action with maximum mean token entropy;
4. execute that action in the real environment exactly once;
5. compare the selected action's greedy prediction with the real observation using a constrained
   OpenAI-compatible semantic judge;
6. perform a full-parameter next-observation update only when reality contradicts the model.

The external judge is only a semantic equivalence checker. ALFWorld remains the source of the
training target.

## Current scope

- ALFWorld `AlfredTWEnv`, all six supported task families
- train / valid-seen / valid-unseen split support
- token-entropy, semantic-sample-entropy, and random acquisition strategies
- mistake-only and all-transition update gates for the 2x2 ablation
- full-parameter Hugging Face training; LoRA/PEFT parameters are explicitly rejected
- Accelerate FSDP2 configurations for 2 or 4 x A100 80GB
- sharded model/optimizer/scheduler/RNG checkpoints and complete JSONL transition logs

Phase 1 evaluates world-model learning and interaction efficiency. Task-success claims need a
separately fixed planner, so they are deliberately not conflated with this first experiment.

## Vast.ai setup

Use an Ubuntu image with two or four visible A100 80GB GPUs, enough system RAM for model loading, and a
persistent volume. Python 3.11 is used because it is compatible with ALFWorld/TextWorld.

```bash
export ALFWORLD_DATA=/workspace/data/alfworld
cp .env.example .env
# Fill JUDGE_API_KEY, JUDGE_BASE_URL, and the provider-specific JUDGE_MODEL.

bash scripts/setup_vastai.sh
set -a && source .env && set +a
# Choose the script matching the visible GPU count.
bash scripts/train_2xa100_80gb.sh
# bash scripts/train_4xa100_80gb.sh
```

The provisional model is `Qwen/Qwen3-8B`, pinned to an immutable Hugging Face revision. Override
both fields together if a different base model is chosen:

```bash
bash scripts/train_2xa100_80gb.sh \
  --set model.name=YOUR_MODEL \
  --set model.revision=COMMIT_HASH \
  --set experiment.max_environment_steps=100
```

For a no-API infrastructure smoke run, use `--set judge.provider=exact_match`. It is not a valid
scientific substitute for semantic judging.

The primary comparison is a `2 x 2` factorial under the same environment-action budget and game
seed:

| Acquisition | Update gate | Role |
| --- | --- | --- |
| `random` | `all_transitions` | uniform data + ordinary next-observation SFT |
| `random` | `mistake_only` | isolates mistake gating |
| `token_entropy` | `all_transitions` | isolates active acquisition |
| `token_entropy` | `mistake_only` | proposed method |

Set the two fields with `--set acquisition.strategy=...` and
`--set training.update_gate=...`, and give every run a distinct `experiment.output_dir`. Random
acquisition still generates every candidate so model-generation compute remains comparable, but it
skips acquisition-time semantic clustering and its API cost.

Token-entropy acquisition performs one greedy prediction per candidate action, averages the full
vocabulary entropy over valid generated tokens, and reuses the selected action's prediction for the
reality check. The external judge is therefore called only after the selected action is executed.
Each candidate log also records `generated_tokens` and `hit_token_limit`, making it possible to
measure how often generation reaches `generation.max_new_tokens` without emitting EOS.
The older `semantic_entropy` strategy remains available as a higher-cost ablation; it samples
`generation.samples_per_action` outcomes and uses the judge to cluster every pair before selection.

## Reproducibility and safety boundaries

- Only rank 0 owns ALFWorld and the API judge; all ranks follow synchronized phases.
- The hand-coded ALFWorld expert is disabled. We use the environment's admissible candidate set.
- Candidate sampling never clones, rewinds, or steps the environment.
- A judge `UNCERTAIN` result does not update in the mistake-only condition.
- Prompt/history/action tokens are masked with `-100`; loss is applied only to the real next
  observation and EOS.
- Invalid judge output, timeout exhaustion, or a rank-0 environment error is broadcast as a stop
  instead of being silently treated as a correct prediction.
- ALFWorld game paths are sorted before registration to avoid filesystem-dependent episode order.

Runtime artifacts are written under the configured output directory:

```text
events.jsonl
resolved_config.json
judge_cache.sqlite3
checkpoints/
  env_step_000100/
  final/
```

Periodic checkpoint retention defaults to the newest three checkpoints via
`training.max_periodic_checkpoints_to_keep`; the final checkpoint is retained separately. Set the
value to `0` only when unlimited periodic checkpoint history is intentional.

Online collection can run synchronized environment waves with
`experiment.parallel_environments`. Candidate predictions from every active environment are
flattened into one distributed generation batch, semantic checks use up to
`judge.max_concurrency` workers, and gated examples accumulate until
`training.update_batch_size` before a teacher-forced SFT update. Each environment receives a
disjoint partition of the ALFWorld game list.

## Held-out transition evaluation

Create fixed probe sets before training so every method/seed is evaluated on identical
environment transitions:

```bash
uv run --extra alfworld experience-learning collect-probes \
  --config configs/alfworld_qwen3_8b.yaml \
  --split eval_in_distribution --episodes 50 \
  --output data/probes_valid_seen.jsonl

uv run --extra alfworld experience-learning collect-probes \
  --config configs/alfworld_qwen3_8b.yaml \
  --split eval_out_of_distribution --episodes 50 \
  --output data/probes_valid_unseen.jsonl
```

Evaluate a sharded checkpoint with the same two-GPU launch configuration:

```bash
uv run --extra train --extra alfworld accelerate launch \
  --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
  -m experience_learning.cli evaluate-probes \
  --config configs/alfworld_qwen3_8b.yaml \
  --checkpoint outputs/alfworld_qwen3_8b_parallel8/checkpoints/final \
  --probes data/probes_valid_unseen.jsonl \
  --output outputs/alfworld_qwen3_8b_parallel8/eval_valid_unseen.jsonl
```

The summary reports conservative semantic accuracy (uncertain counts as incorrect), a semantic
accuracy upper bound, definite-decision accuracy, judge coverage, and target-observation token NLL.
Retain the per-transition JSONL to audit judge errors.

## RWML paper baseline: fixed offline WM SFT

The RWML comparison is intentionally separate from this repository's online `random +
all_transitions` condition. RWML first collects a fixed on-policy transition corpus and trains WM
SFT and RWML on the same postprocessed data. The provided two-A100 workflow pins the paper's
ALFWorld base model, `Qwen/Qwen2.5-7B-Instruct`, and follows its three rollouts per 2,048 training
tasks, temperature 1.0, 30-step horizon, two SFT epochs, learning rate `2e-6`, and effective batch
size 32:

```bash
export ALFWORLD_DATA=/workspace/data/alfworld
bash scripts/collect_rwml_data_2xa100_80gb.sh
bash scripts/train_rwml_wm_sft_2xa100_80gb.sh
```

The default collection script creates a deterministic 2,000-transition training subset for the
interaction-budget-matched baseline:

```bash
RWML_SFT_EPOCHS=1 bash scripts/train_rwml_wm_sft_2xa100_80gb.sh \
  data/rwml_alfworld_qwen25_7b_train_matched2000.jsonl \
  data/rwml_alfworld_qwen25_7b_validation.jsonl \
  --set experiment.output_dir=outputs/rwml_wm_sft_matched2000_s42
```

The 2,000-record one-epoch run is the default interaction-count-matched control. A 15,813-record
paper-count-matched arm must be requested explicitly; because exact difficulty filtering is not
implemented yet, even that count-matched arm is not a complete paper replication. Run summaries
record dataset hashes, sample exposures, optimizer steps, and training target-token counts.

Collection refuses to overwrite an existing dataset and writes a SHA-256 manifest. Each rollout
pass recreates the sorted, strided ALFWorld environments so a task ID maps to the same game in all
three passes. The split command removes invalid-action transitions and deterministically makes a
90/10 train/validation split. WM SFT uses the paper's empty-reasoning target:
`<think> </think><next_state>...</next_state>`.

Important reproduction boundary: the generated split is a fixed shared **raw/postprocessed**
corpus, not yet the paper's final 15,813-example difficulty-filtered corpus. Exact RWML filtering
requires a separately SFT-trained filtering model, ten predictions per transition, and
Qwen3-Embedding-8B thresholding. The split manifest labels this explicitly. Do not report the
current result as the paper's filtered WM SFT until that stage is run. The same fixed raw corpus is
intended to feed both that filtering stage and the forthcoming RWML-GRPO implementation.

## End-to-end ALFWorld success rate

After training, run a single-seed paired comparison of the unchanged base model and final
checkpoint on all 140 `valid_seen` and all 134 `valid_unseen` games:

```bash
bash scripts/evaluate_sr_2xa100_80gb.sh
```

The deterministic ReAct policy sees the same admissible actions and game order for both models.
Each rollout runs for at most 30 environment steps and the summary reports both `SR@20` and
`SR@30`, invalid-action rate, mean successful trajectory length, and per-task-type success. This
evaluation uses environment task completion directly and does not call the semantic judge. Raw
trajectories and adjacent `*.summary.json` files are written below
`outputs/alfworld_qwen3_8b_parallel8/sr_eval/`.

The two-A100 script evaluates 64 ALFWorld environments concurrently and uses a generation
micro-batch of 32 per rank; these values are intentionally independent of the smaller online
training batches.

## Local checks

The tests do not download a language model or ALFWorld data:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

ALFWorld follows a batch-size-one API: `reset()` returns observations and infos; `step()` accepts
a one-element action list. The adapter exposes an ordinary single-state interface to the
experiment controller.
