# E2L 远端训练 Playbook（8 × A100 80GB）

写给第一次拿到这套代码的远端同学：从零把环境搭起来，跑完 Stage 2 的 6 个训练臂。
实验设计、通过标准、产物回传清单在 `docs/handoff_experiments.md`，本文只管"怎么搭、怎么跑"。

## 0. 总览

| 东西 | 从哪来 | 说明 |
|---|---|---|
| e2-learning 仓库 | `git clone https://github.com/Ch1nyzzz/e2-learning.git` | 训练脚本、本文档、`docker/` |
| 训练镜像 | HF `erv1n/e2l-train-image`（或本地 `docker build`） | 含 verl-agent(stage2) 与 RWML 两套环境 |
| 冷启动 ckpt ×2 + RWML 数据 | HF `erv1n/e2l-*`（public） | `scripts/pull_hf_artifacts.sh` 一键拉 |
| 基座模型 ×3 | HF `Qwen/*` | 预拉或训练首次自动下载 |
| ALFWorld 游戏数据 | `alfworld-download` | 容器内跑一次即可 |
| verl-agent | 镜像内已带；非 Docker 路线跑 `scripts/setup_verl_agent.sh` | 上游 `20bd331` 原样 + 本仓库补丁（工作区 diff 形式），见 §9.2 |

## 1. 机器要求

- 单机 8 × A100 80GB（H100 同流程）；空闲磁盘 ≥ 1.5TB（镜像 ~40GB、模型与数据 ~150GB、各臂 checkpoint）
- NVIDIA 驱动 ≥ 550；Docker ≥ 24 + nvidia-container-toolkit（`docker run --gpus all` 可用）
- 能访问 huggingface.co 与 github.com（受限见 §8）

## 2. 拿训练镜像

先 clone 仓库，再从 Hugging Face 拉预构建包（~21GB）load 进 Docker。不必自己 `docker build`。

```bash
git clone https://github.com/Ch1nyzzz/e2-learning.git && cd e2-learning

hf download erv1n/e2l-train-image --repo-type dataset --local-dir ./e2l-train-image
cd e2l-train-image
sha256sum -c e2l-train.tar.gz.sha256
gunzip -c e2l-train.tar.gz | docker load   # 出现 e2l-train:latest
cd ..
```

下载慢或 huggingface.co 不通时：`export HF_ENDPOINT=https://hf-mirror.com` 后再跑 `hf download`。

备选：自己构建（1–2 小时，需能访问 GitHub / PyPI / Docker Hub）：

```bash
docker build -f docker/Dockerfile -t e2l-train:latest .
```

镜像里有什么：

- `/opt/verl-agent`：verl-agent 上游 `20bd331` 原样检出 + `patches/verl-agent-stage2-dual-reward.patch`（以工作区 diff 形式打上，= 未推送的 stage2-dual-reward 改动 559f9bd），装进系统 python 3.12（torch 2.8.0 / vllm 0.11.0 / flash-attn 2.8.3，下称 **env A**）
- `/opt/venvs/rwml`：python 3.10 独立 venv（verl 0.4.1 / vllm 0.8.3 / torch 2.6.0，**env B**），只给 E6 RWML 基线用
- 已预设 `VERL_AGENT_DIR`、`VERL_PYTHON`、`VERL_VENV_PYTHON`，训练脚本开箱即用

不用 Docker 也可以在已有 conda/venv 上手工复刻（与镜像同源）：

```bash
# env A（py3.12）：torch 2.8 + vllm 0.11 先行，其余按锁定清单装
pip install vllm==0.11.0
pip install -r docker/requirements-stage2.txt
# verl-agent：clone 上游 + 打补丁 + 可编辑安装（幂等）
VERL_AGENT_DIR=~/verl-agent INSTALL_EDITABLE=true VERL_PYTHON=python3 \
  bash scripts/setup_verl_agent.sh
# env B（py3.10，仅 E6 需要）
uv venv --python 3.10 .venv-verl
uv pip install --python .venv-verl/bin/python -r docker/requirements-rwml.txt
```

## 3. 拉模型与数据

```bash
# ① 冷启动 ckpt ×2 + RWML 数据（~60GB；默认命名空间 erv1n，可 HF_NAMESPACE 覆盖）
bash scripts/pull_hf_artifacts.sh

# ② 基座模型（~50GB；不预拉则训练首次自动下载）
hf download Qwen/Qwen2.5-7B-Instruct
hf download Qwen/Qwen3-8B
hf download Qwen/Qwen3-Embedding-0.6B

# ③ ALFWorld 游戏数据（容器内跑一次，落到挂载卷）
docker run --rm -v /path/to/alfworld-data:/data/alfworld e2l-train:latest \
  alfworld-download --data-dir /data/alfworld
```

## 4. 启动容器

```bash
docker run --gpus all -it --name e2l --shm-size=64g \
  -v $PWD:/workspace/e2-learning \
  -v /path/to/alfworld-data:/data/alfworld \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e ALFWORLD_DATA=/data/alfworld \
  e2l-train:latest
# 进容器后：
cd /workspace/e2-learning
```

要点：仓库以挂载方式进容器（脚本改动即时生效、checkpoint 直接落宿主机盘）；HF cache 挂载避免重复下载。W&B key 已写在 `.env.example`（随仓库走），训练脚本会自动读；run 文件写在仓库 `wandb/`（已 gitignore）。

## 5. 自检（10 分钟）

```bash
nvidia-smi        # 8 卡可见
python3 -c "import verl, vllm, alfworld, flash_attn; print('env A ok')"
/opt/venvs/rwml/bin/python -c "import verl, vllm; print('env B ok')"
cd /opt/verl-agent && CUDA_VISIBLE_DEVICES="" ALFWORLD_DATA=/data/alfworld \
  python3 tests_stage2/test_futile_tracker_cpu.py    # 5 个模式全 PASS
```

## 6. 正式训练（6 臂）

W&B 默认开：项目 `e2l_policy_grpo`（E6 为 `e2l_rwml`），run 名即 `EXPERIMENT_NAME`。API key 已在 `.env.example`，clone 下来即可用。上不了 wandb.ai 时加 `WANDB_MODE=offline`，训完在仓库根目录 `wandb sync wandb/`。只要 console：`LOGGER="['console']"`。

8 卡统一前缀（对齐 verl-agent 官方与 8 卡开源配置，依据见 `scripts/train_policy_grpo_stage2.sh` 头注释）：

```bash
cd /workspace/e2-learning
E8="GPUS=0,1,2,3,4,5,6,7 ROLLOUT_TP=1 GPU_MEM_UTIL=0.6 FSDP_PARAM_OFFLOAD=false FSDP_OPTIMIZER_OFFLOAD=false ENFORCE_EAGER=false FREE_CACHE_ENGINE=false"
```

三条 Qwen2.5 臂优先级最高（P0，单机串行按 E5→E2→E1）：

```bash
# E5 plain-from-base
env $E8 MODEL_PATH=Qwen/Qwen2.5-7B-Instruct EXPERIMENT_NAME=q25_stage2_plain \
  USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh
# E2 冷启动、无惩罚
env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen25-7b-coldstart EXPERIMENT_NAME=q25_stage2_pure \
  USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh
# E1 冷启动 + futile 惩罚（主臂）
env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen25-7b-coldstart EXPERIMENT_NAME=q25_stage2_dual \
  bash scripts/train_policy_grpo_stage2.sh
```

Qwen3 两臂（P1；nothink 由脚本内置）：

```bash
env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen3-8b-coldstart EXPERIMENT_NAME=q3_stage2_dual \
  bash scripts/train_policy_grpo_stage2.sh
env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen3-8b-coldstart EXPERIMENT_NAME=q3_stage2_pure \
  USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh
```

E6 RWML 基线（P2，自动使用镜像里的 env B；数据 §3① 已拉）：

```bash
RWML_TRAIN_GPUS=0,1,2,3,4,5,6 RWML_EMBED_GPU=7 \
RWML_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B RWML_TAU_D=0.20571374893188477 \
bash scripts/rwml/run_rwml_grpo.sh
```

断点续训：直接重跑同一命令（verl 自动找最新 checkpoint）。每个臂训完用
`scripts/keep_best_grpo_ckpt.py` 的逻辑按 val 最优保留 HF checkpoint。

## 7. 评测与产物回传

评测协议与回传清单照 `docs/handoff_experiments.md` §5/§7 执行：训练内 val（自动）作选
ckpt 依据；best-by-val 与 final 两 ckpt 做 T=0 双 split 全集（主口径）+ T=0.4×3（可比列）；
Qwen3 臂评测同样 nothink。回传：完整日志、两 ckpt、轨迹 JSONL + summary、每步耗时。

## 8. 常见故障

- `kl_loss` 第 0 步 ≠ 0 → prompt/tokenizer 配置错了，停下排查，不要硬跑。
- vLLM 侧 OOM → `GPU_MEM_UTIL=0.5`（再不行 0.45，或 `ROLLOUT_TP=2`）。
- Ray 冲突 → 脚本已隔离 `RAY_TMPDIR`；共享机上不要手删 `/tmp/ray*`。
- HF 下载失败/慢 → `export HF_ENDPOINT=https://hf-mirror.com`（镜像包 `erv1n/e2l-train-image`、`hf download`、pull 脚本都认）。
- wandb.ai 不可达 / `wandb: Network error` → `WANDB_MODE=offline` 先训，回头 `wandb sync wandb/`；或设 HTTP 代理。没 key 又想先跑：`LOGGER="['console']"`。
- GitHub 不可达 → 不要自己 `docker build`，用 §2 的 HF 镜像包 `docker load`；Dockerfile 里 `git clone langfengQ/verl-agent` 必须在有网机器完成。
- 8 卡每步耗时没有明显低于 4 卡预期 → 检查 `ROLLOUT_TP=1` 是否生效（日志里 vLLM engine 数应为 8）。
- 需要 e2-learning 自带的 stage-1 工具/离线评测（uv 环境）→ 进容器后在挂载仓库里 `uv sync --extra train --extra alfworld`。

## 9. 附录

### 9.1 环境版本（与本机工作环境逐项对齐）

| 环境 | 用途 | 关键版本 |
|---|---|---|
| env A（py3.12，镜像系统 python） | Stage 2 GRPO（E1–E5） | torch 2.8.0 / vllm 0.11.0 / flash-attn 2.8.3 / transformers 4.57.3 / verl 0.3.1.dev0（= verl-agent 20bd331+patch）/ alfworld 0.4.2 / ray 2.50.0 / tensordict 0.10.0 |
| env B（py3.10，`/opt/venvs/rwml`） | E6 RWML GRPO 基线 | torch 2.6.0 / vllm 0.8.3 / verl 0.4.1 / flash-attn 2.7.4.post1 / transformers 4.52.4 / tensordict 0.6.2 |
| env C（py3.11，运行时 `uv sync`） | stage-1 在线训练与评测工具 | 由仓库 `uv.lock` 锁定 |

完整锁定清单：`docker/requirements-stage2.txt`、`docker/requirements-rwml.txt`。

### 9.2 verl-agent 补丁机制（上游保持原样）

Stage 2 对 verl-agent 的改动（futile 惩罚、stage2 prompt、双奖励指标等，对应未推送的 commit
`559f9bd`）**不维护任何 fork/分支**：`patches/verl-agent-stage2-dual-reward.patch` 是唯一载体，
`scripts/setup_verl_agent.sh` 把它以**未提交的工作区 diff** 形式打到上游 pinned commit
`20bd331` 上。效果：

- 检出永远停在 upstream commit，`git -C $VERL_AGENT_DIR diff` 任何时候都精确显示我方改动（含新增文件）；
- `git checkout . && git clean -fd` 一键还原纯净上游；升上游只需改 pinned commit 并重新生成补丁；
- 脚本幂等：已打过补丁或已是 559f9bd 的检出会直接跳过；
- Docker 构建与非 Docker 装环境走同一脚本，两条路线不分叉。

改了 stage2 代码后重新生成补丁：

```bash
cd $VERL_AGENT_DIR && git add -A && git commit -m "stage2 changes"
git format-patch -1 --stdout > patches/verl-agent-stage2-dual-reward.patch   # 在 e2-learning 仓库内
# 并同步 scripts/setup_verl_agent.sh 里的 STAGE2_COMMIT 值
```

### 9.3 超参数依据

训练配方（lr 1e-6、KL 0.01 low_var_kl、16 任务 × 组内 8、150 步、val 128 局 T=0.4 每 5 步、
无效动作惩罚 0.1）对齐 verl-agent 官方 `examples/grpo_trainer/run_alfworld.sh` 与 GiGPO 论文附录 E；
8 卡系统侧覆盖项（`ROLLOUT_TP=1`、`GPU_MEM_UTIL=0.6`、关 FSDP offload、关 enforce_eager）对齐
EP-R1（1×8 A100 80GB）与 CoEvoSkill（8×H200）的开源配置。更新批量不随卡数放大，保证各臂可比。
