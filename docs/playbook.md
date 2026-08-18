# E2L 远端训练 Playbook（8 × A100 80GB）

给第一次拿到任务的人：从一台空机器跑完 Stage 2 的 6 个训练臂。
**不必 `git clone`。** 代码和环境都在 Docker 镜像里。

实验设计、通过标准、回传清单：`docs/handoff_experiments.md`（浏览器打开即可：
https://github.com/Ch1nyzzz/e2-learning/blob/main/docs/handoff_experiments.md ）。
**操作步骤以 GitHub `main` 上这份 playbook 为准**（镜像内文档可能滞后）：
https://github.com/Ch1nyzzz/e2-learning/blob/main/docs/playbook.md

## 0. 从零清单（按顺序勾）

1. 宿主机：GPU + Docker + `hf` CLI（§1–§2）
2. `docker pull` 或下载 tar 再 `docker load`（§3）
3. 建工作目录，启动容器（§4）
4. 容器内拉齐 ckpt / 数据 / 基座模型（§5）
5. 自检（§6）
6. 按 E5→E2→E1 开训，再 P1/P2（§7）
7. 留 ckpt、评测、回传（§8）

训练镜像在 **Docker Hub**（可 `docker pull`）。ckpt / 数据在 **Hugging Face 公开仓库**，不需要 token。路径、命名空间、镜像源都可以用环境变量改。

## 0.1 资源目录（全部可下载、可覆盖）

| 资源 | 默认从哪拉 | 覆盖方式 |
|---|---|---|
| 训练镜像（代码+环境，load 后 ~47GB） | Docker Hub [`yuhan778/e2l-train:latest`](https://hub.docker.com/r/yuhan778/e2l-train) | 国内慢：HF tar [`erv1n/e2l-train-image`](https://huggingface.co/datasets/erv1n/e2l-train-image) + `HF_ENDPOINT`；自己 `docker build` 见 §3 备选 |
| Qwen2.5 冷启动 ckpt | [`erv1n/e2l-alfworld-qwen25-7b-coldstart`](https://huggingface.co/erv1n/e2l-alfworld-qwen25-7b-coldstart) | `HF_NAMESPACE`、`Q25_REPO`、或训练时 `MODEL_PATH=...` |
| Qwen3 冷启动 ckpt | [`erv1n/e2l-alfworld-qwen3-8b-coldstart`](https://huggingface.co/erv1n/e2l-alfworld-qwen3-8b-coldstart) | `HF_NAMESPACE`、`Q3_REPO`、或 `MODEL_PATH=...` |
| RWML 数据（E6 必需） | 数据集 [`erv1n/e2l-rwml-alfworld-data`](https://huggingface.co/datasets/erv1n/e2l-rwml-alfworld-data) | `HF_NAMESPACE`、`RWML_DATA_REPO`、`RWML_DATA_DIR` |
| 基座 `Qwen2.5-7B-Instruct` | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `MODEL_PATH` / `RWML_MODEL_PATH` |
| 基座 `Qwen3-8B` | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | `MODEL_PATH` |
| Embedding `Qwen3-Embedding-0.6B`（E6） | [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | `RWML_EMBED_MODEL` |
| ALFWorld 游戏数据 | 镜像内 `alfworld-download` | `ALFWORLD_DATA`（容器里默认 `/data/alfworld`） |
| W&B | 镜像内 `.env.example` 已带 key | `WANDB_API_KEY`、`WANDB_ENTITY`、`WANDB_PROJECT`、`WANDB_MODE` |
| HF 镜像站（下载慢/被墙） | `https://hf-mirror.com` | `export HF_ENDPOINT=https://hf-mirror.com` |

拉冷启动 + RWML 数据的脚本：`scripts/pull_hf_artifacts.sh`（认上面那些 `HF_*` / `Q25_*` / `RWML_*` 变量）。
训练命令里的 `MODEL_PATH=erv1n/...` 或 `Qwen/...` 也可以不预拉，第一次跑时自动进 HF cache。
**E6 例外**：必须先有 `data/rwml_grpo/{train,val}.parquet`，所以 §5 的 `pull_hf_artifacts.sh` 对 E6 是必做。

## 1. 机器要求

- 单机 8 × A100 80GB（H100 同流程）。4 卡也能跑，把 §7 的 `GPUS=` 改成实际卡号，并改用 `ROLLOUT_TP=2`（不要用下面的 8 卡 `E8` 前缀）
- 空闲磁盘 ≥ 1.5TB（镜像 ~47GB、模型与数据 ~150GB、各臂 checkpoint）
- NVIDIA 驱动 ≥ 550；Docker ≥ 24 + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，保证 `docker run --gpus all` 可用
- 能访问 Docker Hub（拉镜像）和 huggingface.co（或 §0.1 的 HF 镜像站，拉 ckpt/数据）。主路径不需要 GitHub

自检 GPU / Docker：

```bash
nvidia-smi          # 应看到 8 张卡
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 2. 宿主机：只装下载工具

§3 走 `docker pull` 时这里可跳过。走 HF tar、以及后面拉 ckpt/数据时，宿主机需要 `hf` 或 `wget`。

```bash
python3 -m pip install -U "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
# 下载慢或 huggingface.co 不通：
# export HF_ENDPOINT=https://hf-mirror.com
```

没有 pip 时用 §3 的 `wget` 备选。

## 3. 拿训练镜像

优先 `docker pull`（平台只要填镜像网址时用这个）：

```bash
docker pull yuhan778/e2l-train:latest
docker tag yuhan778/e2l-train:latest e2l-train:latest   # 后面命令都用这个本地 tag
docker images e2l-train:latest                          # ~46.6GB
```

国内 Docker Hub 慢或不通时，下 HF tar 再 `docker load`（约 21GB 压缩包）：

```bash
hf download erv1n/e2l-train-image --repo-type dataset --local-dir ./e2l-train-image
cd e2l-train-image
sha256sum -c e2l-train.tar.gz.sha256
gunzip -c e2l-train.tar.gz | docker load   # 应出现 Loaded image: e2l-train:latest
cd ..
docker images e2l-train:latest             # ~46.6GB
```

`wget` 拉 tar（`HF_ENDPOINT` 默认同上；国内先 `export HF_ENDPOINT=https://hf-mirror.com`）：

```bash
BASE="${HF_ENDPOINT:-https://huggingface.co}/datasets/erv1n/e2l-train-image/resolve/main"
mkdir -p e2l-train-image && cd e2l-train-image
wget -c "$BASE/e2l-train.tar.gz"
wget -c "$BASE/e2l-train.tar.gz.sha256"
sha256sum -c e2l-train.tar.gz.sha256
gunzip -c e2l-train.tar.gz | docker load
```

镜像里有什么：

- `/workspace/e2-learning`：训练脚本、配置、`.env.example`（W&B key）、本 playbook
- `/opt/verl-agent`：上游 `20bd331` + `patches/verl-agent-stage2-dual-reward.patch`（env A：torch 2.8.0 / vllm 0.11.0 / flash-attn 2.8.3）
- `/opt/venvs/rwml`：Python 3.10（env B，仅 E6：torch 2.6.0 / vllm 0.8.3 / verl 0.4.1）
- 已设 `VERL_AGENT_DIR=/opt/verl-agent`、`VERL_PYTHON=python3`、`VERL_VENV_PYTHON=/opt/venvs/rwml/bin/python`
- 工作目录就是 `/workspace/e2-learning`，`docker run` 进 bash

**不要**把宿主机 git 仓库 mount 到 `/workspace/e2-learning`，会把镜像里的代码盖掉。

备选：自己构建（1–2 小时，需要 GitHub / PyPI / Docker Hub）：

```bash
git clone https://github.com/Ch1nyzzz/e2-learning.git && cd e2-learning
docker build -f docker/Dockerfile -t e2l-train:latest .
```

## 4. 工作目录 + 启动容器

checkpoint、数据、日志必须挂到宿主机，否则删容器就没了。

```bash
export WORK=/data/e2l    # 改成你盘上的路径，后面命令都依赖它
mkdir -p "$WORK"/{checkpoints,outputs,data,runs,wandb,alfworld}

# ALFWorld 游戏数据，只跑一次
docker run --rm -v "$WORK/alfworld":/data/alfworld e2l-train:latest \
  alfworld-download --data-dir /data/alfworld

docker run --gpus all -it --name e2l --shm-size=64g \
  -v "$WORK/checkpoints":/workspace/e2-learning/checkpoints \
  -v "$WORK/outputs":/workspace/e2-learning/outputs \
  -v "$WORK/data":/workspace/e2-learning/data \
  -v "$WORK/runs":/workspace/e2-learning/runs \
  -v "$WORK/wandb":/workspace/e2-learning/wandb \
  -v "$WORK/alfworld":/data/alfworld \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e ALFWORLD_DATA=/data/alfworld \
  e2l-train:latest
```

进去后 cwd 已是 `/workspace/e2-learning`。另开一个宿主机终端进同一容器：

```bash
docker exec -it e2l bash
```

容器停了再开：`docker start -ai e2l`。HF cache 挂在宿主机 `~/.cache/huggingface`，模型只下一次。

## 5. 容器内拉齐权重和数据

若宿主机设过 `HF_ENDPOINT`，容器里要再设一次（`-e` 没传进去就不会有）：

```bash
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# ① 冷启动 ckpt → outputs/… ；RWML 数据 → data/rwml_grpo/（E6 必需）。约 60GB
#    换命名空间：HF_NAMESPACE=someone bash scripts/pull_hf_artifacts.sh
bash scripts/pull_hf_artifacts.sh

# ② 三个基座（约 50GB）。不预拉则第一次训练/E6 时自动下到 HF cache
hf download Qwen/Qwen2.5-7B-Instruct
hf download Qwen/Qwen3-8B
hf download Qwen/Qwen3-Embedding-0.6B
```

核对：

```bash
ls data/rwml_grpo/train.parquet data/rwml_grpo/val.parquet
ls outputs/alfworld_qwen25_7b_active_10000_hf/checkpoints/env_step_001200/config.json
ls outputs/alfworld_qwen3_8b_active_10000_hf/checkpoints/env_step_006220/config.json
```

训练命令用 Hub id（`MODEL_PATH=erv1n/...` 或 `Qwen/...`）即可，不必改成本地路径。

## 6. 自检（约 10 分钟）

```bash
nvidia-smi        # 8 卡可见
python3 -c "import verl, vllm, alfworld, flash_attn; print('env A ok')"
/opt/venvs/rwml/bin/python -c "import verl, vllm; print('env B ok')"
cd /opt/verl-agent && CUDA_VISIBLE_DEVICES="" ALFWORLD_DATA=/data/alfworld \
  python3 tests_stage2/test_futile_tracker_cpu.py    # 5 个模式全 PASS
cd /workspace/e2-learning
```

## 7. 正式训练（6 臂）

W&B 默认开：项目 `e2l_policy_grpo`（E6 为 `e2l_rwml`），run 名 = `EXPERIMENT_NAME`。
key 在镜像 `.env.example`。上不了 wandb.ai：`WANDB_MODE=offline`，训完 `wandb sync wandb/`。
只要 console：`LOGGER="['console']"`。

8 卡统一前缀（对齐 verl-agent 官方与 8 卡开源配置）：

```bash
E8="GPUS=0,1,2,3,4,5,6,7 ROLLOUT_TP=1 GPU_MEM_UTIL=0.6 FSDP_PARAM_OFFLOAD=false FSDP_OPTIMIZER_OFFLOAD=false ENFORCE_EAGER=false FREE_CACHE_ENGINE=false"
```

**P0，单机串行 E5→E2→E1**（论文核心消融）。每个臂另开一个 `docker exec` 跑 keep_best（见 §8），日志用 `tee`：

```bash
# E5 plain-from-base
env $E8 MODEL_PATH=Qwen/Qwen2.5-7B-Instruct EXPERIMENT_NAME=q25_stage2_plain \
  USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh \
  2>&1 | tee runs/q25_stage2_plain.log

# E2 冷启动、无惩罚
env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen25-7b-coldstart EXPERIMENT_NAME=q25_stage2_pure \
  USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh \
  2>&1 | tee runs/q25_stage2_pure.log

# E1 冷启动 + futile 惩罚（主臂）
env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen25-7b-coldstart EXPERIMENT_NAME=q25_stage2_dual \
  bash scripts/train_policy_grpo_stage2.sh \
  2>&1 | tee runs/q25_stage2_dual.log
```

Qwen3 两臂（P1；nothink 由脚本内置）：

```bash
env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen3-8b-coldstart EXPERIMENT_NAME=q3_stage2_dual \
  bash scripts/train_policy_grpo_stage2.sh \
  2>&1 | tee runs/q3_stage2_dual.log

env $E8 MODEL_PATH=erv1n/e2l-alfworld-qwen3-8b-coldstart EXPERIMENT_NAME=q3_stage2_pure \
  USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh \
  2>&1 | tee runs/q3_stage2_pure.log
```

E6 RWML 基线（P2，自动用 env B；§5 的 parquet 必须已在）：

```bash
RWML_TRAIN_GPUS=0,1,2,3,4,5,6 RWML_EMBED_GPU=7 \
RWML_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B RWML_TAU_D=0.20571374893188477 \
bash scripts/rwml/run_rwml_grpo.sh \
  2>&1 | tee runs/rwml_grpo_merged10k.log
```

断点续训：同样命令再跑一次（verl 会找最新 checkpoint）。

常用覆盖（不改脚本）：`GPUS`、`MODEL_PATH`、`EXPERIMENT_NAME`、`TRAIN_STEPS`、`CKPT_ROOT`、`ALFWORLD_DATA`、`LOGGER`、`WANDB_*`；E6 还有 `RWML_TRAIN_PARQUET`、`RWML_VAL_PARQUET`、`RWML_MODEL_PATH`、`RWML_OUTPUT_DIR`、`RWML_TAU_D`。

## 8. 留 ckpt、评测、回传

每个臂开训后，另开一个终端（`docker exec -it e2l bash`）盯 val 最优并拷成 HF 目录：

```bash
# 把 EXPERIMENT 换成当前臂名，例如 q25_stage2_plain
python3 scripts/keep_best_grpo_ckpt.py \
  runs/${EXPERIMENT}.log \
  checkpoints/e2l_policy_grpo/${EXPERIMENT}
```

产物：`checkpoints/e2l_policy_grpo/<臂名>/best_hf/` 以及 verl 自己转的 final。

评测协议与回传清单见
[`docs/handoff_experiments.md`](https://github.com/Ch1nyzzz/e2-learning/blob/main/docs/handoff_experiments.md) §5 / §7：
训练内 val（自动）作选 ckpt 依据；best-by-val 与 final 两 ckpt 做 T=0 双 split 全集（主口径）+ T=0.4×3（可比列）；Qwen3 评测同样 nothink。

离线 SR 评测用仓库自带的 `experience_learning.cli evaluate-sr`，需要先装 env C：

```bash
uv sync --extra train --extra alfworld
# 然后按 scripts/evaluate_sr_2xa100_80gb.sh 的模式，把 --checkpoint 指到 best_hf 或 final
```

回传：完整日志（`runs/*.log`）、两 ckpt、轨迹 JSONL + summary、每步耗时。

## 9. 常见故障

- `kl_loss` 第 0 步 ≠ 0 → prompt/tokenizer 配错了，停下排查，不要硬跑
- vLLM OOM → `GPU_MEM_UTIL=0.5`（再不行 0.45，或 `ROLLOUT_TP=2`）
- Ray 冲突 → 脚本已隔离 `RAY_TMPDIR`；共享机不要手删 `/tmp/ray*`
- HF 下载失败/慢 → 宿主机和容器都 `export HF_ENDPOINT=https://hf-mirror.com`
- `hf: command not found`（宿主机）→ §2 的 pip；或用 §3 `wget`
- wandb.ai 不可达 → `WANDB_MODE=offline`；没 key 又想先跑：`LOGGER="['console']"`
- GitHub 不可达 → 不要自己 `docker build`，用 §3 的 `docker pull` 或 HF tar `docker load`
- `docker pull yuhan778/e2l-train` 慢/失败 → 国内改走 §3 的 HF tar + `HF_ENDPOINT=https://hf-mirror.com`
- 8 卡每步没有明显快于 4 卡 → 检查 `ROLLOUT_TP=1`（日志里 vLLM engine 数应为 8）
- `docker run` 起来却是 vLLM API 服务而不是 bash → 镜像版本旧，重新 §3 load 带代码的那一版
- E6 报找不到 parquet → 没跑 §5 的 `pull_hf_artifacts.sh`，或没挂 `data/`

## 10. 附录

### 10.1 环境版本

| 环境 | 用途 | 关键版本 |
|---|---|---|
| env A（py3.12，镜像系统 python） | Stage 2 GRPO（E1–E5） | torch 2.8.0 / vllm 0.11.0 / flash-attn 2.8.3 / transformers 4.57.3 / verl 0.3.1.dev0（= verl-agent 20bd331+patch）/ alfworld 0.4.2 / ray 2.50.0 / tensordict 0.10.0 |
| env B（py3.10，`/opt/venvs/rwml`） | E6 RWML GRPO 基线 | torch 2.6.0 / vllm 0.8.3 / verl 0.4.1 / flash-attn 2.7.4.post1 / transformers 4.52.4 / tensordict 0.6.2 |
| env C（py3.11，运行时 `uv sync`） | stage-1 工具与离线 SR 评测 | 由仓库 `uv.lock` 锁定 |

锁定清单：`docker/requirements-stage2.txt`、`docker/requirements-rwml.txt`。

### 10.2 不用 Docker 时手工复刻（与镜像同源）

```bash
git clone https://github.com/Ch1nyzzz/e2-learning.git && cd e2-learning
pip install vllm==0.11.0
pip install -r docker/requirements-stage2.txt
VERL_AGENT_DIR=~/verl-agent INSTALL_EDITABLE=true VERL_PYTHON=python3 \
  bash scripts/setup_verl_agent.sh
uv venv --python 3.10 .venv-verl
uv pip install --python .venv-verl/bin/python -r docker/requirements-rwml.txt
```

### 10.3 verl-agent 补丁机制（上游保持原样）

Stage 2 改动（对应未推送的 `559f9bd`）**不维护 fork**：`patches/verl-agent-stage2-dual-reward.patch` 是唯一载体，`scripts/setup_verl_agent.sh` 以未提交工作区 diff 打到上游 `20bd331`。Docker 构建与手工安装走同一脚本。

### 10.4 超参数依据

训练配方（lr 1e-6、KL 0.01 low_var_kl、16 任务 × 组内 8、150 步、val 128 局 T=0.4 每 5 步、无效动作惩罚 0.1）对齐 verl-agent 官方 `examples/grpo_trainer/run_alfworld.sh` 与 GiGPO 附录 E。8 卡系统侧（`ROLLOUT_TP=1`、`GPU_MEM_UTIL=0.6`、关 FSDP offload、关 enforce_eager）对齐 EP-R1 与 CoEvoSkill。更新批量不随卡数放大，保证各臂可比。
