# 交接实验清单：Stage 2 双奖励 GRPO + RWML 基线

> 2026-08-17 定稿。目标机器：8 × A100 80GB（≥4 × 80GB 均可，8 卡用 §2 的
> 8-GPU 覆盖项）。所有训练用 verl-agent
> 的 `stage2-dual-reward` 分支（commit `559f9bd`）+ 本仓库 `main`（commit `87e8560`）。
> 环境搭建与镜像打包按 `docs/playbook.md` 执行，本文只列实验与验收。
> 疑问先看本文档末尾的"常见故障"，再联系 yuhan。

## 0. 背景一句话

两阶段方法：Stage 1（在线环境学习，已完成，产物是冷启动 checkpoint）→
**Stage 2（experience learning RL）= GRPO 双奖励**：环境成功奖励（+10，主）+
futile 重复惩罚（辅，连击递增、封顶、随成功率退火到 0）。惩罚判定用 TextWorld
特权谓词做实体切片比较，只进 reward 通道，策略只看文本。所有臂共用同一
policy prompt（含一句"先回顾历史"的中性指令）与全历史窗口（history_length=50，
max_prompt_length=4096）。

## 1. 前置条件（开跑前逐项核对）

| 项 | 内容 |
|---|---|
| 代码 | e2-learning `main`；verl-agent = 上游 `20bd331` 原样 + 本仓库 `patches/verl-agent-stage2-dual-reward.patch`（`scripts/setup_verl_agent.sh` 打补丁，镜像内已打好） |
| 环境 | 按 verl-agent README 装（Python 3.12 / vllm 0.11 / verl 0.3.1.dev / flash-attn 2.8.3 / alfworld 0.4.2）；`ALFWORLD_DATA` 指向 ALFWorld `json_2.1.1` 数据 |
| 冷启动 ckpt | HF Hub：`erv1n/e2l-alfworld-qwen25-7b-coldstart`（= `alfworld_qwen25_7b_active_10000_hf/checkpoints/env_step_001200`）、`erv1n/e2l-alfworld-qwen3-8b-coldstart`（= `alfworld_qwen3_8b_active_10000_hf/checkpoints/env_step_006220`）。`MODEL_PATH` 可直接填 repo id，或先 `bash scripts/pull_hf_artifacts.sh` 拉到本地 |
| RWML 数据 | HF 数据集仓库 `erv1n/e2l-rwml-alfworld-data`：`rwml_grpo/{train,val}.parquet`、`rwml_tau_calibration.json`（tau_d≈0.2057）、`rwml_alfworld_qwen25_7b_validation_merged10k.jsonl`。`pull_hf_artifacts.sh` 会按仓库 `data/` 布局还原 |
| 基座模型 | `Qwen/Qwen2.5-7B-Instruct`、`Qwen/Qwen3-8B`、`Qwen/Qwen3-Embedding-0.6B`（HF 直接拉） |
| CPU 自检 | `cd verl-agent && CUDA_VISIBLE_DEVICES="" ALFWORLD_DATA=... .venv/bin/python tests_stage2/test_futile_tracker_cpu.py` → 5 个模式全 PASS |

不需要任何 judge API key：Stage 2 训练与 SR 评测都只用环境真值。

## 2. 第一步：训练冒烟（半天，必做）

任何长训之前，先在目标机器跑 10 步小规模（8 卡机把 `GPUS` 换成
`0,1,2,3,4,5,6,7` 并加上 §3 的 8-GPU 覆盖项）：

```bash
GPUS=0,1,2,3 MODEL_PATH=<q25_coldstart> EXPERIMENT_NAME=smoke_stage2 \
  TRAIN_STEPS=10 SAVE_FREQ=-1 TEST_FREQ=5 \
  bash scripts/train_policy_grpo_stage2.sh
```

冒烟通过标准（日志逐项核对）：

1. `actor/kl_loss` 第 0 步 ≈ 0（模板一致性；不为 0 说明 prompt/tokenizer 配置错了，停下排查）；
2. 出现 `futile/coef`、`futile/sr_ema`、`futile/weighted_mean`、`futile/penalty_mean` 指标，且 coef 从 0.25 起步、随 sr_ema 上升而下降；
3. `prompt_length/max` < 4096 且 `clip_ratio` ≈ 0（全历史装得下）；
4. 每步时长记录下来（预估 4 卡 1500-2500s/步，用于排期）。

## 3. 训练实验（6 项）

统一超参（脚本默认）：GRPO，150 步，每步 16 任务 × 组内 8 rollout，episode 上限
50 步，lr 1e-6，KL loss 系数 0.01（参考模型 = 初始化模型，即冷启动 ckpt），
无效动作惩罚 0.1，futile 惩罚 coef 0.25 / 封顶 12 units / sr_target 0.7，
val 每 5 步（128 局 valid_seen，T=0.4）。Qwen3 全部臂 **nothink**
（脚本已设 `enable_thinking=False`），Qwen2.5 不受该开关影响。
4 卡时建议 `ROLLOUT_TP=2`（脚本默认），让 vLLM 引擎落在 NVLink 对内。
8 × A100 80GB 时按脚本头部注释的 8-GPU 覆盖项跑（对齐 verl-agent 官方与
EP-R1 等 8 卡开源配置）：`ROLLOUT_TP=1 GPU_MEM_UTIL=0.6
FSDP_PARAM_OFFLOAD=false FSDP_OPTIMIZER_OFFLOAD=false ENFORCE_EAGER=false
FREE_CACHE_ENGINE=false`。更新批量保持 16 任务 × 组内 8（128 条/次），
8 卡只换吞吐、不改训练量，各臂与 4 卡结果可比。

| # | 实验名 | 初始化 | 惩罚 | 命令要点 | 优先级 |
|---|---|---|---|---|---|
| E5 | `q25_stage2_plain` | Q2.5 base | 关 | `USE_FUTILE_PENALTY=false MODEL_PATH=Qwen/Qwen2.5-7B-Instruct` | **P0** |
| E2 | `q25_stage2_pure` | Q2.5 冷启动 (1200) | 关 | `USE_FUTILE_PENALTY=false` | **P0** |
| E1 | `q25_stage2_dual` | Q2.5 冷启动 (1200) | 开 | 脚本默认 | **P0** |
| E3 | `q3_stage2_dual` | Q3 冷启动 (6220) | 开 | 脚本默认 | P1 |
| E4 | `q3_stage2_pure` | Q3 冷启动 (6220) | 关 | `USE_FUTILE_PENALTY=false` | P1 |
| E6 | `rwml_grpo_merged10k` | Q2.5 base | —（RWML 奖励） | 见 §4 | P2 |

精确的 6 条启动命令写在 `scripts/train_policy_grpo_stage2.sh` 顶部注释里。
**三条 Qwen2.5 臂（E5→E2→E1）是最高优先级**：它们构成完整因果链
base+RL vs 冷启动+RL vs 冷启动+RL+惩罚，一次隔离出 Stage 1 与惩罚项
各自的净效应，是论文的核心消融。只有一台 4 卡机时按 E5→E2→E1 顺序跑；
有多台时三条并行。
（Q3 plain-from-base 的旧 prompt 版本已在原机器完成，无需重跑；如需新 prompt
版本对照，按脚本注释第 6 条跑，标 P3。）

每个臂训完后运行 `scripts/keep_best_grpo_ckpt.py` 的同款逻辑（按 val 最优保
checkpoint，HF 格式）或至少保留 val 曲线最高点与最终步两个 checkpoint。

## 4. E6：RWML GRPO 基线

数据已全部就绪（过滤、tau 校准、parquet 都做完了），只差训练与评测：

```bash
# 4 卡：3 卡训练 + 1 卡 embedding 奖励服务（或按脚本默认 2+1 卡）；
# 8 卡机可 RWML_TRAIN_GPUS=0,1,2,3,4,5,6 RWML_EMBED_GPU=7
RWML_TRAIN_GPUS=0,1,2 RWML_EMBED_GPU=3 \
  RWML_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B \
  RWML_TAU_D=0.20571374893188477 \
  bash scripts/rwml/run_rwml_grpo.sh
# 训完:merge 到 HF + 双 split SR 评测,照抄 scripts/rwml/run_rwml_pipeline.sh 的 [6/6] 段
```

注意：不要跑 `run_rwml_pipeline.sh` 全流程（其 [1/6]-[4/6] 已完成，且第 5 步的
等待循环绑定原机器进程名）；直接跑 `run_rwml_grpo.sh` 即可，脚本已内置
`RAY_ADDRESS=local` 隔离。

## 5. 评测协议（每个训练臂都要）

**训练中 val**：脚本内置（GiGPO 官方配方：T=0.4、128 局 valid_seen、50 步），
不用额外操作，作为选 checkpoint 依据。

**最终测试（对 best-by-val 与 final 两个 checkpoint）**：

1. **主口径**：T=0 贪心、单次、双 split 全集（valid_seen 140 + valid_unseen 134）、
   30 步上限、报 SR@20 与 SR@30 + 无效动作率 + 平均成功步长。确定性环境 +
   贪心 = 单次即精确值。
2. **可比列**：同 checkpoint 用 T=0.4 采样评 3 次，报 mean±std（对齐 GiGPO/RWML
   谱系的报告方式）。
3. **模式一致**：Qwen3 的臂训练是 nothink，评测必须同样 nothink（社区一致实践，
   verl-agent 的 val 自动继承；离线 SR 评测脚本注意传相同的 chat-template 设置）。
4. **过程指标**（比 SR 灵敏一个量级，每个 checkpoint 都算）：无效重复率
   （futile/集）、P(换动作|Nothing happens)、重访率——用轨迹 JSONL 离线算。

## 6. 监控与中断处理

- 盯 `episode/success_rate`（训练）与 `val/success_rate`；双奖励臂另盯
  `futile/weighted_mean`（应随训练下降）与 `futile/coef`（应随 SR 上升衰减）。
- 退火哨兵：若出现"futile 降到 0 但 success 不涨"（学成不重复也不干活），
  提前告警——理论上组内比较会压制此模式，但要人工确认。
- 断点续训：直接重跑同一命令（verl 自动找最新 checkpoint）；`futile_sr_ema`
  已随 checkpoint 持久化，退火状态不会重置。
- 机器共享时切勿删 `/tmp/ray*`；脚本已用独立 `RAY_TMPDIR`。

## 7. 需要回传的产物

每个臂：① 完整训练日志（含全部 metrics 行）；② best-by-val 与 final 的 HF
checkpoint；③ 最终测试的轨迹 JSONL + summary（两种口径 × 两个 split）；
④ 冒烟记录的每步时长。目录结构随意，实验名对上表即可。
