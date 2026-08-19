# 交接实验清单：Stage 2 双奖励 GRPO + RWML 基线

> 2026-08-17 定稿，2026-08-18 更新（rollout 全量记录 + prompt 定稿为 verl 官方模板、去掉 stage2 句）。
> 目标机器：8 × A100 80GB（≥4 × 80GB 均可）。所有训练用 verl-agent = 上游
> `20bd331` + 本仓库 `patches/verl-agent-stage2-dual-reward.patch`（对应本地改动
> `559f9bd`+`810d961`+`c5bbd88`+`d3155b9`：双奖励、rollout 记录（含原始生成）、e2l prompt 诊断格式）。
> **2026-08-18 之前旧 prompt 格式启动的任何 Stage-2 臂全部作废，必须停掉、
> 拉新代码/新镜像后重跑**（旧 verl 模板把冷启动 ckpt 的 val 压到 0.078，
> 且反转臂间排序，数字不可用）。
> 环境搭建与镜像打包按 `docs/playbook.md` 执行，本文只列实验与验收。
> 疑问先看本文档末尾的"常见故障"，再联系 yuhan。

## 0. 背景一句话

两阶段方法：Stage 1（在线环境学习，已完成，产物是冷启动 checkpoint）→
**Stage 2（experience learning RL）= GRPO 双奖励**：环境成功奖励（+10，主）+
futile 重复惩罚（辅，连击递增、封顶、随成功率退火到 0）。惩罚判定用 TextWorld
特权谓词做实体切片比较，只进 reward 通道，策略只看文本。所有臂共用同一
policy prompt 与全历史窗口（history_length=50）。
**Prompt 格式（2026-08-18 终定）= verl-agent 官方 ALFWorld 模板原文**
（`PROMPT_FORMAT=verl` 为脚本默认），不加任何额外指令——原设计的 stage2
"先回顾历史"句已删除（`STAGE2_PROMPT=false` 为默认，Q3 起点 A/B 实测该句
对 val 影响≈0）。与官方配方的唯一差异是 `history_length=50`（futile 机制
需要全历史）。起点基线（正式口径：无句版，128 局 valid_seen，T=0.4，50 步，Q2.5）：
**冷启动 0.156 / base 0.125**（SR@20：0.086 / 0.031；带句版对照 0.148 / 0.227
——该句对冷启动无影响，但给 base 的 50 步游荡式晚胜 +0.10，删句后臂间
起点为冷启动领先，因果链方向正确）。诊断用的
e2l 格式（Stage-1 同款，system+user 双消息）保留在 `PROMPT_FORMAT=e2l`，
实测其 128 局 val 最长 prompt ≈2.9k tokens；verl 模板下 Q2.5 @4096、Q3
@5120 亦无截断。

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

任何长训之前，先在目标机器跑 10 步小规模（`launch_stage2_arm.sh` 会按臂自动
选 8 卡 Q2.5 档位或 4 卡 Q3 档位，见 §3）：

```bash
ARM=q25_dual EXPERIMENT_NAME=smoke_stage2 TRAIN_STEPS=10 SAVE_FREQ=-1 TEST_FREQ=5 \
  bash scripts/launch_stage2_arm.sh
```

冒烟通过标准（日志逐项核对）：

1. `actor/kl_loss` 第 0 步 ≈ 0（模板一致性；不为 0 说明 prompt/tokenizer 配置错了，停下排查）；
2. 出现 `futile/coef`、`futile/sr_ema`、`futile/weighted_mean`、`futile/penalty_mean` 指标，且 coef 从 0.25 起步、随 sr_ema 上升而下降；
3. `prompt_length/max` < 档位上限（Q2.5 4096 / Q3 5120）且 `clip_ratio` ≈ 0
   （全历史装得下）；
4. 每步时长记录下来（预估 4 卡 1500-2500s/步，用于排期）。

## 3. 训练实验（6 项）

统一超参（脚本默认）：GRPO，150 步，每步 16 任务 × 组内 8 rollout，episode 上限
50 步，lr 1e-6，KL loss 系数 0.01（参考模型 = 初始化模型，即冷启动 ckpt），
无效动作惩罚 0.1，futile 惩罚 coef 0.25 / 封顶 12 units / sr_target 0.7，
val 每 5 步（128 局 valid_seen，T=0.4）。Qwen3 全部臂 **nothink**
（脚本已设 `enable_thinking=False`），Qwen2.5 不受该开关影响。
**统一入口：`ARM=<臂名> bash scripts/launch_stage2_arm.sh`**，按臂自动选
硬件档位：Q2.5 臂 = 8 卡 A100 档位（`ROLLOUT_TP=1 GPU_MEM_UTIL=0.6`、关
FSDP offload、关 enforce_eager，对齐 verl-agent 官方与 EP-R1 等 8 卡配置）；
Q3 臂 = 4 卡 80GB 档位（`ROLLOUT_TP=2`，NVLink 对内，默认
`GPU_MEM_UTIL=0.6` + `PPO_MAX_TOKEN_LEN=16384`/`LOGPROB_MAX_TOKEN_LEN=32768`
实测 H100 提速档；共享机被挤占时降回 `GPU_MEM_UTIL=0.35` 并 unset 两个
token 预算）。更新批量保持 16 任务 × 组内 8（128 条/次），8 卡只换吞吐、
不改训练量，各臂与 4 卡结果可比。
所有 train/val rollout 自动全量落盘为
`checkpoints/e2l_policy_grpo/<臂名>/rollouts/{train,val}_gstep*.jsonl.gz`
（meta + step 行含 prompt/response/reward + episode 行；§5 的过程指标直接用它算，
`ROLLOUT_LOG_DIR=null` 可关）。

| # | 实验名 | 初始化 | 惩罚 | 启动命令 | 优先级 |
|---|---|---|---|---|---|
| E5 | `q25_stage2_plain` | Q2.5 base | 关 | `ARM=q25_plain bash scripts/launch_stage2_arm.sh` | **P0** |
| E2 | `q25_stage2_pure` | Q2.5 冷启动 (1200) | 关 | `ARM=q25_pure bash scripts/launch_stage2_arm.sh` | **P0** |
| E1 | `q25_stage2_dual` | Q2.5 冷启动 (1200) | 开 | `ARM=q25_dual bash scripts/launch_stage2_arm.sh` | **P0** |
| E7 | `q25_stage2_plain_hint` | Q2.5 base + 回顾句 | 关 | `ARM=q25_hint_plain bash scripts/launch_stage2_arm.sh` | **P0.5** |
| E8 | `q25_stage2_dual_hint` | Q2.5 base + 回顾句 | 开 | `ARM=q25_hint_dual bash scripts/launch_stage2_arm.sh` | **P0.5** |
| E3 | `q3_stage2_dual` | Q3 冷启动 (6220) | 开 | `ARM=q3_dual bash scripts/launch_stage2_arm.sh` | P1 |
| E4 | `q3_stage2_pure` | Q3 冷启动 (6220) | 关 | `ARM=q3_pure bash scripts/launch_stage2_arm.sh` | P1 |
| E6 | `rwml_grpo_merged10k` | Q2.5 base | —（RWML 奖励） | 见 §4 | P2 |

（第 6 臂 `ARM=q3_plain` = Q3 plain-from-base，P3 对照。）
E7/E8（2026-08-18 新增）：base 模型带 stage2 "回顾历史"句的两个臂。该句对
base 起点值 +0.10（0.125→0.227，机制是诱发长复盘、带来 30+ 步游荡式晚胜，
代价 46% 步撞 512 响应上限），对冷启动无影响。E7 测"指令红利之上纯 GRPO
还能涨多少"，E8 测 futile 惩罚与该指令的相互作用；与 E5（无句 base）对照可
分离"指令 vs RL vs 惩罚"三者贡献。起点基线：E7/E8 = 0.227（v2 实测），
E5 = 0.125。冷启动 ckpt 路径
默认指向本机 `outputs/` 树，远端机器用 `Q25_COLDSTART=`/`Q3_COLDSTART=` 或
`MODEL_PATH=erv1n/...` 覆盖。所有臂默认 `PROMPT_FORMAT=verl`、`STAGE2_PROMPT=false`。
**三条 Qwen2.5 臂（E5→E2→E1）是最高优先级**：它们构成完整因果链
base+RL vs 冷启动+RL vs 冷启动+RL+惩罚，一次隔离出 Stage 1 与惩罚项
各自的净效应，是论文的核心消融。只有一台 4 卡机时按 E5→E2→E1 顺序跑；
有多台时三条并行。
（Q3 plain-from-base 曾在原机器用旧 verl 模板跑过一版——prompt 格式切换后
该结果与新臂不可比，仅作历史参考；如需对照，用 `ARM=q3_plain` 重跑，标 P3。
**同理，任何已在远端用旧格式开跑的 P1 臂都要停掉，换新 patch/镜像重跑。**）

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
   训练与训练内 val 共用 verl 模板；离线 `evaluate-sr` 是 Stage-1 的 e2l
   格式（Stage-1 格式），不存在跨格式换算。
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
