# Agentic world-model references

Primary papers most relevant to the ALFWorld environment-learning experiments in this repository.
PDF snapshots are tracked under `references/papers/` in this private repository. Use the canonical
arXiv links below when sharing or citing a paper.

## Priority 0: direct comparisons and required controls

| Paper | Local PDF | Why it matters here |
| --- | --- | --- |
| [Reinforcement World Model Learning for LLM-based Agents (RWML)](https://arxiv.org/abs/2602.05842) | `papers/01-rwml-2602.05842.pdf` | Closest two-stage comparison: world-model GRPO, Policy GRPO, and their composition. Reproduce its `Policy RL` and `WM + Policy RL` comparison under matched data and compute. |
| [Policy and World Modeling Co-Training for Language Agents (PaW)](https://arxiv.org/abs/2606.02388) | `papers/02-paw-2606.02388.pdf` | Closest co-training comparison. It combines Policy RL with an auxiliary next-observation objective and uses action-entropy data selection. |
| [The Dark Room in the Reward Channel](https://arxiv.org/abs/2607.21273) | `papers/03-dark-room-2607.21273.pdf` | Motivates keeping prediction supervision out of normalized task advantage. Its shuffled-gold placebo motivates matched target, token-budget, and update-count controls. |

## Priority 1: alternative training interfaces and targets

| Paper | Local PDF | Why it matters here |
| --- | --- | --- |
| [TAPO: Transition-Aware Policy Optimization for LLM Agents](https://arxiv.org/abs/2607.27973) | `papers/04-tapo-2607.27973.pdf` | Alternates task-policy updates with action-conditioned next-observation supervision instead of mixing both losses in one update. |
| [ECHO: Terminal Agents Learn World Models for Free](https://arxiv.org/abs/2605.24517) | `papers/05-echo-2605.24517.pdf` | Routes policy-gradient loss to action tokens and environment-prediction cross-entropy to observation tokens in the same on-policy training stream. |
| [Beyond State Consistency: Behavior Consistency in Text-Based World Models](https://arxiv.org/abs/2604.13824) | `papers/06-behr-2604.13824.pdf` | Replaces surface state similarity with behavior consistency under a frozen reference policy; useful for evaluating whether a prediction preserves action-relevant information. |

## Baseline mapping for this repository

The minimum matched comparison should include:

1. Base Qwen2.5-7B-Instruct ReAct.
2. True next-observation SFT (current method).
3. Shuffled-observation SFT with matched updates and token budget.
4. Random acquisition instead of token-entropy acquisition.
5. All-transition SFT instead of the semantic mistake gate.
6. Base to Policy GRPO.
7. Current world-model checkpoint to the same Policy GRPO.

RWML is the primary algorithmic baseline; PaW and TAPO are the main training-interface
alternatives; Dark Room supplies the most important causal controls; BehR supplies the strongest
alternative prediction target.
