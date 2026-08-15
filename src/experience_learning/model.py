from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from experience_learning.config import AppConfig
from experience_learning.distributed import all_gather_objects
from experience_learning.prompts import (
    extract_observation,
    extract_policy_action,
    policy_messages,
    prediction_messages,
    rwml_wm_sft_prediction_messages,
    rwml_wm_sft_training_target,
    training_target,
)
from experience_learning.types import (
    EpisodeContext,
    Experience,
    PolicyDecision,
    TokenEntropyPrediction,
)


class _CheckpointStateSink:
    """Accept scheduler state while loading a model-only evaluation checkpoint."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


def _load_model_only_accelerator_state(accelerator: Any, checkpoint: str) -> None:
    """Restore an Accelerate checkpoint without applying its optimizer state.

    FSDP2 requires an optimizer during ``prepare`` even for inference, so Accelerate
    registers the throwaway preparation optimizer. Temporarily hiding that optimizer
    lets ``load_state`` restore the sharded model while skipping the incompatible
    training optimizer checkpoint.
    """
    optimizers = accelerator._optimizers
    accelerator._optimizers = []
    try:
        accelerator.load_state(checkpoint)
    finally:
        accelerator._optimizers = optimizers


def _is_hf_checkpoint(path: str | Path) -> bool:
    """Return whether ``path`` contains a loadable Hugging Face model export."""
    checkpoint = Path(path)
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        return False
    return any(checkpoint.glob("*.safetensors")) or any(checkpoint.glob("*.bin"))


def valid_generated_token_mask(
    generated_tokens: Any,
    *,
    eos_token_id: int | Sequence[int] | None,
) -> tuple[Any, Any]:
    """Return the through-first-EOS token mask and whether each sequence emitted EOS."""
    import torch

    valid = torch.ones_like(generated_tokens, dtype=torch.bool)
    has_eos = torch.zeros(
        generated_tokens.shape[0], dtype=torch.bool, device=generated_tokens.device
    )
    if eos_token_id is not None:
        eos_ids = (
            list(eos_token_id)
            if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, (str, bytes))
            else [eos_token_id]
        )
        is_eos = torch.zeros_like(generated_tokens, dtype=torch.bool)
        for token_id in eos_ids:
            is_eos |= generated_tokens.eq(token_id)
        has_eos = is_eos.any(dim=1)
        prior_eos = is_eos.cumsum(dim=1) - is_eos.to(dtype=torch.long)
        valid &= prior_eos.eq(0)
    return valid, has_eos


def mean_generated_token_entropy(
    logits: Sequence[Any],
    generated_tokens: Any,
    *,
    eos_token_id: int | Sequence[int] | None,
) -> Any:
    """Average full-vocabulary entropy through the first EOS for each sequence."""
    import torch

    if not logits:
        raise ValueError("token entropy requires at least one generation logit tensor")
    step_entropies = torch.stack(
        [torch.special.entr(step.float().softmax(dim=-1)).sum(dim=-1) for step in logits],
        dim=1,
    )
    valid, _ = valid_generated_token_mask(
        generated_tokens,
        eos_token_id=eos_token_id,
    )
    valid_float = valid.to(dtype=step_entropies.dtype)
    return (step_entropies * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1)


def build_causal_training_sequence(
    prompt_ids: list[int],
    target_ids: list[int],
    *,
    eos_token_id: int | None,
    max_tokens: int,
) -> tuple[list[int], list[int]]:
    """Create a causal-LM example whose loss is zero on every prompt token."""
    target = list(target_ids)
    if eos_token_id is not None:
        target.append(eos_token_id)
    if len(target) >= max_tokens:
        target = target[:max_tokens]
        if eos_token_id is not None:
            target[-1] = eos_token_id
        prompt = []
    else:
        prompt = prompt_ids[-(max_tokens - len(target)) :]
    return prompt + target, [-100] * len(prompt) + target


class TransformersWorldModel:
    """Fully trainable causal LM prepared by Accelerate and FSDP."""

    def __init__(self, config: AppConfig, *, training: bool = True):
        import torch
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs, set_seed
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.accelerator = Accelerator(
            kwargs_handlers=[
                InitProcessGroupKwargs(
                    timeout=timedelta(seconds=config.model.distributed_timeout_seconds)
                )
            ]
        )
        set_seed(config.experiment.seed, device_specific=True)
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.training_enabled = training
        self._optimizer_step = 0
        resume_path = (
            Path(config.training.resume_from) if config.training.resume_from else None
        )
        hf_checkpoint = resume_path is not None and _is_hf_checkpoint(resume_path)
        if hf_checkpoint and training:
            raise ValueError(
                "HF-only checkpoints contain model weights but no optimizer state; "
                "use them for evaluation or start a new training output directory"
            )
        model_source = str(resume_path) if hf_checkpoint else config.model.name
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(config.model.dtype)
        if dtype is None:
            raise ValueError(f"unsupported model dtype: {config.model.dtype}")

        common_kwargs: dict[str, Any] = {
            "trust_remote_code": config.model.trust_remote_code,
        }
        if config.model.revision and not hf_checkpoint:
            common_kwargs["revision"] = config.model.revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **common_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            **common_kwargs,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if config.model.attn_implementation:
            model_kwargs["attn_implementation"] = config.model.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        model.config.use_cache = False
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        frozen = [
            name for name, parameter in model.named_parameters() if not parameter.requires_grad
        ]
        if frozen:
            raise RuntimeError(f"full fine-tuning found frozen parameters: {frozen[:5]}")
        if any("lora" in name.lower() for name, _ in model.named_parameters()):
            raise RuntimeError("LoRA parameters detected in a full-fine-tuning experiment")

        if training:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                betas=(config.training.adam_beta1, config.training.adam_beta2),
                eps=1.0e-8,
                weight_decay=config.training.weight_decay,
                foreach=False,
                fused=False,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_multiplier)
            self.model, self.optimizer = self.accelerator.prepare(model, optimizer)
            # Do not wrap this scheduler in AcceleratedScheduler: for data-parallel training that
            # wrapper can advance once per process. We need exactly one LR step per optimizer
            # update.
            self.scheduler = scheduler
            self.accelerator.register_for_checkpointing(self.scheduler)
        else:
            # Accelerate FSDP2 rebuilds parameter objects while sharding and therefore requires
            # a model and optimizer to be prepared together, even for evaluation-only jobs. SGD
            # has no state until a step occurs; it is used only to let Accelerate remap parameters
            # and is discarded immediately after preparation.
            preparation_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            self.model, _ = self.accelerator.prepare(model, preparation_optimizer)
            self.model.eval()
            self.optimizer = None
            self.scheduler = None
            self.accelerator.register_for_checkpointing(_CheckpointStateSink())
        if config.training.resume_from and not hf_checkpoint:
            if training:
                self.accelerator.load_state(config.training.resume_from)
            else:
                _load_model_only_accelerator_state(
                    self.accelerator, config.training.resume_from
                )
            controller_path = Path(config.training.resume_from) / "controller_state.json"
            if controller_path.exists():
                controller = json.loads(controller_path.read_text(encoding="utf-8"))
                self._optimizer_step = int(controller.get("optimizer_step", 0))
                if (
                    training
                    and self.scheduler is not None
                    and self.scheduler.last_epoch != self._optimizer_step
                ):
                    raise RuntimeError(
                        "checkpoint scheduler/controller mismatch: "
                        f"scheduler={self.scheduler.last_epoch}, "
                        f"controller={self._optimizer_step}"
                    )

    def _lr_multiplier(self, update: int) -> float:
        warmup = self.config.training.warmup_updates
        if warmup > 0 and update < warmup:
            return max(1.0e-8, (update + 1) / warmup)
        return 1.0

    def _chat_prompt(self, context: EpisodeContext, action: str) -> str:
        profile = getattr(
            getattr(getattr(self, "config", None), "training", None),
            "prompt_profile",
            "default",
        )
        if profile == "rwml_wm_sft":
            messages = rwml_wm_sft_prediction_messages(context, action)
        else:
            messages = prediction_messages(context, action)
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _fit_chat_prompt(
        self, context: EpisodeContext, action: str, *, max_tokens: int
    ) -> str:
        """Keep the instruction, initial state, newest history, and proposed action in context."""
        if max_tokens < 1:
            raise ValueError("prompt token budget must be positive")

        def encode(prompt_context: EpisodeContext) -> tuple[str, int]:
            prompt = self._chat_prompt(prompt_context, action)
            length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            return prompt, length

        retained_history = list(context.history)
        while True:
            packed = EpisodeContext(
                initial_observation=context.initial_observation,
                history=list(retained_history),
            )
            prompt, length = encode(packed)
            if length <= max_tokens:
                return prompt
            if retained_history:
                retained_history.pop(0)
                continue
            break

        initial_ids = self.tokenizer.encode(
            context.initial_observation, add_special_tokens=False
        )
        best: str | None = None
        lower, upper = 0, len(initial_ids)
        while lower <= upper:
            retained = (lower + upper) // 2
            head = (retained + 1) // 2
            tail = retained // 2
            kept_ids = initial_ids[:head]
            if tail:
                kept_ids += initial_ids[-tail:]
            shortened = self.tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
            if retained < len(initial_ids):
                shortened = f"{shortened}\n[older initial-state text truncated]".strip()
            candidate = EpisodeContext(initial_observation=shortened)
            prompt, length = encode(candidate)
            if length <= max_tokens:
                best = prompt
                lower = retained + 1
            else:
                upper = retained - 1
        if best is None:
            raise ValueError(
                "model.max_context_tokens is too small to preserve the system prompt and action"
            )
        return best

    def _fit_policy_prompt(
        self,
        context: EpisodeContext,
        admissible_actions: tuple[str, ...],
        *,
        max_tokens: int,
    ) -> str:
        def encode(prompt_context: EpisodeContext) -> tuple[str, int]:
            messages = policy_messages(prompt_context, admissible_actions)
            kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, enable_thinking=True, **kwargs
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
            return prompt, len(self.tokenizer.encode(prompt, add_special_tokens=False))

        retained_history = list(context.history)
        while True:
            packed = EpisodeContext(
                initial_observation=context.initial_observation,
                history=list(retained_history),
            )
            prompt, length = encode(packed)
            if length <= max_tokens:
                return prompt
            if retained_history:
                retained_history.pop(0)
                continue
            raise ValueError(
                "model.max_context_tokens is too small for the policy prompt and action list"
            )

    @staticmethod
    def _balanced_slice(total: int, world_size: int, rank: int) -> tuple[int, int, int]:
        per_rank = max(1, math.ceil(total / world_size))
        return rank * per_rank, (rank + 1) * per_rank, per_rank

    def predict(
        self,
        requests: Sequence[tuple[EpisodeContext, str]],
        *,
        samples_per_request: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> list[list[str]]:
        predictions, _, _, _ = self._predict(
            requests,
            samples_per_request=samples_per_request,
            do_sample=do_sample,
            seed=seed,
            return_token_entropy=False,
        )
        return predictions

    def predict_with_token_entropy(
        self,
        requests: Sequence[tuple[EpisodeContext, str]],
        *,
        seed: int | None = None,
    ) -> list[TokenEntropyPrediction]:
        predictions, entropies, token_counts, hit_token_limits = self._predict(
            requests,
            samples_per_request=1,
            do_sample=False,
            seed=seed,
            return_token_entropy=True,
        )
        assert entropies is not None
        assert token_counts is not None
        assert hit_token_limits is not None
        return [
            TokenEntropyPrediction(
                observation=outcomes[0],
                mean_token_entropy=entropy,
                generated_tokens=token_count,
                hit_token_limit=hit_token_limit,
            )
            for outcomes, entropy, token_count, hit_token_limit in zip(
                predictions,
                entropies,
                token_counts,
                hit_token_limits,
                strict=True,
            )
        ]

    def choose_actions(
        self,
        requests: Sequence[tuple[EpisodeContext, tuple[str, ...]]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> list[PolicyDecision]:
        import torch

        if not requests:
            return []
        if max_new_tokens < 1 or max_new_tokens >= self.config.model.max_context_tokens:
            raise ValueError("policy max_new_tokens must fit inside the model context")
        if seed is not None:
            torch.manual_seed(seed + self.rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + self.rank)
        start, end, per_rank = self._balanced_slice(len(requests), self.world_size, self.rank)
        indexed = list(enumerate(requests))
        while len(indexed) < per_rank * self.world_size:
            indexed.append((-1, requests[0]))
        local = indexed[start:end]
        prompt_budget = self.config.model.max_context_tokens - max_new_tokens
        prompts = [
            self._fit_policy_prompt(context, actions, max_tokens=prompt_budget)
            for _, (context, actions) in local
        ]
        old_padding_side = self.tokenizer.padding_side
        old_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.config.model.gradient_checkpointing:
            unwrapped.gradient_checkpointing_disable()
        local_results: list[tuple[int, str, str | None]] = []
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "synced_gpus": self.world_size > 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
            )
        try:
            micro_batch_size = self.config.generation.micro_batch_size
            for chunk_start in range(0, len(local), micro_batch_size):
                chunk_local = local[chunk_start : chunk_start + micro_batch_size]
                encoded = self.tokenizer(
                    prompts[chunk_start : chunk_start + micro_batch_size],
                    padding=True,
                    return_tensors="pt",
                ).to(self.accelerator.device)
                with torch.no_grad():
                    generated = unwrapped.generate(**encoded, **generation_kwargs)
                prompt_length = encoded["input_ids"].shape[1]
                decoded = self.tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
                for (global_index, (_, actions)), raw_response in zip(
                    chunk_local, decoded, strict=True
                ):
                    if global_index >= 0:
                        local_results.append(
                            (
                                global_index,
                                raw_response.strip(),
                                extract_policy_action(raw_response, actions),
                            )
                        )
        finally:
            if self.config.model.gradient_checkpointing:
                unwrapped.gradient_checkpointing_enable()
            if self.training_enabled:
                self.model.train()
            else:
                self.model.eval()
            self.tokenizer.padding_side = old_padding_side
            self.tokenizer.truncation_side = old_truncation_side
        flattened = [
            item for rank_items in all_gather_objects(local_results) for item in rank_items
        ]
        flattened.sort(key=lambda item: item[0])
        if len(flattened) != len(requests):
            raise RuntimeError(
                f"distributed policy generation lost requests: expected {len(requests)}, "
                f"got {len(flattened)}"
            )
        return [
            PolicyDecision(raw_response=raw, action=action)
            for _, raw, action in flattened
        ]

    def _predict(
        self,
        requests: Sequence[tuple[EpisodeContext, str]],
        *,
        samples_per_request: int,
        do_sample: bool,
        seed: int | None = None,
        return_token_entropy: bool,
    ) -> tuple[
        list[list[str]],
        list[float] | None,
        list[int] | None,
        list[bool] | None,
    ]:
        import torch

        if not requests:
            empty = [] if return_token_entropy else None
            return [], empty, empty, empty
        if return_token_entropy and (samples_per_request != 1 or do_sample):
            raise ValueError("token entropy prediction requires one greedy sequence per request")
        if seed is not None:
            torch.manual_seed(seed + self.rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + self.rank)
        start, end, per_rank = self._balanced_slice(len(requests), self.world_size, self.rank)
        indexed = list(enumerate(requests))
        padded_total = per_rank * self.world_size
        while len(indexed) < padded_total:
            indexed.append((-1, requests[0]))
        local = indexed[start:end]
        prompt_budget = max(
            1, self.config.model.max_context_tokens - self.config.generation.max_new_tokens
        )
        prompts = [
            self._fit_chat_prompt(context, action, max_tokens=prompt_budget)
            for _, (context, action) in local
        ]

        old_padding_side = self.tokenizer.padding_side
        old_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.config.model.gradient_checkpointing:
            unwrapped.gradient_checkpointing_disable()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.generation.max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": samples_per_request,
            "use_cache": True,
            "synced_gpus": self.world_size > 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
            )
        if return_token_entropy:
            generation_kwargs.update(return_dict_in_generate=True, output_logits=True)
        local_results: list[tuple[int, list[str], float | None, int | None, bool | None]] = []
        micro_batch_size = self.config.generation.micro_batch_size
        try:
            for chunk_start in range(0, len(local), micro_batch_size):
                chunk_local = local[chunk_start : chunk_start + micro_batch_size]
                chunk_prompts = prompts[chunk_start : chunk_start + micro_batch_size]
                encoded = self.tokenizer(
                    chunk_prompts, padding=True, return_tensors="pt"
                ).to(self.accelerator.device)
                if encoded["input_ids"].shape[1] > prompt_budget:
                    raise RuntimeError(
                        "structured prompt packing exceeded the generation token budget"
                    )
                # FSDP2 may materialize full parameters during generation. inference_mode would
                # mark those tensors as inference-only and make a later training backward fail;
                # no_grad avoids graph construction without changing tensor provenance.
                with torch.no_grad():
                    generation_output = unwrapped.generate(**encoded, **generation_kwargs)
                prompt_length = encoded["input_ids"].shape[1]
                generated = (
                    generation_output.sequences
                    if return_token_entropy
                    else generation_output
                )
                decoded = self.tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
                mean_entropies: list[float] | None = None
                generated_token_counts: list[int] | None = None
                hit_token_limits: list[bool] | None = None
                if return_token_entropy:
                    logits = generation_output.logits
                    if not logits:
                        raise RuntimeError("generation returned no token logits")
                    generated_tokens = generated[
                        :, prompt_length : prompt_length + len(logits)
                    ]
                    mean_entropies = mean_generated_token_entropy(
                        logits,
                        generated_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                    ).tolist()
                    valid_tokens, has_eos = valid_generated_token_mask(
                        generated_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    generated_token_counts = valid_tokens.sum(dim=1).tolist()
                    hit_token_limits = (
                        ~has_eos & (len(logits) >= self.config.generation.max_new_tokens)
                    ).tolist()
                for chunk_index, (global_index, _) in enumerate(chunk_local):
                    if global_index < 0:
                        continue
                    offset = chunk_index * samples_per_request
                    outcomes = [
                        extract_observation(text)
                        for text in decoded[offset : offset + samples_per_request]
                    ]
                    entropy = None if mean_entropies is None else mean_entropies[offset]
                    token_count = (
                        None if generated_token_counts is None else generated_token_counts[offset]
                    )
                    hit_token_limit = (
                        None if hit_token_limits is None else hit_token_limits[offset]
                    )
                    local_results.append(
                        (global_index, outcomes, entropy, token_count, hit_token_limit)
                    )
        finally:
            if self.config.model.gradient_checkpointing:
                unwrapped.gradient_checkpointing_enable()
            self.model.train()
            self.tokenizer.padding_side = old_padding_side
            self.tokenizer.truncation_side = old_truncation_side

        gathered = all_gather_objects(local_results)
        flattened = [item for rank_items in gathered for item in rank_items]
        flattened.sort(key=lambda item: item[0])
        if len(flattened) != len(requests):
            raise RuntimeError(
                f"distributed generation lost requests: expected {len(requests)}, "
                f"got {len(flattened)}"
            )
        predictions = [outcomes for _, outcomes, _, _, _ in flattened]
        entropies = (
            [float(entropy) for _, _, entropy, _, _ in flattened if entropy is not None]
            if return_token_entropy
            else None
        )
        if return_token_entropy and len(entropies or []) != len(predictions):
            raise RuntimeError("distributed generation lost token entropy scores")
        token_counts = (
            [int(count) for _, _, _, count, _ in flattened if count is not None]
            if return_token_entropy
            else None
        )
        hit_token_limits = (
            [bool(hit) for _, _, _, _, hit in flattened if hit is not None]
            if return_token_entropy
            else None
        )
        if return_token_entropy and (
            len(token_counts or []) != len(predictions)
            or len(hit_token_limits or []) != len(predictions)
        ):
            raise RuntimeError("distributed generation lost token-limit diagnostics")
        return predictions, entropies, token_counts, hit_token_limits

    def _encode_training_example(self, experience: Experience) -> tuple[list[int], list[int]]:
        profile = getattr(self.config.training, "prompt_profile", "default")
        target = (
            rwml_wm_sft_training_target(experience.actual_observation)
            if profile == "rwml_wm_sft"
            else training_target(experience.actual_observation)
        )
        target_ids = self.tokenizer.encode(
            target, add_special_tokens=False
        )
        reserved_target_tokens = len(target_ids) + int(self.tokenizer.eos_token_id is not None)
        prompt_budget = self.config.model.max_context_tokens - reserved_target_tokens
        if prompt_budget > 0:
            prompt = self._fit_chat_prompt(
                experience.context,
                experience.action,
                max_tokens=prompt_budget,
            )
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = []
        return build_causal_training_sequence(
            prompt_ids,
            target_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            max_tokens=self.config.model.max_context_tokens,
        )

    def _collate(self, experiences: Sequence[Experience]) -> dict[str, Any]:
        import torch

        encoded = [self._encode_training_example(item) for item in experiences]
        maximum = max(len(input_ids) for input_ids, _ in encoded)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for input_ids, labels in encoded:
            padding = maximum - len(input_ids)
            input_rows.append(input_ids + [self.tokenizer.pad_token_id] * padding)
            label_rows.append(labels + [-100] * padding)
            attention_rows.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long, device=self.accelerator.device),
            "labels": torch.tensor(label_rows, dtype=torch.long, device=self.accelerator.device),
            "attention_mask": torch.tensor(
                attention_rows, dtype=torch.long, device=self.accelerator.device
            ),
        }

    def learn(self, experiences: Sequence[Experience]) -> dict[str, float]:
        import torch

        if not self.training_enabled or self.optimizer is None or self.scheduler is None:
            raise RuntimeError("this model was initialized for evaluation only")
        if not experiences:
            raise ValueError("learn requires at least one experience")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=self.accelerator.device)
        accumulated_target_tokens = torch.zeros((), device=self.accelerator.device)
        accumulation = self.config.training.gradient_accumulation_steps
        for index in range(accumulation):
            selected = [
                experiences[(self.rank * accumulation + index + offset) % len(experiences)]
                for offset in range(self.config.training.micro_batch_size)
            ]
            batch = self._collate(selected)
            accumulated_target_tokens += (batch["labels"] != -100).sum()
            output = self.model(**batch, use_cache=False)
            loss = output.loss / accumulation
            self.accelerator.backward(loss)
            accumulated_loss += loss.detach()
        self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.config.training.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._optimizer_step += 1
        mean_loss = self.accelerator.reduce(accumulated_loss, reduction="mean").item()
        target_tokens = self.accelerator.reduce(
            accumulated_target_tokens, reduction="sum"
        ).item()
        return {
            "loss": float(mean_loss),
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
            "optimizer_step": float(self._optimizer_step),
            "target_tokens": float(target_tokens),
        }

    def score(self, experiences: Sequence[Experience]) -> dict[str, float]:
        import torch

        if not experiences:
            raise ValueError("score requires at least one experience")
        self.model.eval()
        batch = self._collate(experiences)
        target_tokens = (batch["labels"] != -100).sum().to(dtype=torch.float32)
        with torch.inference_mode():
            loss = self.model(**batch, use_cache=False).loss
        self.model.train()
        mean_loss = self.accelerator.reduce(loss.detach(), reduction="mean")
        mean_tokens = self.accelerator.reduce(target_tokens, reduction="mean")
        return {
            "target_nll_sum": float((mean_loss * mean_tokens).item()),
            "target_tokens": float(mean_tokens.item()),
        }

    def save(self, output_dir: str) -> None:
        if not self.training_enabled:
            raise RuntimeError("evaluation-only models cannot save training checkpoints")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        # Save only a standard HF safetensors export. The old accelerator.save_state path also
        # copied AdamW, scheduler, RNG, and FSDP optimizer shards for every checkpoint, which is
        # unnecessary for model artifacts and multiplies disk usage.
        self.accelerator.save_model(
            self.model,
            str(output),
            max_shard_size="5GB",
            safe_serialization=True,
        )
        if self.accelerator.is_main_process:
            model_to_save = self.accelerator.unwrap_model(self.model)
            model_to_save.config.save_pretrained(output)
            if getattr(model_to_save, "generation_config", None) is not None:
                model_to_save.generation_config.save_pretrained(output)
            self.tokenizer.save_pretrained(output)
        self.accelerator.wait_for_everyone()
