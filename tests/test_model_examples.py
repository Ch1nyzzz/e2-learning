import math

import pytest

from experience_learning.model import (
    TransformersWorldModel,
    build_causal_training_sequence,
    mean_generated_token_entropy,
)
from experience_learning.types import EpisodeContext, Experience


class WhitespaceTokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, **_kwargs) -> str:
        return " ".join(f"{item['role']} {item['content']}" for item in messages)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, tokens, *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(tokens)


def test_prompt_tokens_are_masked() -> None:
    input_ids, labels = build_causal_training_sequence(
        [1, 2, 3], [4, 5], eos_token_id=6, max_tokens=8
    )
    assert input_ids == [1, 2, 3, 4, 5, 6]
    assert labels == [-100, -100, -100, 4, 5, 6]


def test_long_prompt_is_left_truncated_but_target_is_preserved() -> None:
    input_ids, labels = build_causal_training_sequence(
        [1, 2, 3, 4], [5, 6], eos_token_id=7, max_tokens=5
    )
    assert input_ids == [3, 4, 5, 6, 7]
    assert labels == [-100, -100, 5, 6, 7]


def test_mean_token_entropy_includes_first_eos_and_excludes_later_padding() -> None:
    torch = pytest.importorskip("torch")
    logits = (
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [100.0, -100.0]]),
    )
    generated_tokens = torch.tensor([[1, 1], [0, 1]])

    entropies = mean_generated_token_entropy(
        logits,
        generated_tokens,
        eos_token_id=1,
    )

    assert entropies[0].item() == pytest.approx(math.log(2))
    assert entropies[1].item() == pytest.approx(math.log(2) / 2)


def test_structured_prompt_packing_keeps_latest_state_and_action() -> None:
    model = TransformersWorldModel.__new__(TransformersWorldModel)
    model.tokenizer = WhitespaceTokenizer()
    context = EpisodeContext("task goal")
    context.append("old action", "obsolete state")
    context.append("new action", "latest state")
    latest_only = EpisodeContext("task goal", history=[context.history[-1]])
    budget = len(
        model.tokenizer.encode(
            model._chat_prompt(latest_only, "open fridge"), add_special_tokens=False
        )
    )

    prompt = model._fit_chat_prompt(context, "open fridge", max_tokens=budget)

    assert "obsolete state" not in prompt
    assert "latest state" in prompt
    assert "open fridge" in prompt
    assert "world model" in prompt


def test_rwml_wm_sft_profile_uses_empty_thinking_and_next_state_tags() -> None:
    model = TransformersWorldModel.__new__(TransformersWorldModel)
    model.tokenizer = WhitespaceTokenizer()
    model.config = type(
        "Config",
        (),
        {
            "training": type("Training", (), {"prompt_profile": "rwml_wm_sft"})(),
            "model": type("Model", (), {"max_context_tokens": 256})(),
        },
    )()
    experience = Experience(EpisodeContext("start"), "open fridge", "", "opened", 0, 0)

    input_ids, labels = model._encode_training_example(experience)
    trained_tokens = [
        token for token, label in zip(input_ids, labels, strict=True) if label != -100
    ]

    assert "<think>" in trained_tokens
    assert any("<next_state>opened</next_state>" == token for token in trained_tokens)
