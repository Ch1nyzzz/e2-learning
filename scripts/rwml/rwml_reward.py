"""veRL custom reward function for RWML GRPO on ALFWorld.

Loaded via custom_reward_function.path/name; the naive reward manager calls
compute_score(...) once per generated sample. r = 1.0 if the <next_state>
prediction is within cosine distance RWML_TAU_D of the ground-truth next state
under Qwen3-Embedding (paper Appendix B.1: training tau_d = 0.2 for ALFWorld),
else 0.0. Missing/malformed <next_state> tags score 0.0.

Embedding lookups go through the local embed_server.py; RWML_EMBED_URL points
at it. A tiny in-process cache collapses repeated (pred, gold) pairs inside a
GRPO group.
"""

from __future__ import annotations

import os
from functools import lru_cache

import requests

_TAU_D = float(os.environ.get("RWML_TAU_D", "0.2"))
_EMBED_URL = os.environ.get("RWML_EMBED_URL", "http://127.0.0.1:8901")


def _extract_next_state(text: str) -> str | None:
    cleaned = text.strip()
    open_tag = cleaned.rfind("<next_state>")
    if open_tag < 0:
        return None
    close_tag = cleaned.find("</next_state>", open_tag + len("<next_state>"))
    if close_tag < 0:
        return None
    return cleaned[open_tag + len("<next_state>") : close_tag].strip()


@lru_cache(maxsize=65536)
def _distance(prediction: str, reference: str) -> float:
    response = requests.post(
        f"{_EMBED_URL}/distances",
        json={"predictions": [prediction], "references": [reference]},
        timeout=120,
    )
    response.raise_for_status()
    return float(response.json()["distances"][0])


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    del data_source, extra_info
    prediction = _extract_next_state(solution_str or "")
    if not prediction:
        return 0.0
    reference = str(ground_truth).strip()
    if not reference:
        return 0.0
    if prediction == reference:
        return 1.0
    return 1.0 if _distance(prediction, reference) < _TAU_D else 0.0
