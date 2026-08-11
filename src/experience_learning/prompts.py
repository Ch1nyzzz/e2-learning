from __future__ import annotations

from experience_learning.types import EpisodeContext

WORLD_MODEL_SYSTEM_PROMPT = """You are a world model for a deterministic text environment.
Given the interaction history and one proposed action, predict only the immediate environment
observation caused by that action. Do not choose a different action. Do not explain, plan, or
mention uncertainty. Return exactly one prediction inside <observation>...</observation>."""


def render_transcript(context: EpisodeContext) -> str:
    parts = [f"INITIAL OBSERVATION:\n{context.initial_observation}"]
    for index, step in enumerate(context.history, start=1):
        parts.append(f"STEP {index}\nACTION: {step.action}\nOBSERVATION: {step.observation}")
    return "\n\n".join(parts)


def prediction_messages(context: EpisodeContext, action: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": WORLD_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": f"{render_transcript(context)}\n\nPROPOSED ACTION:\n{action}"},
    ]


def training_target(observation: str) -> str:
    return f"<observation>{observation.strip()}</observation>"


def extract_observation(text: str) -> str:
    cleaned = text.strip()
    open_tag = cleaned.find("<observation>")
    close_tag = cleaned.find("</observation>")
    if open_tag >= 0 and close_tag > open_tag:
        return cleaned[open_tag + len("<observation>") : close_tag].strip()
    return cleaned

