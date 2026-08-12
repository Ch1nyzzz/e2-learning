from __future__ import annotations

import re

from experience_learning.types import EpisodeContext

WORLD_MODEL_SYSTEM_PROMPT = """You are a world model for a deterministic text environment.
Given the interaction history and one proposed action, predict only the immediate environment
observation caused by that action. Do not choose a different action. Do not explain, plan, or
mention uncertainty. Return exactly one prediction inside <observation>...</observation>."""

POLICY_SYSTEM_PROMPT = """You are an agent solving a household task in the deterministic
ALFWorld text environment. Use the task, observations, and action history to make progress toward
the goal. Choose exactly one command from the supplied admissible actions. Briefly reason inside
<think>...</think>, then return the command verbatim inside <action>...</action>. Do not predict the
next observation and do not return more than one action."""


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


def policy_messages(
    context: EpisodeContext, admissible_actions: tuple[str, ...]
) -> list[dict[str, str]]:
    action_list = "\n".join(f"- {action}" for action in admissible_actions)
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{render_transcript(context)}\n\nADMISSIBLE ACTIONS:\n{action_list}\n\n"
                "Select the single best next action."
            ),
        },
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


def extract_policy_action(text: str, admissible_actions: tuple[str, ...]) -> str | None:
    """Accept only an action explicitly present in the current environment action set."""
    cleaned = text.strip()
    open_tag = cleaned.rfind("<action>")
    close_tag = cleaned.find("</action>", open_tag + len("<action>"))
    if open_tag >= 0 and close_tag > open_tag:
        candidate = cleaned[open_tag + len("<action>") : close_tag].strip()
    else:
        candidate = ""
        for line in reversed(cleaned.splitlines()):
            if line.strip().lower().startswith("act:"):
                candidate = line.split(":", 1)[1].strip()
                break
    normalized = candidate.casefold().strip().rstrip(".")
    matches = [
        action
        for action in admissible_actions
        if action.casefold().strip().rstrip(".") == normalized
    ]
    if len(matches) == 1:
        return matches[0]

    # Reasoning models can consume their budget immediately after </think> despite an explicit
    # action tag instruction. Recover only an exact admissible command mentioned in the generated
    # response, choosing its last occurrence. The action list itself is in the prompt, not `text`.
    mentioned: list[tuple[int, str]] = []
    folded = cleaned.casefold()
    for action in admissible_actions:
        pattern = rf"(?<![\w]){re.escape(action.casefold())}(?![\w])"
        mentioned.extend((match.start(), action) for match in re.finditer(pattern, folded))
    return max(mentioned, default=(-1, None), key=lambda item: item[0])[1]
