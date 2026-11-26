"""Judge prompts (prototype wording; see DECISIONS.md).

The judge answers with a single letter. Scoring reads the probability of
"A" and "B" as the first answer token, so no free-text parsing is needed.
Changing any wording means a new prompt ID, so old results stay comparable.
"""
from __future__ import annotations

import hashlib

from .items import Item

SYSTEM = "You are an impartial judge. You compare two AI assistant responses and decide which one is better."

QUESTION = "Which response is better? Reply with a single letter: A or B."

RUBRIC = (
    "Judge the responses on helpfulness, correctness, coherence, complexity and verbosity, in that "
    "order of importance. The order in which the responses are shown does not matter. Do not prefer "
    "a response for being longer or more formatted unless that makes it more helpful."
)

PROMPTS = {
    "baseline_v1": {"system": SYSTEM, "instructions": None},
    "rubric_v1": {"system": SYSTEM, "instructions": RUBRIC},
}

_ROLE = {"user": "User", "assistant": "Assistant"}


def _transcript(context: tuple) -> str:
    return "\n\n".join(f"{_ROLE[role]}: {content}" for role, content in context)


def render(item: Item, order: tuple[int, int], prompt_id: str) -> list[dict]:
    """Chat messages for one judgment. order=(1, 2) shows response 1 as A."""
    if sorted(order) != [1, 2]:
        raise ValueError("order must be (1, 2) or (2, 1)")
    spec = PROMPTS[prompt_id]
    side = {1: (item.context_1, item.response_1), 2: (item.context_2, item.response_2)}
    (ctx_a, resp_a), (ctx_b, resp_b) = side[order[0]], side[order[1]]
    parts = []
    if ctx_a == ctx_b:
        parts.append(f"[Conversation]\n{_transcript(ctx_a)}\n[End of Conversation]")
        parts.append(f"[Response A]\n{resp_a}\n[End of Response A]")
        parts.append(f"[Response B]\n{resp_b}\n[End of Response B]")
    else:  # MT-Bench turn 2: each side has its own earlier answer
        parts.append(f"[Conversation A]\n{_transcript(ctx_a)}\n[End of Conversation A]")
        parts.append(f"[Response A]\n{resp_a}\n[End of Response A]")
        parts.append(f"[Conversation B]\n{_transcript(ctx_b)}\n[End of Conversation B]")
        parts.append(f"[Response B]\n{resp_b}\n[End of Response B]")
        parts.append("Compare only the final responses, given each conversation.")
    if spec["instructions"]:
        parts.append(spec["instructions"])
    parts.append(QUESTION)
    return [{"role": "system", "content": spec["system"]}, {"role": "user", "content": "\n\n".join(parts)}]


def prompt_sha(prompt_id: str) -> str:
    spec = PROMPTS[prompt_id]
    return hashlib.sha256(repr((spec, QUESTION)).encode()).hexdigest()[:16]
