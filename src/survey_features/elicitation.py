"""
Selector elicitation: build free-text chat messages for one selection call.

The raw essay is cached verbatim and later parsed by ``extraction`` (fixed model).
No structure is imposed on the selector's output.
"""

from __future__ import annotations

from .prompts import (
    DEFAULT_PROMPT_ARM,
    FREETEXT_COUNTRY,
    FREETEXT_UNPROMPTED,
    PROMPT_ARMS,
)


def system_prompt_for_arm(arm: str | None = None) -> str | None:
    """Resolve a prompt-arm key to system content (None = omit system message)."""
    key = DEFAULT_PROMPT_ARM if arm is None else arm
    if key not in PROMPT_ARMS:
        raise ValueError(f"Unknown prompt arm {key!r}; choose from {sorted(PROMPT_ARMS)}")
    return PROMPT_ARMS[key]


def freetext_messages(
    question_text: str,
    country: str | None = None,
    *,
    prompt_arm: str | None = None,
) -> list[dict]:
    """Chat messages for one free-text selection call (country=None -> unprompted).

    ``prompt_arm`` selects the system message (default: social_scientist).
    Arm ``none`` omits the system role entirely.
    """
    if country:
        user_msg = FREETEXT_COUNTRY.format(question=question_text, country=country)
    else:
        user_msg = FREETEXT_UNPROMPTED.format(question=question_text)
    messages: list[dict] = []
    system = system_prompt_for_arm(prompt_arm)
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    return messages
