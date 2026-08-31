"""
Selector elicitation: build free-text chat messages for one selection call.

The raw essay is cached verbatim and later parsed by ``extraction`` (fixed model).
No structure is imposed on the selector's output.
"""

from __future__ import annotations

from .prompts import (
    DEFAULT_PROMPT_ARM,
    DEFAULT_PROMPT_PACK,
    DEFAULT_REFERENT,
    FREETEXT_BY_REFERENT,
    PROMPT_ARMS,
    PROMPT_PACKS,
)


def system_prompt_for_arm(arm: str | None = None) -> str | None:
    """Resolve a prompt-arm key to system content (None = omit system message)."""
    key = DEFAULT_PROMPT_ARM if arm is None else arm
    if key not in PROMPT_ARMS:
        raise ValueError(f"Unknown prompt arm {key!r}; choose from {sorted(PROMPT_ARMS)}")
    return PROMPT_ARMS[key]


def resolve_prompt_pack(pack: str | None = None) -> tuple[str, str]:
    """Return (system_arm, referent) for a named v2 pack."""
    key = DEFAULT_PROMPT_PACK if pack is None else pack
    if key not in PROMPT_PACKS:
        raise ValueError(f"Unknown prompt pack {key!r}; choose from {sorted(PROMPT_PACKS)}")
    spec = PROMPT_PACKS[key]
    return spec["system"], spec["referent"]


def freetext_messages(
    question_text: str,
    country: str | None = None,
    *,
    prompt_arm: str | None = None,
    referent: str | None = None,
    prompt_pack: str | None = None,
) -> list[dict]:
    """Chat messages for one free-text selection call (country=None -> unprompted).

    ``prompt_pack`` sets both system arm and user referent (v2). Otherwise
    ``prompt_arm`` selects the system message (default: social_scientist) and
    ``referent`` selects respondent vs person wording (default: respondent).
    Arm ``none`` omits the system role entirely.
    """
    if prompt_pack is not None:
        prompt_arm, referent = resolve_prompt_pack(prompt_pack)
    ref = DEFAULT_REFERENT if referent is None else referent
    if ref not in FREETEXT_BY_REFERENT:
        raise ValueError(
            f"Unknown referent {ref!r}; choose from {sorted(FREETEXT_BY_REFERENT)}"
        )
    templates = FREETEXT_BY_REFERENT[ref]
    if country:
        user_msg = templates["country"].format(question=question_text, country=country)
    else:
        user_msg = templates["unprompted"].format(question=question_text)
    messages: list[dict] = []
    system = system_prompt_for_arm(prompt_arm)
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    return messages
