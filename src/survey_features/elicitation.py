"""
Selector elicitation: build free-text chat messages for one selection call.

The raw essay is cached verbatim and later parsed by ``extraction`` (fixed model).
No structure is imposed on the selector's output.
"""

from __future__ import annotations

from .prompts import FREETEXT_COUNTRY, FREETEXT_UNPROMPTED, SYSTEM_PROMPT


def freetext_messages(question_text: str, country: str | None = None) -> list[dict]:
    """Chat messages for one free-text selection call (country=None -> unprompted)."""
    if country:
        user_msg = FREETEXT_COUNTRY.format(question=question_text, country=country)
    else:
        user_msg = FREETEXT_UNPROMPTED.format(question=question_text)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
