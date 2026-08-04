"""
Selection elicitation: ask the selector model which respondent features would predict
the answer to a target survey question.

CURRENT path (free-text): ``freetext_messages`` builds the chat messages; the raw essay
is cached verbatim and later parsed by ``survey_features.extraction`` (fixed extractor
model). No structure is imposed on the selector's output.

LEGACY path (strict JSON, pilot-1): ``run_single`` / ``run_batch`` demand a JSON list
and parse it directly. Kept runnable for appendix reproducibility (archive/run_grid.py).
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from .prompts import (
    FREETEXT_COUNTRY,
    FREETEXT_UNPROMPTED,
    PROMPT_COUNTRY,
    PROMPT_UNPROMPTED,
    SYSTEM_PROMPT,
)


# ── CURRENT: free-text elicitation ────────────────────────────────────────────

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


# ── LEGACY: strict-JSON elicitation (pilot-1) ─────────────────────────────────

def _normalize_parsed_features(parsed: object) -> list[dict] | None:
    """
    Coerce diverse LLM JSON shapes into [{feature, reasoning}, ...].
    Handles wrapped objects, list-of-strings, and alternate key names.
    """
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        for key in ("features", "items", "candidates", "answers", "result", "list", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            if any(k in parsed for k in ("feature", "reasoning", "name", "label")):
                parsed = [parsed]
            else:
                return None
    if not isinstance(parsed, list):
        return None
    out: list[dict] = []
    for item in parsed:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append({"feature": s, "reasoning": ""})
        elif isinstance(item, dict):
            feat = item.get("feature") or item.get("name") or item.get("label") or item.get("title")
            reason = (
                item.get("reasoning")
                or item.get("reason")
                or item.get("explanation")
                or item.get("rationale")
                or ""
            )
            if feat:
                out.append(
                    {
                        "feature": str(feat).strip(),
                        "reasoning": str(reason).strip() if reason else "",
                    }
                )
    return out if out else None


def run_single(
    var_code: str,
    question_text: str,
    country: str | None,
    model: str,
    generate_fn,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> dict:
    """Run one JSON feature-selection call. Returns result dict with raw response and parsed features."""

    if country:
        user_msg = PROMPT_COUNTRY.format(question=question_text, country=country)
        condition = "country_provided"
    else:
        user_msg = PROMPT_UNPROMPTED.format(question=question_text)
        condition = "unprompted"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    raw = generate_fn(
        messages, max_tokens=max_tokens, temperature=temperature, usage_phase="feature_list"
    )
    if raw is None:
        raw = ""

    features = None
    parse_error = None
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        features = _normalize_parsed_features(parsed if parsed is not None else None)
        if features is None and parsed is not None:
            parse_error = "JSON parsed but could not normalize to a list of features"
    except json.JSONDecodeError as e:
        parse_error = str(e)

    return {
        "target": var_code,
        "question_text": question_text,
        "country": country,
        "condition": condition,
        "model": model,
        "raw_response": raw,
        "features": features,
        "n_features": len(features) if features is not None else None,
        "parse_error": parse_error,
        "timestamp": datetime.now().isoformat(),
    }


def run_batch(
    targets: dict[str, str],
    countries: list[str],
    model: str,
    generate_fn,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> list[dict]:
    """
    Run JSON feature selection for all target × country combinations.

    Args:
        targets: {var_code: question_text}
        countries: list of country names
        model: model identifier string
        generate_fn: callable(messages, max_tokens, temperature) -> str
        max_tokens: max tokens for generation
        temperature: sampling temperature

    Returns:
        List of result dicts.
    """
    results = []

    for var_code, question_text in targets.items():
        # Unprompted
        r = run_single(var_code, question_text, None, model, generate_fn, max_tokens, temperature)
        results.append(r)
        status = "OK" if r["features"] else f"PARSE ERROR: {r['parse_error']}"
        print(f"  {var_code} | {'unprompted':15s} | {r['n_features'] or 0} features | {status}")

        # Country-conditioned
        for country in countries:
            r = run_single(var_code, question_text, country, model, generate_fn, max_tokens, temperature)
            results.append(r)
            status = "OK" if r["features"] else f"PARSE ERROR: {r['parse_error']}"
            print(f"  {var_code} | {country:15s} | {r['n_features'] or 0} features | {status}")

    return results


def save_results(results: list[dict], path: str = "phase0b_results.json"):
    """Save results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {path}")
