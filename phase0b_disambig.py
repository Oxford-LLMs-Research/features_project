"""
Phase 0b: Step 2 — LLM-based disambiguation
Given embedding shortlist candidates, an LLM picks the best match or 'none'.
"""

from __future__ import annotations

import json
import re

DISAMBIG_PROMPT = """You are helping map abstract feature descriptions to concrete survey questions.

A researcher said they would want to know a respondent's:
"{feature_label}"
Reasoning: "{feature_reasoning}"

Below are candidate survey questions that might capture this information. Pick the ONE question that best matches what the researcher is asking for, or respond "none" if no question is a good match.

Candidates:
{candidates_block}

Respond with ONLY the letter (A, B, C, ...) of the best match, or "none". No explanation."""


def format_candidates(candidates: list[dict]) -> str:
    """Format candidates as a lettered list."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"{letters[i]}. [{c['var_code']}] {c['question_text']}")
    return "\n".join(lines)


def parse_disambig_response(raw: str, n_candidates: int) -> int | None:
    """
    Parse the LLM response into a candidate index or None.
    Returns index (0-based) or None for 'none'.
    """
    cleaned = raw.strip().upper()

    if "NONE" in cleaned:
        return None

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(n_candidates):
        if letters[i] in cleaned:
            return i

    return None


def disambiguate_single(
    feature_label: str,
    feature_reasoning: str,
    candidates: list[dict],
    generate_fn,
    model: str,
    max_tokens: int = 16,
    temperature: float = 0.0,
) -> dict:
    """
    Ask LLM to pick the best candidate or 'none'.

    Returns:
        {
            "selected_code": var_code or None,
            "selected_text": question_text or None,
            "selected_rank": index in candidates or None,
            "raw_response": str,
        }
    """
    prompt = DISAMBIG_PROMPT.format(
        feature_label=feature_label,
        feature_reasoning=feature_reasoning,
        candidates_block=format_candidates(candidates),
    )

    messages = [{"role": "user", "content": prompt}]
    raw = generate_fn(messages, max_tokens=max_tokens, temperature=temperature)

    idx = parse_disambig_response(raw, len(candidates))

    if idx is not None and idx < len(candidates):
        return {
            "selected_code": candidates[idx]["var_code"],
            "selected_text": candidates[idx]["question_text"],
            "selected_rank": idx,
            "raw_response": raw.strip(),
        }
    else:
        return {
            "selected_code": None,
            "selected_text": None,
            "selected_rank": None,
            "raw_response": raw.strip(),
        }


def disambiguate_mappings(
    mappings: list[dict],
    generate_fn,
    model: str = "deepseek-ai/DeepSeek-V3.2",
) -> list[dict]:
    """
    Run disambiguation on all mappings. Adds 'disambig' key to each mapping dict.

    Args:
        mappings: output of map_features_to_variables
        generate_fn: callable(messages, max_tokens, temperature) -> str
        model: model identifier

    Returns:
        mappings with 'disambig' field added to each entry.
    """
    for i, m in enumerate(mappings):
        if not m["candidates"]:
            m["disambig"] = {
                "selected_code": None,
                "selected_text": None,
                "selected_rank": None,
                "raw_response": "no candidates",
            }
            print(f"  [{i+1}/{len(mappings)}] {m['target']} | {m['feature_label'][:30]:30s} -> SKIP (no candidates)")
            continue

        result = disambiguate_single(
            feature_label=m["feature_label"],
            feature_reasoning=m["feature_reasoning"],
            candidates=m["candidates"],
            generate_fn=generate_fn,
            model=model,
        )
        m["disambig"] = result

        code = result["selected_code"] or "none"
        print(f"  [{i+1}/{len(mappings)}] {m['target']} | {m['feature_label'][:30]:30s} -> {code:8s} (raw: {result['raw_response']})")

    return mappings


def print_disambig_summary(mappings: list[dict], ground_truth: dict[str, list[str]] = None):
    """Print summary of disambiguation results with optional ground truth comparison."""
    current_key = None
    for m in mappings:
        if "disambig" not in m:
            continue

        key = (m["target"], m["country"], m["condition"])
        if key != current_key:
            current_key = key
            country_str = m["country"] or "unprompted"
            gt_vars = ground_truth.get(m["target"], []) if ground_truth else []
            print(f"\n{'='*70}")
            print(f"{m['target']} | {country_str} | GT: {gt_vars}")
            print(f"{'='*70}")

        code = m["disambig"]["selected_code"]
        rank = m["disambig"]["selected_rank"]

        hit = ""
        if ground_truth and code and code in ground_truth.get(m["target"], []):
            hit = " *** HIT ***"

        if code:
            print(f"  [{m['feature_rank']}] {m['feature_label'][:40]:40s} -> {code} (rank {rank}){hit}")
        else:
            print(f"  [{m['feature_rank']}] {m['feature_label'][:40]:40s} -> none")


# ── Usage ──
#
# from phase0b_mapping import map_features_to_variables, ...
#
# # After running embedding retrieval:
# mappings = disambiguate_mappings(mappings, generate_chat)
#
# ground_truth = {
#     "Q199": ["Q200", "Q4", "Q98"],
#     "Q235": ["Q236"],
#     "Q57": ["Q61", "Q59", "Q66"],
#     "Q47": ["Q262", "Q46"],
#     "Q164": ["Q165", "Q172", "Q6"],
# }
# print_disambig_summary(mappings, ground_truth)
#
# # Save
# with open("phase0b_disambiguated.json", "w") as f:
#     json.dump(mappings, f, indent=2, ensure_ascii=False)
