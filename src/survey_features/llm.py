"""
LLM generation wrapper — the ONE client used everywhere in the pipeline.

Returns a generate_fn(messages, max_tokens, temperature, *, usage_phase=...) -> str
compatible with all pipeline modules (elicitation, extraction, disambig).

Backed by an OpenAI-compatible API, so it works with:
  - SGLang local server  (LLM_BASE_URL=http://localhost:30000/v1, LLM_API_KEY=EMPTY)
  - OpenRouter           (LLM_BASE_URL=https://openrouter.ai/api/v1)
  - OpenAI               (LLM_BASE_URL=https://api.openai.com/v1)
  - Nebius, Together.ai, etc.

Transient errors (429s, 5xx, provider hiccups like Nebius "Already borrowed") are
retried with linear backoff so one flaky call cannot abort a long sweep. After the
retries are exhausted the behaviour is controlled by ``on_error``:
  - "raise" (default): re-raise the last exception (legacy run_grid behaviour — the
    grid worker catches and records the cell error).
  - "empty": log and return "" (free-text pipeline behaviour — an empty response is
    treated as no-selection / no-mapping and the checkpointed run stays alive).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import NotFoundError, OpenAI

from . import config  # noqa: F401  (imports load .env once)


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    """Best-effort flatten of OpenAI-style CompletionUsage for JSONL."""
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, name, None)
        if v is not None:
            out[name] = int(v)
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        if hasattr(details, "model_dump"):
            out["completion_tokens_details"] = details.model_dump()
        elif isinstance(details, dict):
            out["completion_tokens_details"] = details
        else:
            rt = getattr(details, "reasoning_tokens", None)
            if rt is not None:
                out["reasoning_tokens"] = int(rt)
    return out


@dataclass
class TokenUsageLog:
    """Append-only JSONL token log; safe for concurrent grid workers."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # phase -> aggregated counters
    _agg: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        phase: str,
        model: str,
        usage: Any,
        finish_reason: str | None,
        max_tokens_requested: int,
    ) -> None:
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "model": model,
            "finish_reason": finish_reason,
            "max_tokens_requested": max_tokens_requested,
            **_usage_to_dict(usage),
        }
        ud = _usage_to_dict(usage)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            g = self._agg.setdefault(
                phase,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            g["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = ud.get(k)
                if v is not None:
                    g[k] = g.get(k, 0) + int(v)

    def print_summary(self) -> None:
        """Print per-phase and pooled totals (stdout)."""
        if not self._agg:
            print("\n[llm usage] No token records (usage object missing from API responses?)")
            return
        print(f"\n[llm usage] JSONL: {self.path.resolve()}")
        tot_p = tot_c = tot_t = tot_n = 0
        for phase in sorted(self._agg):
            g = self._agg[phase]
            n = g["calls"]
            p, c, t = g.get("prompt_tokens", 0), g.get("completion_tokens", 0), g.get("total_tokens", 0)
            tot_n += n
            tot_p += p
            tot_c += c
            tot_t += t
            print(
                f"  {phase:16s}  calls={n:5d}  "
                f"prompt~{p:,}  completion~{c:,}  total~{t:,}"
            )
        print(
            f"  {'ALL':16s}  calls={tot_n:5d}  "
            f"prompt~{tot_p:,}  completion~{tot_c:,}  total~{tot_t:,}"
        )


def make_generate_fn(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    usage_log: TokenUsageLog | None = None,
    usage_log_ref: list[TokenUsageLog | None] | None = None,
    max_retries: int = 5,
    on_error: str = "raise",
) -> tuple[Any, str]:
    """
    Build a generate_fn and return it together with the model name.

    Falls back to environment variables when arguments are None.

    Optional ``usage_log`` records per-request usage when callers pass
    ``usage_phase=`` (feature_list | extract | disambig | ...).

    If ``usage_log_ref`` is a one-element list, the element is read on each
    call (set it after you know the output path, e.g. once ``output_tag`` is
    resolved in ``run_grid``).

    ``max_retries`` transient-error attempts with linear backoff (2s, 4s, ...).
    ``on_error``: "raise" (re-raise after retries) or "empty" (return "").

    Returns:
        (generate_fn, model_name)
    """
    if on_error not in ("raise", "empty"):
        raise ValueError(f"on_error must be 'raise' or 'empty', got {on_error!r}")
    base_url = base_url or os.environ["LLM_BASE_URL"]
    api_key = api_key or os.environ["LLM_API_KEY"]
    model = model or os.environ["LLM_MODEL"]
    fallback_model = os.environ.get("LLM_FALLBACK_MODEL")
    if not fallback_model and model == "deepseek-ai/DeepSeek-V3":
        fallback_model = "deepseek-ai/DeepSeek-V3.2"

    client = OpenAI(base_url=base_url, api_key=api_key)

    def _create(use_model: str, messages: list[dict], max_tokens: int, temperature: float):
        return client.chat.completions.create(
            model=use_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_fn(
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        *,
        usage_phase: str | None = None,
    ) -> str:
        effective_model = model
        response = None
        last_exc: Exception | None = None
        for attempt in range(max(1, max_retries)):
            try:
                response = _create(model, messages, max_tokens, temperature)
                break
            except NotFoundError:
                if not fallback_model:
                    raise
                print(f"[generate] Model '{model}' not found, retrying with '{fallback_model}'")
                effective_model = fallback_model
                response = _create(fallback_model, messages, max_tokens, temperature)
                break
            except Exception as e:  # transient: 429s, 5xx, connection resets, ...
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        if response is None:
            if on_error == "raise" and last_exc is not None:
                raise last_exc
            print(
                f"[generate] giving up after {max_retries} retries: "
                f"{type(last_exc).__name__}: {str(last_exc)[:100]}"
            )
            return ""
        choice = response.choices[0]
        content = choice.message.content
        fr = getattr(choice, "finish_reason", None)

        _log = usage_log_ref[0] if usage_log_ref is not None else usage_log
        if _log is not None and usage_phase:
            _log.record(
                phase=usage_phase,
                model=effective_model,
                usage=getattr(response, "usage", None),
                finish_reason=fr,
                max_tokens_requested=max_tokens,
            )

        if content is None:
            print(f"[generate] Empty message content (finish_reason={fr!r}); treating as \"\"")
            return ""
        return content

    return generate_fn, model
