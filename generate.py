"""
LLM generation wrapper.

Returns a generate_fn(messages, max_tokens, temperature) -> str
compatible with all pipeline modules (phase0b_pipeline, phase0b_disambig).

Backed by an OpenAI-compatible API, so it works with:
  - SGLang local server  (LLM_BASE_URL=http://localhost:30000/v1, LLM_API_KEY=EMPTY)
  - OpenRouter           (LLM_BASE_URL=https://openrouter.ai/api/v1)
  - OpenAI               (LLM_BASE_URL=https://api.openai.com/v1)
  - Together.ai, etc.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import NotFoundError, OpenAI

load_dotenv()


def make_generate_fn(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[callable, str]:
    """
    Build a generate_fn and return it together with the model name.

    Falls back to environment variables when arguments are None.

    Returns:
        (generate_fn, model_name)
    """
    base_url = base_url or os.environ["LLM_BASE_URL"]
    api_key = api_key or os.environ["LLM_API_KEY"]
    model = model or os.environ["LLM_MODEL"]
    fallback_model = os.environ.get("LLM_FALLBACK_MODEL")
    if not fallback_model and model == "deepseek-ai/DeepSeek-V3":
        fallback_model = "deepseek-ai/DeepSeek-V3.2"

    client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_fn(messages: list[dict], max_tokens: int = 2048, temperature: float = 0.0) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except NotFoundError:
            if not fallback_model:
                raise
            print(f"[generate] Model '{model}' not found, retrying with '{fallback_model}'")
            response = client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return response.choices[0].message.content

    return generate_fn, model
