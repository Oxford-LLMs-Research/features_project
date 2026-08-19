"""Pricing / cost helpers for token usage logs."""

from survey_features.llm import estimate_cost_usd, load_nebius_pricing


def test_load_nebius_pricing_has_v4_pro():
    table = load_nebius_pricing()
    assert "deepseek-ai/DeepSeek-V4-Pro" in table
    row = table["deepseek-ai/DeepSeek-V4-Pro"]
    assert row["input"] == 1.75
    assert row["output"] == 3.5


def test_estimate_cost_usd():
    # 1M prompt + 1M completion at Pro rates -> 1.75 + 3.5
    usd = estimate_cost_usd(1_000_000, 1_000_000, "deepseek-ai/DeepSeek-V4-Pro")
    assert usd is not None
    assert abs(usd - 5.25) < 1e-9


def test_estimate_cost_unknown_model():
    assert estimate_cost_usd(100, 100, "not-a-real/model") is None
