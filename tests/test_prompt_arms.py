"""Unit tests for selector system-prompt arms."""

from survey_features.elicitation import freetext_messages, system_prompt_for_arm
from survey_features.prompts import DEFAULT_PROMPT_ARM, PROMPT_ARMS, SYSTEM_PROMPT


def test_default_arm_is_social_scientist():
    assert DEFAULT_PROMPT_ARM == "social_scientist"
    assert PROMPT_ARMS["social_scientist"] == SYSTEM_PROMPT


def test_none_arm_omits_system_message():
    msgs = freetext_messages("How old are you?", prompt_arm="none")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "How old are you?" in msgs[0]["content"]


def test_helpful_arm_uses_neutral_system():
    msgs = freetext_messages("Q?", country="France", prompt_arm="helpful")
    assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert msgs[1]["role"] == "user"
    assert "France" in msgs[1]["content"]


def test_default_messages_keep_social_scientist():
    msgs = freetext_messages("Q?")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_system_prompt_for_arm_rejects_unknown():
    try:
        system_prompt_for_arm("not_an_arm")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not_an_arm" in str(e)
