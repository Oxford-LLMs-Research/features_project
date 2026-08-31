"""Unit tests for selector system-prompt arms."""

from survey_features.elicitation import (
    freetext_messages,
    resolve_prompt_pack,
    system_prompt_for_arm,
)
from survey_features.prompts import (
    DEFAULT_PROMPT_ARM,
    DEFAULT_PROMPT_PACK,
    PROMPT_ARMS,
    PROMPT_PACKS,
    SYSTEM_PROMPT,
)


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


def test_analyst_arm_is_short_role():
    assert PROMPT_ARMS["analyst"] == "You are an analyst."


def test_default_pack_is_scientist_respondent():
    assert DEFAULT_PROMPT_PACK == "scientist_respondent"
    sys_arm, referent = resolve_prompt_pack()
    assert sys_arm == "social_scientist"
    assert referent == "respondent"


def test_analyst_person_pack_changes_system_and_referent():
    msgs = freetext_messages("Q?", country="Kenya", prompt_pack="analyst_person")
    assert msgs[0] == {"role": "system", "content": "You are an analyst."}
    assert "people in Kenya" in msgs[1]["content"]
    assert "person" in msgs[1]["content"]
    assert "respondent" not in msgs[1]["content"]


def test_none_respondent_pack_omits_system_keeps_respondent():
    msgs = freetext_messages("Q?", country="France", prompt_pack="none_respondent")
    assert msgs[0]["role"] == "user"
    assert "respondents in France" in msgs[0]["content"]
    assert "respondent" in msgs[0]["content"]


def test_prompt_packs_are_the_v2_stage1_set():
    assert set(PROMPT_PACKS) == {
        "scientist_respondent",
        "analyst_person",
        "none_respondent",
    }


def test_v2_stage1_runs_are_default_twice_plus_two_packs():
    from survey_features.prompts import (
        PROMPT_SENSITIVITY_V2_CONDITION,
        PROMPT_SENSITIVITY_V2_SELECTORS,
        prompt_sensitivity_v2_runs,
    )

    assert PROMPT_SENSITIVITY_V2_CONDITION == "country_provided"
    assert PROMPT_SENSITIVITY_V2_SELECTORS == (
        "deepseek_v4", "kimi", "minimax", "hermes",
    )
    assert prompt_sensitivity_v2_runs() == [
        ("scientist_respondent", 1),
        ("scientist_respondent", 2),
        ("analyst_person", None),
        ("none_respondent", None),
    ]
    assert prompt_sensitivity_v2_runs("scientist_respondent", 2) == [
        ("scientist_respondent", 2),
    ]
    assert prompt_sensitivity_v2_runs("analyst_person") == [("analyst_person", None)]
    try:
        prompt_sensitivity_v2_runs("analyst_person", 1)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "scientist_respondent" in str(exc)


def test_v2_temperature_draws_are_not_stage1_jobs():
    from survey_features.prompts import (
        PROMPT_SENSITIVITY_V2_STAGE1_TEMPERATURE,
        PROMPT_SENSITIVITY_V2_TEMPERATURE_RUNS_TEMPERATURE,
        prompt_sensitivity_v2_temperature_draws,
    )

    assert PROMPT_SENSITIVITY_V2_STAGE1_TEMPERATURE == 0.0
    assert PROMPT_SENSITIVITY_V2_TEMPERATURE_RUNS_TEMPERATURE == 1.0
    assert prompt_sensitivity_v2_temperature_draws() == [1, 2]
    assert prompt_sensitivity_v2_temperature_draws(2) == [2]
    try:
        prompt_sensitivity_v2_temperature_draws(3)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "1 or 2" in str(exc)
