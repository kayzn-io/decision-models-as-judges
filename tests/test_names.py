"""Tests for the plain display names and number phrasing helpers."""

from decision_judges.gates.names import (
    as_money,
    as_percent,
    as_points,
    judge_name,
    signal_name,
    variant_name,
)


def test_judge_name_maps_every_study_judge_to_plain_words() -> None:
    assert judge_name("code") == "the rule-based check"
    assert judge_name("llm_cheap") == "the fast text model"
    assert judge_name("llm_strong") == "the strong text model"
    assert judge_name("jev") == "Jev"
    assert judge_name("laya_base") == "Laya as published"
    assert judge_name("laya_ft") == "Laya trained on these conversations"


def test_judge_name_falls_back_to_the_id_when_unknown() -> None:
    assert judge_name("mystery") == "mystery"


def test_variant_name_maps_the_two_agents() -> None:
    assert variant_name("baseline") == "the careful agent"
    assert variant_name("degraded") == "the rushed agent"
    assert variant_name("other") == "other"


def test_signal_name_describes_each_calibration_signal_in_words() -> None:
    assert signal_name("completed_noul") == 'its answer to "did the agent complete the request"'
    assert signal_name("verdict_pass_prob") == "its pass-or-fail probability"
    assert signal_name("verdict_confidence") == "its stated confidence in its own verdict"
    assert signal_name("unknown") == "unknown"


def test_as_percent_rounds_to_a_whole_number_with_a_sign() -> None:
    assert as_percent(0.5478) == "55%"
    assert as_percent(0.0) == "0%"
    assert as_percent(1.0) == "100%"


def test_as_money_shows_two_decimals() -> None:
    assert as_money(0.015538) == "$0.02"
    assert as_money(0.03263) == "$0.03"
    assert as_money(0.0005) == "$0.00"


def test_as_points_is_absolute_and_pluralizes() -> None:
    assert as_points(-0.339) == "34 points"
    assert as_points(0.01) == "1 point"
    assert as_points(0.0435) == "4 points"
