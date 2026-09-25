"""Pure unit tests for the live judging helpers."""

from pathlib import Path

import pytest

from decision_judges.config import PricingTable, StudyConfig, load_pricing, load_study
from decision_judges.ui import live

_CONFIG = Path(__file__).resolve().parent / "fixtures" / "ui_cache" / "config"


class _FakeTypeSafeClient:
    """Stand-in for the TypeSafe SDK client that records its keyword arguments."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.fixture
def study() -> StudyConfig:
    """Return the fixture study configuration."""
    return load_study(_CONFIG / "study.toml")


@pytest.fixture
def pricing() -> PricingTable:
    """Return the fixture pricing table."""
    return load_pricing(_CONFIG / "pricing.toml")


def _patch_typesafe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the TypeSafe client class with a fake so nothing reaches the network."""
    import typesafe_sdk

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", _FakeTypeSafeClient)


def test_build_live_judges_returns_expected_ids(
    monkeypatch: pytest.MonkeyPatch, study: StudyConfig, pricing: PricingTable
) -> None:
    _patch_typesafe(monkeypatch)
    keys = live.LiveKeys(openrouter="sk-fake", typesafe=None)

    judges = live.build_live_judges(study, pricing, {}, {}, keys, include_laya=False)

    assert [judge.judge_id for judge in judges] == ["llm_strong", "jev"]


def test_build_live_judges_skips_laya_without_consulting_cache(
    monkeypatch: pytest.MonkeyPatch, study: StudyConfig, pricing: PricingTable
) -> None:
    _patch_typesafe(monkeypatch)
    monkeypatch.setattr(live, "laya_available", lambda _study: pytest.fail("laya was checked"))
    keys = live.LiveKeys(openrouter="sk-fake")

    judges = live.build_live_judges(study, pricing, {}, {}, keys, include_laya=False)

    assert all(judge.judge_id != "laya" for judge in judges)


def test_live_keys_redact_never_exposes_raw_values() -> None:
    keys = live.LiveKeys(openrouter="sk-secret-123", typesafe="ts-secret-456")

    summary = keys.redact()

    assert "sk-secret-123" not in summary
    assert "ts-secret-456" not in summary


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, True), (1, True), (19, True), (20, False), (21, False)],
)
def test_can_call_respects_default_cap(count: int, expected: bool) -> None:
    assert live.can_call(count) is expected


def test_can_call_respects_custom_cap() -> None:
    assert live.can_call(2, cap=3) is True
    assert live.can_call(3, cap=3) is False
