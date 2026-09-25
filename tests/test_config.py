"""Tests for study and pricing configuration loading."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from decision_judges.config import (
    PricingTable,
    StudyConfig,
    UnpricedModel,
    load_pricing,
    load_study,
)
from decision_judges.types import Usage

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def test_load_study_succeeds() -> None:
    study = load_study(CONFIG_DIR / "study.toml")
    assert isinstance(study, StudyConfig)
    assert study.models.jev == "jev-1.13.0"
    assert study.models.laya.repo_id == "convaiinnovations/laya"
    assert study.models.laya.revision
    assert study.repeats >= 1
    assert study.seed == 7
    assert study.llm_base_url == "https://openrouter.ai/api/v1"
    assert study.thresholds.cascade == [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    assert "agent" in study.spend_caps


def test_load_study_reads_g2_repeats() -> None:
    study = load_study(CONFIG_DIR / "study.toml")
    assert study.g2.repeats == {
        "jev": 5,
        "laya_base": 5,
        "laya_ft": 5,
        "llm_cheap": 1,
        "llm_strong": 1,
        "code": 1,
    }


def test_study_without_g2_table_uses_defaults(tmp_path: Path) -> None:
    minimal = tmp_path / "study.toml"
    minimal.write_text(
        "\n".join(
            [
                "repeats = 5",
                "seed = 7",
                'llm_base_url = "https://openrouter.ai/api/v1"',
                "",
                "[models]",
                'agent = "openai/gpt-4.1"',
                'user_sim = "openai/gpt-4o-mini"',
                'llm_cheap = "openai/gpt-4o-mini"',
                'llm_strong = "openai/gpt-5"',
                "",
                "[models.laya]",
                'revision = "abc"',
                "",
                "[concurrency]",
                "llm = 1",
                "",
                "[spend_caps]",
                "agent = 1.0",
            ]
        )
    )
    study = load_study(minimal)
    assert study.g2.repeats["jev"] == 5
    assert study.g2.repeats["llm_cheap"] == 1


def test_study_g2_table_overrides_defaults(tmp_path: Path) -> None:
    override = tmp_path / "study.toml"
    override.write_text(
        "\n".join(
            [
                "seed = 7",
                'llm_base_url = "https://openrouter.ai/api/v1"',
                "",
                "[models]",
                'agent = "openai/gpt-4.1"',
                'user_sim = "openai/gpt-4o-mini"',
                'llm_cheap = "openai/gpt-4o-mini"',
                'llm_strong = "openai/gpt-5"',
                "",
                "[models.laya]",
                'revision = "abc"',
                "",
                "[concurrency]",
                "llm = 1",
                "",
                "[spend_caps]",
                "agent = 1.0",
                "",
                "[g2]",
                "repeats = { jev = 3 }",
            ]
        )
    )
    study = load_study(override)
    assert study.g2.repeats == {"jev": 3}


def test_load_pricing_succeeds() -> None:
    pricing = load_pricing(CONFIG_DIR / "pricing.toml")
    assert isinstance(pricing, PricingTable)
    assert pricing.effective_date == date(2026, 9, 24)
    assert "jev-1.13.0" in pricing.models


def test_cost_arithmetic() -> None:
    pricing = load_pricing(CONFIG_DIR / "pricing.toml")
    price = pricing.models["jev-1.13.0"]
    usage = Usage(input_tokens=1_000_000, output_tokens=2_000_000)
    expected = price.input_per_mtok + 2 * price.output_per_mtok
    assert pricing.cost("jev-1.13.0", usage) == pytest.approx(expected)


def test_cost_known_case() -> None:
    table = PricingTable(
        effective_date=date(2026, 9, 24),
        source="test",
        models={"m": {"input_per_mtok": 3.0, "output_per_mtok": 6.0}},
    )
    usage = Usage(input_tokens=500_000, output_tokens=250_000)
    assert table.cost("m", usage) == pytest.approx(3.0 * 0.5 + 6.0 * 0.25)


def test_unknown_model_raises() -> None:
    pricing = load_pricing(CONFIG_DIR / "pricing.toml")
    with pytest.raises(UnpricedModel) as exc:
        pricing.cost("no-such-model", Usage())
    assert "no-such-model" in str(exc.value)
    assert isinstance(exc.value, KeyError)


def test_study_missing_key_names_it(tmp_path: Path) -> None:
    incomplete = tmp_path / "study.toml"
    incomplete.write_text(
        "\n".join(
            [
                "repeats = 5",
                "seed = 7",
                'llm_base_url = "https://openrouter.ai/api/v1"',
                "",
                "[models]",
                'user_sim = "openai/gpt-4o-mini"',
                'llm_cheap = "openai/gpt-4o-mini"',
                'llm_strong = "openai/gpt-5"',
                "",
                "[models.laya]",
                'revision = "abc"',
                "",
                "[concurrency]",
                "jev = 1",
                "llm = 1",
                "laya = 1",
                "",
                "[spend_caps]",
                "agent = 1.0",
                "",
                "[thresholds]",
                "cascade = [0.5]",
            ]
        )
    )
    with pytest.raises(ValidationError) as exc:
        load_study(incomplete)
    assert "agent" in str(exc.value)
