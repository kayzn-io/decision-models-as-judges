"""Unit tests for the table-formatting helpers."""

import math

import pandas as pd

from decision_judges.ui import formatting


def _config() -> dict:
    """Build a column config over a frame with one column of each kind."""
    frame = pd.DataFrame(
        {
            "accuracy": [0.9],
            "error_rate": [0.1],
            "cost_per_item": [0.0123],
            "input_usd": [1.5],
            "latency_p50": [180],
            "kappa": [0.42],
            "n": [7],
            "repeats": [5],
            "errors": [0],
            "detected": [True],
            "model_id": ["none"],
        }
    )
    return formatting.column_config_for(frame)


def _type(config: dict, column: str) -> str:
    """Return the column-config type discriminator for a column."""
    return config[column]["type_config"]["type"]


def _format(config: dict, column: str) -> str:
    """Return the column-config number format for a column."""
    return config[column]["type_config"]["format"]


def test_ratio_columns_use_three_decimals() -> None:
    config = _config()
    assert _type(config, "accuracy") == "number"
    assert _format(config, "accuracy") == "%.3f"
    assert _format(config, "error_rate") == "%.3f"


def test_cost_and_usd_columns_use_dollar_four_decimals() -> None:
    config = _config()
    assert _format(config, "cost_per_item") == "$%.4f"
    assert _format(config, "input_usd") == "$%.4f"


def test_latency_column_uses_integer_milliseconds() -> None:
    config = _config()
    assert _format(config, "latency_p50") == "%d ms"


def test_kappa_column_uses_three_decimals() -> None:
    assert _format(_config(), "kappa") == "%.3f"


def test_count_columns_use_integers() -> None:
    config = _config()
    for column in ("n", "repeats", "errors"):
        assert _format(config, column) == "%d"


def test_boolean_column_becomes_disabled_checkbox() -> None:
    config = _config()
    assert _type(config, "detected") == "checkbox"
    assert config["detected"]["disabled"] is True


def test_default_column_is_plain_with_friendly_label() -> None:
    config = _config()
    assert "type_config" not in config["model_id"]
    assert config["model_id"]["label"] == "Model id"


def test_friendly_label_keeps_known_acronyms() -> None:
    assert formatting.friendly_label("auroc") == "AUROC"
    assert formatting.friendly_label("ece_after_isotonic") == "ECE after isotonic"
    assert formatting.friendly_label("f1_fail") == "F1 fail"
    assert formatting.friendly_label("input_usd") == "Input USD"
    assert formatting.friendly_label("cost_per_item") == "Cost per item"


def test_pct_formats_ratio_as_one_decimal_percent() -> None:
    assert formatting.pct(0.924) == "92.4%"
    assert formatting.pct(1.0) == "100.0%"


def test_pct_handles_missing_values() -> None:
    assert formatting.pct(float("nan")) == "—"
    assert formatting.pct(None) == "—"
    assert not math.isnan(0.0)  # guard: sanity of the fixture import
