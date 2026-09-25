"""Table formatting helpers: column configs and label and percent rendering.

These helpers shape already-computed values for display. They set number
formats, friendly headers, and percent strings; they compute no metrics.
"""

from typing import Any

import pandas as pd
import streamlit as st

_ACRONYMS = frozenset({"AUROC", "ECE", "F1", "USD"})
_JUDGE_COLUMNS = frozenset({"judge", "judge_id"})
_RATIO_KEYWORDS = (
    "rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "ece",
    "brier",
    "agreement",
    "probability",
    "confidence",
    "necessary",
    "consistent",
)
_COUNT_KEYWORDS = ("count", "repeats", "errors")
_MISSING = "—"


def friendly_label(column: str) -> str:
    """Turn a snake_case column into a Sentence-case header, keeping known acronyms."""
    words: list[str] = []
    for index, token in enumerate(column.split("_")):
        upper = token.upper()
        if upper in _ACRONYMS:
            words.append(upper)
        elif index == 0:
            words.append(token.capitalize())
        else:
            words.append(token.lower())
    return " ".join(words)


def axis_title(column: str) -> str:
    """Return a chart axis or legend title for a column name.

    Judge columns collapse to ``Judge``, cost columns gain a ``(USD)`` unit, and
    every other column keeps its friendly Sentence-case header.
    """
    lower = column.lower()
    if lower in _JUDGE_COLUMNS:
        return "Judge"
    label = friendly_label(column)
    if "cost" in lower and "usd" not in lower:
        return f"{label} (USD)"
    return label


def column_config_for(df: pd.DataFrame) -> dict[str, Any]:
    """Map each column to a Streamlit column config by its name and dtype.

    Ratios render to three decimals, USD and cost to four decimals with a
    dollar sign, latency as integer milliseconds, kappa to three decimals, and
    counts as integers; boolean columns become disabled checkboxes and every
    other column keeps a plain friendly-labeled header.
    """
    config: dict[str, Any] = {}
    for column in df.columns:
        name = str(column)
        lower = name.lower()
        label = friendly_label(name)
        if pd.api.types.is_bool_dtype(df[column]):
            config[column] = st.column_config.CheckboxColumn(label=label, disabled=True)
        elif "usd" in lower or "cost" in lower:
            config[column] = st.column_config.NumberColumn(
                label=label, format="$%.4f", width="small"
            )
        elif "latency" in lower:
            config[column] = st.column_config.NumberColumn(
                label=label, format="%d ms", width="small"
            )
        elif "kappa" in lower:
            config[column] = st.column_config.NumberColumn(
                label=label, format="%.3f", width="small"
            )
        elif any(keyword in lower for keyword in _RATIO_KEYWORDS):
            config[column] = st.column_config.NumberColumn(
                label=f"{label} (0–1)", format="%.3f", width="small"
            )
        elif lower == "n" or any(keyword in lower for keyword in _COUNT_KEYWORDS):
            config[column] = st.column_config.NumberColumn(label=label, format="%d", width="small")
        else:
            config[column] = st.column_config.Column(label=label)
    return config


def pct(x: float | None) -> str:
    """Render a ratio in ``[0, 1]`` as a one-decimal percent, blank when missing."""
    if x is None or x != x:
        return _MISSING
    return f"{x * 100:.1f}%"
