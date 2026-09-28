"""Plain display names and number phrasing shared by the analysis findings.

The findings are read by someone new to the study, so this module turns
internal identifiers into words a newcomer understands and formats numbers with
their units. A judge is named by what it is, a variant by how the agent behaved,
and a calibration signal by the question its probability answers. Every helper
falls back to the raw identifier only for ids the study never uses.
"""

_JUDGE_NAMES: dict[str, str] = {
    "code": "the rule-based check",
    "llm_cheap": "the fast text model",
    "llm_strong": "the strong text model",
    "jev": "Jev",
    "laya_base": "Laya as published",
    "laya_ft": "Laya trained on these conversations",
}

_VARIANT_NAMES: dict[str, str] = {
    "baseline": "the careful agent",
    "degraded": "the rushed agent",
}

_SIGNAL_NAMES: dict[str, str] = {
    "completed_noul": 'its answer to "did the agent complete the request"',
    "verdict_pass_prob": "its pass-or-fail probability",
    "verdict_confidence": "its stated confidence in its own verdict",
}


def judge_name(judge_id: str) -> str:
    """Return the plain name for a judge id, or the id when it is unknown."""
    return _JUDGE_NAMES.get(judge_id, judge_id)


def variant_name(variant: str) -> str:
    """Return the plain name for an agent variant, or the id when it is unknown."""
    return _VARIANT_NAMES.get(variant, variant)


def signal_name(signal: str) -> str:
    """Return the plain description of a calibration signal, or the id when unknown."""
    return _SIGNAL_NAMES.get(signal, signal)


def as_percent(fraction: float) -> str:
    """Render a fraction between zero and one as a whole-number percentage."""
    return f"{round(fraction * 100)}%"


def as_money(usd: float) -> str:
    """Render a US dollar amount with two decimals and a leading dollar sign."""
    return f"${usd:.2f}"


def as_points(fraction: float) -> str:
    """Render a fraction as a whole number of percentage points, ignoring sign."""
    points = round(abs(fraction) * 100)
    unit = "point" if points == 1 else "points"
    return f"{points} {unit}"
