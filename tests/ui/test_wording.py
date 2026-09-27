"""Wording guards for the learner-facing UI: plain vocabulary, samples, and tooltips.

These tests protect the rewrite that tells one story: conversations, a ground
truth, and judges tested against it. The reader never learns where the ground
truth came from or how it was produced.

The forbidden-word scan parses every module under ``decision_judges/ui`` with
``ast`` and inspects string literals only. Its heuristics are deliberate, not a
proof:

* **Docstrings are skipped.** The first string statement of a module, class, or
  function documents code for maintainers, not the learner, so it is exempt.
* **Single-token strings are skipped.** A string with no whitespace is treated
  as an identifier, key, URL path, file path, or short label (for example
  ``"trajectory"`` as a widget key or ``"Trajectories"`` as the kept page
  title), not prose. This is where ``getattr(record, "reward")`` is allowed:
  ``reward`` names an attribute in code, never learner prose.
* **Command-like strings are skipped.** A string containing ``--`` or starting
  with ``judges `` is a shell command shown to the reader as code, so a real
  ``--gate`` flag or ``--variant baseline`` is allowed.
* **Markup strings are skipped.** A string carrying HTML or SVG markup (an angle
  bracket, an ``attr="`` fragment, or a ``data-`` attribute) is code the browser
  renders, not prose, so ``data-gate`` in a tile does not trip the scan.

Every remaining prose string must avoid the internal jargon below. The plain
vocabulary is ``ground truth``, ``careful agent``, ``rushed agent``,
``conversation``, ``customer request``, ``judge``, ``verdict``, ``decision
model``, and ``experiment``.
"""

import ast
import re
from pathlib import Path

from decision_judges.ui import components, steps
from decision_judges.ui.data import Paths
from decision_judges.ui.flow import Station, render_svg

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI = _REPO_ROOT / "decision_judges" / "ui"


def _targets() -> list[Path]:
    """Return every module under the UI package, sorted."""
    return sorted(_UI.rglob("*.py"))


# Existing jargon bans kept from the first rewrite, plus the one-story vocabulary.
_FORBIDDEN = {
    "serialize": re.compile(r"serializ", re.IGNORECASE),
    "trajectory": re.compile(r"\btrajector", re.IGNORECASE),
    "state (noun)": re.compile(r"\bstates?\b", re.IGNORECASE),
    "gate": re.compile(r"\bgates?\b", re.IGNORECASE),
    "tau-bench": re.compile(r"tau[\s-]?bench", re.IGNORECASE),
    "benchmark": re.compile(r"\bbenchmark", re.IGNORECASE),
    "checker": re.compile(r"\bchecker", re.IGNORECASE),
    "grader": re.compile(r"\bgrader", re.IGNORECASE),
    "graded": re.compile(r"\bgraded\b", re.IGNORECASE),
    "harness": re.compile(r"\bharness", re.IGNORECASE),
    "reward": re.compile(r"\breward", re.IGNORECASE),
    "database": re.compile(r"\bdatabase", re.IGNORECASE),
    "sandbox": re.compile(r"\bsandbox", re.IGNORECASE),
    "fake store": re.compile(r"\bfake store", re.IGNORECASE),
    "scripted": re.compile(r"\bscripted", re.IGNORECASE),
    "simulated": re.compile(r"\bsimulated", re.IGNORECASE),
    "expected action": re.compile(r"\bexpected actions?", re.IGNORECASE),
    "required output": re.compile(r"\brequired outputs?", re.IGNORECASE),
    "answer key": re.compile(r"\banswer keys?", re.IGNORECASE),
    "program": re.compile(r"\bprograms?\b", re.IGNORECASE),
    "commit hash": re.compile(r"\bcommit hash", re.IGNORECASE),
}

_SAMPLE_CAPTION = "# example (your own appears here after the step runs)"


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Return the id() of every docstring Constant node, to exempt from the scan."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _is_markup(text: str) -> bool:
    """Return whether a string carries HTML or SVG markup rather than prose."""
    return "<" in text or ">" in text or "data-" in text or '="' in text


def _is_prose(text: str) -> bool:
    """Return whether a string is learner-facing prose rather than code or a label."""
    if not any(character.isspace() for character in text):
        return False
    if "--" in text or text.startswith("judges "):
        return False
    if _is_markup(text):
        return False
    return True


def _prose_strings(path: Path) -> list[str]:
    """Return every prose string literal in a file, minus docstrings and code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _docstring_ids(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            if _is_prose(node.value):
                found.append(node.value)
    return found


def test_owned_ui_prose_avoids_internal_jargon() -> None:
    """No learner-facing string uses the internal jargon the rewrite replaced."""
    hits: list[str] = []
    for path in _targets():
        for text in _prose_strings(path):
            for label, pattern in _FORBIDDEN.items():
                if pattern.search(text):
                    hits.append(f"{path.name}: {label}: {text!r}")
    assert not hits, "jargon found in UI prose:\n" + "\n".join(hits)


def _paths(root: Path) -> Paths:
    """Build a Paths rooted at an empty directory so samples fall back to examples."""
    return Paths(
        repo_root=root,
        cache_dir=root / "cache",
        results_dir=root / "results",
        config_dir=root / "config",
        data_dir=root / "data",
    )


def test_sample_examples_begin_with_the_caption(tmp_path: Path) -> None:
    """On an empty tree every example falls back to a sample that opens with the caption."""
    paths = _paths(tmp_path)
    for step in steps.STEPS:
        assert step.example_input(paths).startswith(_SAMPLE_CAPTION), step.id
        assert step.example_output(paths).startswith(_SAMPLE_CAPTION), step.id


def test_term_returns_an_abbr_with_the_meaning_as_title() -> None:
    """term() wraps the word in an abbr whose title carries the meaning."""
    rendered = components.term("conversation", "a full exchange")
    assert rendered.startswith("<abbr")
    assert 'title="a full exchange"' in rendered
    assert ">conversation</abbr>" in rendered


def test_term_escapes_markup_in_word_and_meaning() -> None:
    """term() escapes angle brackets and quotes so the tooltip cannot break out."""
    rendered = components.term("a & b", 'he said "hi" <x>')
    assert "&amp;" in rendered
    assert "&quot;" in rendered
    assert "&lt;x&gt;" in rendered


_FULL_CAPTIONS = [
    "what each customer asked for",
    "customer and agent, start to finish",
    "the conversation with the ground truth removed",
    "every answer a judge gave",
    "what the experiments concluded",
]


def _counts() -> dict[Station, int]:
    """Return a stub count for every station."""
    return dict.fromkeys(Station, 0)


def test_full_flow_strip_shows_a_caption_under_every_station() -> None:
    """In full mode each station carries its plain one-line caption."""
    svg = render_svg(_counts(), active=None, running=None, paid=False, compact=False)
    for caption in _FULL_CAPTIONS:
        assert f">{caption}</text>" in svg


def test_compact_flow_strip_omits_the_captions() -> None:
    """The compact strip stays terse: no station captions."""
    svg = render_svg(_counts(), active=None, running=None, paid=False, compact=True)
    for caption in _FULL_CAPTIONS:
        assert caption not in svg
