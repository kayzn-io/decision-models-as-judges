"""Wording guards for the learner-facing UI: plain vocabulary, samples, and tooltips.

These tests protect the rewrite that replaced internal jargon with plain words.

The forbidden-word scan parses ``steps.py`` and the five owned page modules with
``ast`` and inspects string literals only. Its heuristics are deliberate, not a
proof:

* **Docstrings are skipped.** The first string statement of a module, class, or
  function documents code for maintainers, not the learner, so it is exempt.
* **Single-token strings are skipped.** A string with no whitespace is treated
  as an identifier, key, URL path, or short label (for example ``"trajectory"``
  as a widget key or ``"Trajectories"`` as the kept page title), not prose.
* **Command-like strings are skipped.** A string containing ``--`` or starting
  with ``judges `` is a shell command shown to the reader as code, so a real
  ``--gate`` flag is allowed.

Every remaining prose string must avoid ``serialize``, ``trajectory``, the word
``state``/``states`` as a standalone word, and the word ``gate``/``gates``. The
``\\bstate\\b`` boundary never matches inside ``session_state``.
"""

import ast
import re
from pathlib import Path

from decision_judges.ui import components, steps
from decision_judges.ui.data import Paths
from decision_judges.ui.flow import Station, render_svg

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI = _REPO_ROOT / "decision_judges" / "ui"
_TARGETS = [
    _UI / "steps.py",
    _UI / "pages" / "run.py",
    _UI / "pages" / "trajectories.py",
    _UI / "pages" / "gates.py",
    _UI / "pages" / "label.py",
    _UI / "pages" / "live.py",
]

_FORBIDDEN = {
    "serialize": re.compile(r"serializ", re.IGNORECASE),
    "trajectory": re.compile(r"\btrajector", re.IGNORECASE),
    "state (noun)": re.compile(r"\bstates?\b", re.IGNORECASE),
    "gate": re.compile(r"\bgates?\b", re.IGNORECASE),
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


def _is_prose(text: str) -> bool:
    """Return whether a string is learner-facing prose rather than code or a label."""
    if not any(character.isspace() for character in text):
        return False
    if "--" in text or text.startswith("judges "):
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
    """No learner-facing string uses serialize, trajectory, state, or gate."""
    hits: list[str] = []
    for path in _TARGETS:
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
    "115 scripted customer requests",
    "what the agent and the simulated customer said",
    "what each judge is allowed to read",
    "every judge answer",
    "the written results",
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
