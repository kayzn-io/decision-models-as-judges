"""Live page tests driven through AppTest over a thin script."""

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decision_judges.pipeline import FakeJudge
from decision_judges.ui import keys, live

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_SCRIPT = "from decision_judges.ui.pages.live import render\n\nrender()\n"
_FAKE_KEY = "sk-fake-live-key-XYZ"


def _script(tmp_path: Path) -> str:
    """Write the thin page script and return its path."""
    script = tmp_path / "run_live.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return str(script)


def _local_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, AppTest]:
    """Copy the fixture tree to a writable root and build a local-mode AppTest."""
    root = tmp_path / "root"
    shutil.copytree(_UI_ROOT, root)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    return root, AppTest.from_file(_script(tmp_path), default_timeout=60)


def _judge_files(root: Path) -> list[str]:
    """Return the sorted relative paths of every file under the judge cache."""
    judge_dir = root / "cache" / "judge"
    return sorted(
        path.relative_to(root).as_posix() for path in judge_dir.rglob("*") if path.is_file()
    )


def _fake_build(*ids: str):
    """Return a build_live_judges replacement yielding fake judges with the given ids."""

    def build(*_args: object, **_kwargs: object) -> list[FakeJudge]:
        return [FakeJudge(judge_id=judge_id) for judge_id in ids]

    return build


def _disable_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the checkpoint check to report Laya unavailable, avoiding any cache scan."""
    monkeypatch.setattr("decision_judges.ui.live.laya_available", lambda _study: False)


def _texts(at: AppTest) -> list[str]:
    """Return every rendered markdown and caption value for substring assertions."""
    return [block.value for block in [*at.markdown, *at.caption]]


def test_live_shared_mode_shows_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(_UI_ROOT))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    at = AppTest.from_file(_script(tmp_path), default_timeout=30).run()

    assert not at.exception
    assert any("JUDGES_LOCAL=1" in info.value for info in at.info)


def test_live_renders_with_button_disabled_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_laya(monkeypatch)
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert at.button(key="live_judge").disabled is True


def test_live_entering_key_enables_button(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _disable_laya(monkeypatch)
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    assert not at.exception
    assert at.button(key="live_judge").disabled is False


def test_live_click_shows_verdict_metrics_and_writes_no_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_laya(monkeypatch)
    monkeypatch.setattr(
        "decision_judges.ui.live.build_live_judges", _fake_build("llm_strong", "jev")
    )
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    before = _judge_files(root)

    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()
    at.button(key="live_judge").click().run()

    assert not at.exception
    assert [metric.label for metric in at.metric].count("Verdict") == 2
    assert _judge_files(root) == before


def test_live_button_disabled_after_reaching_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_laya(monkeypatch)
    monkeypatch.setattr(
        "decision_judges.ui.live.build_live_judges", _fake_build("llm_strong", "jev")
    )
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()
    for _ in range(live.LIVE_CALL_CAP):
        at.button(key="live_judge").click().run()
    at.run()

    assert not at.exception
    assert at.button(key="live_judge").disabled is True
    assert any(f"limit of {live.LIVE_CALL_CAP}" in text for text in _texts(at))


def test_live_never_renders_the_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _disable_laya(monkeypatch)
    monkeypatch.setattr(
        "decision_judges.ui.live.build_live_judges", _fake_build("llm_strong", "jev")
    )
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()
    at.button(key="live_judge").click().run()

    assert not at.exception
    assert all(_FAKE_KEY not in str(text) for text in _texts(at))


def _rendered(at: AppTest) -> str:
    """Return concatenated markdown and html text for substring assertions."""
    return "\n".join(node.value for node in [*at.get("markdown"), *at.get("html")])


def test_live_shows_flow_strip_framing_and_next_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_laya(monkeypatch)
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert 'class="flow-strip"' in _rendered(at)
    assert any("shows that difference side by side" in text for text in _texts(at))
    assert '<a href="/overview" class="in-app-link">Next: Overview</a>' in _rendered(at)


def test_live_shows_latency_race_after_judging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_laya(monkeypatch)
    monkeypatch.setattr(
        "decision_judges.ui.live.build_live_judges", _fake_build("llm_strong", "jev")
    )
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()
    at.button(key="live_judge").click().run()

    assert not at.exception
    assert any("Response time by judge" in text for text in _texts(at))
