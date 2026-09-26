"""The cache directory ships the study's conversations, so it must not be ignored."""

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_ignore(path: str) -> subprocess.CompletedProcess[str]:
    """Return the result of ``git check-ignore`` for a synthetic path."""
    return subprocess.run(
        ["git", "check-ignore", path],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cache_is_not_ignored() -> None:
    # git check-ignore exits 1 with no output when a path is not ignored.
    result = _check_ignore("cache/agent/baseline/__synthetic__.json")
    assert result.returncode == 1, result.stdout
    assert result.stdout.strip() == ""


def test_results_progress_is_still_ignored() -> None:
    result = _check_ignore("results/.progress/__synthetic__.json")
    assert result.returncode == 0
    assert "results/.progress" in result.stdout
