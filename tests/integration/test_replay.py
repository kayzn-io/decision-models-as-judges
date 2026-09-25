"""Replay the results command through the installed console script offline."""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKERS = "<!-- results:start -->\nplaceholder\n<!-- results:end -->\n"
EMPTY_SUMMARY = "No cached results yet. Tables and charts appear here as evaluation stages run."


def test_results_replays_offline_on_empty_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    results_dir = tmp_path / "results"
    readme = tmp_path / "README.md"
    readme.write_text(MARKERS, encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k != "SETUPTOOLS_USE_DISTUTILS"}

    result = subprocess.run(
        [
            "uv",
            "run",
            "judges",
            "results",
            "--cache-dir",
            str(cache_dir),
            "--results-dir",
            str(results_dir),
            "--readme",
            str(readme),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (results_dir / "summary.md").is_file()
    readme_text = readme.read_text(encoding="utf-8")
    assert EMPTY_SUMMARY in readme_text
    assert "### Outcome judging" in readme_text
