"""Tests for the cached data layer."""

import shutil
from pathlib import Path

import pytest

from decision_judges.ui import data


def _paths_for(root: Path) -> data.Paths:
    """Build a Paths object rooted at a directory."""
    return data.Paths(
        repo_root=root,
        cache_dir=root / "cache",
        results_dir=root / "results",
        config_dir=root / "config",
        data_dir=root / "data",
    )


def test_dir_fingerprint_changes_when_a_file_changes(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("one", encoding="utf-8")
    before = data.dir_fingerprint(tmp_path)
    target.write_text("one plus more", encoding="utf-8")
    assert data.dir_fingerprint(tmp_path) != before


def test_paths_from_env_honors_judges_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(tmp_path))
    paths = data.Paths.from_env()
    assert paths.repo_root == tmp_path.resolve()
    assert paths.cache_dir == tmp_path.resolve() / "cache"
    assert paths.config_dir == tmp_path.resolve() / "config"


def test_load_verdicts_skips_invalid_files(ui_root: Path, tmp_path: Path) -> None:
    shutil.copytree(ui_root / "cache", tmp_path / "cache")
    shard = tmp_path / "cache" / "judge" / "zz"
    shard.mkdir(parents=True)
    (shard / "broken.json").write_text("{ not valid json", encoding="utf-8")

    verdicts = data.load_verdicts(_paths_for(tmp_path))
    assert len(verdicts) == 4


def test_judge_roster_columns_and_rows(ui_root: Path) -> None:
    study, pricing = data.load_study_and_pricing(_paths_for(ui_root))
    roster = data.judge_roster(study, pricing)
    assert list(roster.columns) == ["judge", "model_id", "input_per_mtok", "output_per_mtok"]
    assert list(roster["judge"]) == ["code", "llm_cheap", "llm_strong", "jev", "laya"]
    code_row = roster[roster["judge"] == "code"].iloc[0]
    assert code_row["model_id"] == "none"


def test_verdicts_by_state_groups_correctly(ui_root: Path) -> None:
    verdicts = data.load_verdicts(_paths_for(ui_root))
    grouped = data.verdicts_by_state(verdicts)
    assert len(grouped) == 2
    assert all(len(group) == 2 for group in grouped.values())
