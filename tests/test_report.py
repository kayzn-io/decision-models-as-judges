"""Tests for report rendering and the results command."""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from typer.testing import CliRunner

from decision_judges.cli import app
from decision_judges.report import (
    render_results,
    render_summary,
    update_readme,
    write_chart,
    write_findings,
    write_table,
)

MARKERS = "<!-- results:start -->\n{body}\n<!-- results:end -->"
EMPTY_SUMMARY = "No cached results yet. Tables and charts appear here as evaluation stages run."
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "results_sample"

GATE_TITLES = [
    "Task triage before any run",
    "Per-step tool-call scoring",
    "Outcome judging",
    "Atomic questions versus one broad question",
    "Confidence-gated cascade",
    "Calibration",
    "Robustness to evaluator-directed text",
    "Regression detection",
    "Failure taxonomy against hand labels",
    "Local decision model: zero-shot versus fine-tuned",
]


def test_write_table_writes_both_files_with_header(tmp_path: Path) -> None:
    df = pd.DataFrame({"model": ["jev", "laya"], "score": [0.8, 0.9]})
    md_path, csv_path = write_table(tmp_path / "out", "scores", df)

    assert md_path == tmp_path / "out" / "scores.md"
    assert csv_path == tmp_path / "out" / "scores.csv"
    assert md_path.is_file()
    assert csv_path.is_file()

    md = md_path.read_text(encoding="utf-8")
    assert "| model" in md
    assert "score" in md
    assert "jev" in md

    csv = csv_path.read_text(encoding="utf-8")
    assert csv.splitlines()[0] == "model,score"


def test_write_chart_writes_both_files(tmp_path: Path) -> None:
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])

    png_path, svg_path = write_chart(tmp_path / "out", "curve", fig)

    assert png_path == tmp_path / "out" / "curve.png"
    assert svg_path == tmp_path / "out" / "curve.svg"
    assert png_path.is_file()
    assert svg_path.is_file()
    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


def test_update_readme_replaces_only_between_markers(tmp_path: Path) -> None:
    prefix = "# Title\n\nIntro paragraph.\n\n<!-- results:start -->"
    middle = "\nold body line\n"
    suffix = "<!-- results:end -->\n\nFooter.\n"
    readme = tmp_path / "README.md"
    readme.write_text(prefix + middle + suffix, encoding="utf-8")

    update_readme(readme, "new body")

    result = readme.read_text(encoding="utf-8")
    assert result == prefix + "\nnew body\n" + suffix
    assert result.startswith(prefix)
    assert result.endswith(suffix)


def test_update_readme_raises_on_missing_start_marker(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n<!-- results:end -->\n", encoding="utf-8")
    try:
        update_readme(readme, "body")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing start marker")


def test_update_readme_raises_on_missing_end_marker(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\nbody\n", encoding="utf-8")
    try:
        update_readme(readme, "body")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing end marker")


def test_update_readme_raises_when_markers_out_of_order(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:end -->\nx\n<!-- results:start -->\n", encoding="utf-8")
    try:
        update_readme(readme, "body")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when markers are out of order")


def test_render_summary_empty_without_cache_dir(tmp_path: Path) -> None:
    assert render_summary(tmp_path / "missing") == EMPTY_SUMMARY


def test_render_summary_empty_with_no_files(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    assert render_summary(tmp_path / "cache") == EMPTY_SUMMARY


def test_render_summary_lists_subdirectory_counts(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "agent").mkdir(parents=True)
    (cache / "state").mkdir(parents=True)
    (cache / "judge").mkdir(parents=True)
    (cache / "agent" / "a.json").write_text("{}", encoding="utf-8")
    (cache / "agent" / "b.json").write_text("{}", encoding="utf-8")
    (cache / "state" / "s.json").write_text("{}", encoding="utf-8")

    summary = render_summary(cache)

    assert summary != EMPTY_SUMMARY
    assert "agent" in summary
    assert "state" in summary
    assert "2" in summary
    assert "1" in summary
    # Directory with no files is omitted.
    assert "judge" not in summary


def test_results_command_writes_summary(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    readme = tmp_path / "README.md"
    readme.write_text(MARKERS.format(body="placeholder"), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "results",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--results-dir",
            str(results_dir),
            "--readme",
            str(readme),
        ],
    )

    assert result.exit_code == 0
    summary_file = results_dir / "summary.md"
    assert summary_file.is_file()
    summary_text = summary_file.read_text(encoding="utf-8")
    assert EMPTY_SUMMARY in summary_text
    assert summary_text.count("Not run yet") == 10
    readme_text = readme.read_text(encoding="utf-8")
    assert EMPTY_SUMMARY in readme_text
    assert "### Outcome judging" in readme_text


def test_write_findings_writes_gate_file(tmp_path: Path) -> None:
    path = write_findings(tmp_path / "results", "g3", "code judge wins")

    assert path == tmp_path / "results" / "g3_findings.md"
    assert path.read_text(encoding="utf-8") == "code judge wins\n"


def test_render_results_on_fixture_renders_all_sections(tmp_path: Path) -> None:
    body = render_results(FIXTURE, tmp_path / "missing")

    # Every gate title appears, in order.
    positions = [body.index(f"### {title}") for title in GATE_TITLES]
    assert positions == sorted(positions)

    # G3 has a table, findings, and a chart image.
    assert "| judge_id" in body
    assert "0.95" in body
    assert "The code judge reaches 0.95 accuracy" in body
    assert "![Outcome judging](results/g3_accuracy.png)" in body

    # G2 has no table yet.
    assert "Not run yet" in body
    assert body.count("Not run yet") == 7

    # Threats and Reproduce close the fragment.
    assert "### Threats to validity" in body
    assert "synthetic" in body
    assert "### Reproduce" in body
    assert "make results" in body


def test_render_results_empty_dir_has_ten_not_run(tmp_path: Path) -> None:
    body = render_results(tmp_path / "results", tmp_path / "cache")

    assert body.count("Not run yet") == 10
    for title in GATE_TITLES:
        assert f"### {title}" in body


def test_render_results_uses_analyze_hint_for_analysis_gates(tmp_path: Path) -> None:
    body = render_results(tmp_path / "results", tmp_path / "cache")

    assert "judges analyze --gate g5" in body
    assert "judges analyze --gate g6" in body
    assert "judges analyze --gate g8" in body
    assert "judges judge --gate g2" in body
