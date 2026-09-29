"""Render result tables, charts, and README summaries from cached verdicts."""

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from matplotlib.figure import Figure

_START = "<!-- results:start -->"
_END = "<!-- results:end -->"
_EMPTY_SUMMARY = "No cached results yet. Tables and charts appear here as evaluation stages run."
_THREATS_PATH = Path(__file__).resolve().parents[1] / "docs" / "threats.md"


class _GateSpec(NamedTuple):
    """A gate's rendering plan: id, section title, table stems, chart stem, kind."""

    gate_id: str
    title: str
    tables: tuple[str, ...]
    chart: str
    analysis: bool


_GATES: tuple[_GateSpec, ...] = (
    _GateSpec(
        "g1", "Task triage before any run", ("g1_summary",), "g1_difficulty_vs_actions", False
    ),
    _GateSpec("g2", "Per-step tool-call scoring", ("g2_summary",), "g2_auroc", False),
    _GateSpec("g3", "Outcome judging", ("g3_summary",), "g3_accuracy", False),
    _GateSpec(
        "g4",
        "Atomic questions versus one broad question",
        ("g4_summary",),
        "g4_auroc_by_aggregator",
        False,
    ),
    _GateSpec(
        "g5", "Confidence-gated cascade", ("g5_frontier", "g5_reference"), "g5_frontier", True
    ),
    _GateSpec("g6", "Calibration", ("g6_summary",), "g6_reliability", True),
    _GateSpec(
        "g7", "Robustness to evaluator-directed text", ("g7_summary",), "g7_flip_rates", False
    ),
    _GateSpec("g8", "Regression detection", ("g8_summary",), "g8_intervals", True),
    _GateSpec("g9", "Failure taxonomy against hand labels", ("g9_summary",), "g9_accuracy", False),
    _GateSpec(
        "g10",
        "Local decision model: zero-shot versus fine-tuned",
        ("g10_summary", "g10_folds"),
        "g10_zero_shot_vs_finetuned",
        False,
    ),
)

_REPRODUCE = """### Reproduce

Render the committed results offline:

```
uv sync
make results
```

Regenerate every result from scratch, in order:

```
judges run-agent --variant baseline
judges run-agent --variant degraded
judges serialize
judges judge --gate g3 --profile full --judges code,llm_cheap,llm_strong,jev
judges judge --gate g3 --profile compact --judges code,llm_cheap,llm_strong,jev
judges judge --gate g3 --profile compact --judges laya_base
judges finetune-laya
judges judge --gate g3 --profile compact --judges laya_ft
judges judge --gate g2
judges judge --gate g4
judges judge --gate g7
judges judge --gate g1
judges judge --gate g9
judges judge --gate g10
judges analyze --gate g3 --profile full --profile compact
judges analyze --gate g5
judges analyze --gate g6
judges analyze --gate g8
make results
```
"""


def write_table(results_dir: Path, name: str, df: pd.DataFrame) -> tuple[Path, Path]:
    """Write a DataFrame as GitHub Markdown and CSV, returning both paths."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / f"{name}.md"
    csv_path = results_dir / f"{name}.csv"
    md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    df.to_csv(csv_path, index=False)
    return md_path, csv_path


def write_chart(results_dir: Path, name: str, fig: Figure) -> tuple[Path, Path]:
    """Save a figure as PNG and SVG, close it, and return both paths."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    png_path = results_dir / f"{name}.png"
    svg_path = results_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    fig.clf()
    return png_path, svg_path


def update_readme(readme_path: Path, body: str) -> None:
    """Replace the text between the result markers with the given body."""
    readme_path = Path(readme_path)
    text = readme_path.read_text(encoding="utf-8")
    start = text.find(_START)
    end = text.find(_END)
    if start == -1 or end == -1 or start > end:
        raise ValueError("README is missing well-ordered result markers")
    prefix = text[: start + len(_START)]
    suffix = text[end:]
    readme_path.write_text(f"{prefix}\n{body}\n{suffix}", encoding="utf-8")


def render_summary(cache_root: Path) -> str:
    """Return a Markdown fragment summarizing cached files per subdirectory."""
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        return _EMPTY_SUMMARY
    lines: list[str] = []
    for entry in sorted(cache_root.iterdir()):
        if not entry.is_dir():
            continue
        count = sum(1 for path in entry.rglob("*") if path.is_file())
        if count:
            lines.append(f"- `{entry.name}`: {count} cached files")
    if not lines:
        return _EMPTY_SUMMARY
    return "\n".join(lines)


def write_findings(results_dir: Path, gate_id: str, text: str) -> Path:
    """Write a gate's findings text to ``<gate_id>_findings.md`` and return the path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{gate_id}_findings.md"
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    return path


def _read_optional(path: Path) -> str | None:
    """Return a file's stripped text, or None when it is absent."""
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def _not_run_line(spec: _GateSpec) -> str:
    """Return the one-line placeholder for a gate with no results yet."""
    command = (
        f"judges analyze --gate {spec.gate_id}"
        if spec.analysis
        else (f"judges judge --gate {spec.gate_id} ...")
    )
    return f"Not run yet. Produce it with `{command}`."


def _gate_section(spec: _GateSpec, results_dir: Path) -> str:
    """Render one gate's section from whatever files exist under ``results_dir``."""
    lines = [f"### {spec.title}", ""]
    primary = results_dir / f"{spec.tables[0]}.md"
    if not primary.is_file():
        lines.append(_not_run_line(spec))
        return "\n".join(lines)

    findings = _read_optional(results_dir / f"{spec.gate_id}_findings.md")
    if findings:
        lines.extend([findings, ""])
    for stem in spec.tables:
        table = _read_optional(results_dir / f"{stem}.md")
        if table:
            lines.extend([table, ""])
    if (results_dir / f"{spec.chart}.png").is_file():
        lines.append(f"![{spec.title}](results/{spec.chart}.png)")
    return "\n".join(lines).rstrip()


def render_results(results_dir: Path, cache_root: Path) -> str:
    """Build the README fragment: an intro, one section per gate, threats, and reproduce steps.

    A gate whose primary table is present is rendered with its persisted
    findings, tables, and chart image; a gate without results renders a single
    placeholder line. Sections follow the fixed gate order.
    """
    results_dir = Path(results_dir)
    rendered = sum(1 for spec in _GATES if (results_dir / f"{spec.tables[0]}.md").is_file())
    intro = f"{rendered}/{len(_GATES)} gates have results below. Cached inputs:"
    parts = [intro, "", render_summary(cache_root)]
    for spec in _GATES:
        parts.extend(["", _gate_section(spec, results_dir)])
    threats = (
        _read_optional(_THREATS_PATH) or "Threats to validity are documented in docs/threats.md."
    )
    parts.extend(["", "### Threats to validity", "", threats])
    parts.extend(["", _REPRODUCE.rstrip()])
    return "\n".join(parts)
