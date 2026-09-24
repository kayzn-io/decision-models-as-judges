"""Render result tables, charts, and README summaries from cached verdicts."""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

_START = "<!-- results:start -->"
_END = "<!-- results:end -->"
_EMPTY_SUMMARY = "No cached results yet. Tables and charts appear here as evaluation stages run."


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
