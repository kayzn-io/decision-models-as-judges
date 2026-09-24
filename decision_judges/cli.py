"""Command-line interface for decision_judges."""

from pathlib import Path
from typing import Annotated

import typer

from decision_judges import __version__
from decision_judges.report import render_summary, update_readme

app = typer.Typer(help="Decision models as judges.")


@app.callback()
def main() -> None:
    """Decision models as judges."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def results(
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("cache"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
    readme: Annotated[Path, typer.Option("--readme")] = Path("README.md"),
) -> None:
    """Render a results summary from the cache and refresh the README."""
    summary = render_summary(cache_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    if readme.is_file():
        update_readme(readme, summary)
        typer.echo(f"Wrote {summary_path} and updated {readme}")
    else:
        typer.echo(f"Wrote {summary_path}")
