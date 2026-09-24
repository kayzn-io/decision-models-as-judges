"""Command-line interface for decision_judges."""

import typer

from decision_judges import __version__

app = typer.Typer(help="Decision models as judges.")


@app.callback()
def main() -> None:
    """Decision models as judges."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
