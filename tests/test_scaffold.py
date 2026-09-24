"""Scaffold-level tests for package import, version, and CLI."""

from typer.testing import CliRunner

import decision_judges
from decision_judges.cli import app


def test_import() -> None:
    assert decision_judges is not None


def test_version_is_str() -> None:
    assert isinstance(decision_judges.__version__, str)


def test_cli_has_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert decision_judges.__version__ in result.stdout
