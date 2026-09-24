"""Guard tests enforcing the thin-view rule for the UI package.

The UI package is a read-only view over cached data. These tests parse every
module under ``decision_judges/ui`` with ``ast`` and assert that:

* no module imports ``numpy``, ``sklearn``, or ``scipy``;
* every import resolves to the standard library, a small third-party display
  allowlist, or the ``decision_judges`` package itself; and
* no module defines a metric-shaped function or computes a metric inline.

The two metric checks are deliberate heuristics, not proofs. A ``def`` whose
name starts with a known metric prefix, or a ``def`` whose body calls
``statistics.*`` or ``math.fsum``, is treated as inline metric computation that
belongs in ``decision_judges.metrics``. They catch the common cases cheaply
rather than detecting every possible statistic.
"""

import ast
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_DIR = _REPO_ROOT / "decision_judges" / "ui"

_FORBIDDEN = ("numpy", "sklearn", "scipy")
_THIRD_PARTY_ALLOWLIST = frozenset({"streamlit", "pandas", "matplotlib", "pydantic", "typer"})
_STDLIB = frozenset(sys.stdlib_module_names)
_INTERNAL_ROOT = "decision_judges"

_METRIC_PREFIXES = (
    "kappa",
    "auroc",
    "ece",
    "brier",
    "accuracy",
    "precision",
    "recall",
    "bootstrap",
    "flip_rate",
    "spearman",
)


def _ui_files() -> list[Path]:
    """Return every ``*.py`` file under the UI package, sorted."""
    return sorted(_UI_DIR.rglob("*.py"))


def _parse(path: Path) -> ast.Module:
    """Parse a file into an AST module."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _rel(path: Path) -> str:
    """Return a path relative to the repository root for messages."""
    return path.relative_to(_REPO_ROOT).as_posix()


def _imported_roots(tree: ast.Module) -> set[str]:
    """Return the root module of every absolute import in a tree.

    Relative imports (``level`` above zero) always resolve inside the current
    package and are omitted.
    """
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _function_defs(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return every function definition anywhere in a tree."""
    return [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _is_inline_metric_call(func: ast.expr) -> bool:
    """Return whether a call target is ``statistics.*`` or ``math.fsum``."""
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return False
    module = func.value.id
    return module == "statistics" or (module == "math" and func.attr == "fsum")


def test_ui_modules_do_not_import_scientific_libraries() -> None:
    """No UI module imports numpy, sklearn, or scipy."""
    hits: list[str] = []
    for path in _ui_files():
        for root in sorted(_imported_roots(_parse(path))):
            if root in _FORBIDDEN:
                hits.append(f"{_rel(path)}: {root}")
    assert not hits, "forbidden imports: " + ", ".join(hits)


def test_ui_imports_are_stdlib_allowlisted_or_internal() -> None:
    """Every UI import is stdlib, on the display allowlist, or internal."""
    hits: list[str] = []
    for path in _ui_files():
        for root in sorted(_imported_roots(_parse(path))):
            if root == _INTERNAL_ROOT or root in _THIRD_PARTY_ALLOWLIST or root in _STDLIB:
                continue
            hits.append(f"{_rel(path)}: {root}")
    assert not hits, "non-allowlisted imports: " + ", ".join(hits)


def test_ui_modules_define_no_metric_functions() -> None:
    """No UI function is named like a metric or computes one inline."""
    name_hits: list[str] = []
    call_hits: list[str] = []
    for path in _ui_files():
        for func in _function_defs(_parse(path)):
            if func.name.startswith(_METRIC_PREFIXES):
                name_hits.append(f"{_rel(path)}: {func.name}")
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and _is_inline_metric_call(node.func):
                    call_hits.append(f"{_rel(path)}: {func.name}")
    assert not name_hits, "metric-named functions: " + ", ".join(name_hits)
    assert not call_hits, "inline metric computation: " + ", ".join(sorted(set(call_hits)))
