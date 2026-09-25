"""Cached, read-only loaders that back the Streamlit app.

Every function reads from a fixed set of directories (``cache/``, ``results/``,
``config/``, ``data/``) under one root. File-reading loaders are wrapped in
``st.cache_data`` keyed on a directory fingerprint so a changed file
invalidates the cache without a manual clear.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from decision_judges import pipeline
from decision_judges.bench.load import Task, load_tasks
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.config import PricingTable, StudyConfig, load_pricing, load_study
from decision_judges.gates.base import Gate, Item
from decision_judges.labels import LabelStore
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Ledger
from decision_judges.types import Verdict

_VARIANTS = ("baseline", "degraded")

_QUARANTINE = "_quarantine"
_MISSING_PRICE = ""
_DEFAULT_THREATS = "The threats-to-validity list is published alongside the results."


@dataclass(frozen=True)
class Paths:
    """Resolved directories the app reads from, rooted at one study directory."""

    repo_root: Path
    cache_dir: Path
    results_dir: Path
    config_dir: Path
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Paths":
        """Build paths from ``JUDGES_ROOT``, defaulting to the current directory."""
        root = Path(os.environ.get("JUDGES_ROOT", ".")).resolve()
        return cls(
            repo_root=root,
            cache_dir=root / "cache",
            results_dir=root / "results",
            config_dir=root / "config",
            data_dir=root / "data",
        )


def _iter_files(path: Path) -> list[Path]:
    """Return every file under a directory, sorted, or empty when absent."""
    if not path.is_dir():
        return []
    return sorted(entry for entry in path.rglob("*") if entry.is_file())


def dir_fingerprint(path: Path) -> str:
    """Return a sha256 over each file's relative path, size, and mtime.

    The digest changes whenever a file under the directory is added, removed,
    resized, or rewritten, so it serves as a cache key for the loaders.
    """
    hasher = hashlib.sha256()
    for file in _iter_files(path):
        stat = file.stat()
        rel = file.relative_to(path).as_posix()
        hasher.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return hasher.hexdigest()


@st.cache_data(show_spinner=False)
def _load_study_and_pricing(config_dir: str, _fingerprint: str) -> tuple[StudyConfig, PricingTable]:
    """Load the study and pricing configs from a directory."""
    base = Path(config_dir)
    return load_study(base / "study.toml"), load_pricing(base / "pricing.toml")


def load_study_and_pricing(paths: Paths) -> tuple[StudyConfig, PricingTable]:
    """Load the study configuration and pricing table."""
    return _load_study_and_pricing(str(paths.config_dir), dir_fingerprint(paths.config_dir))


@st.cache_data(show_spinner=False)
def _load_agent_records(agent_dir: str, _fingerprint: str) -> dict[tuple[str, str], AgentRecord]:
    """Load agent records from every variant under a directory."""
    records: dict[tuple[str, str], AgentRecord] = {}
    for file in _iter_files(Path(agent_dir)):
        try:
            record = AgentRecord.model_validate_json(file.read_text(encoding="utf-8"))
        except (ValueError, ValidationError):
            continue
        records[(record.variant, record.task_id)] = record
    return records


def load_agent_records(paths: Paths) -> dict[tuple[str, str], AgentRecord]:
    """Load agent records keyed by ``(variant, task_id)`` across all variants."""
    agent_dir = paths.cache_dir / "agent"
    return _load_agent_records(str(agent_dir), dir_fingerprint(agent_dir))


@st.cache_data(show_spinner=False)
def _load_states(
    state_dir: str, _fingerprint: str
) -> dict[tuple[str, str, str, str, int | None], StateRecord]:
    """Load serialized state records from a directory, step states included."""
    states: dict[tuple[str, str, str, str, int | None], StateRecord] = {}
    for file in _iter_files(Path(state_dir)):
        try:
            state = StateRecord.model_validate_json(file.read_text(encoding="utf-8"))
        except (ValueError, ValidationError):
            continue
        key = (
            state.variant,
            state.profile.value,
            state.injection.value,
            state.task_id,
            state.step_index,
        )
        states[key] = state
    return states


def load_states(paths: Paths) -> dict[tuple[str, str, str, str, int | None], StateRecord]:
    """Load state records keyed by ``(variant, profile, injection, task_id, step_index)``.

    ``step_index`` is ``None`` for whole-trajectory states and an integer for the
    per-step states the G2 gate serializes, so the two never collide.
    """
    state_dir = paths.cache_dir / "state"
    return _load_states(str(state_dir), dir_fingerprint(state_dir))


def trajectory_states(
    states: dict[tuple[str, str, str, str, int | None], StateRecord],
) -> dict[tuple[str, str, str, str], StateRecord]:
    """Return the whole-trajectory states keyed by ``(variant, profile, injection, task_id)``."""
    return {
        (variant, profile, injection, task_id): state
        for (variant, profile, injection, task_id, step_index), state in states.items()
        if step_index is None
    }


def step_states(
    states: dict[tuple[str, str, str, str, int | None], StateRecord],
    variant: str,
    profile: str,
    task_id: str,
) -> dict[int, StateRecord]:
    """Return one variant, profile, and task's step states keyed by step index."""
    result: dict[int, StateRecord] = {}
    for (v, prof, _injection, tid, step_index), state in states.items():
        if step_index is not None and v == variant and prof == profile and tid == task_id:
            result[step_index] = state
    return result


@st.cache_data(show_spinner=False)
def _load_tasks_for_ui(data_dir: str, _fingerprint: str) -> dict[str, Task]:
    """Load exported tasks keyed by task id, empty when the file is absent."""
    path = Path(data_dir) / "tasks.json"
    if not path.is_file():
        return {}
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return {task.task_id: task for task in load_tasks(source=source)}


def load_tasks_for_ui(paths: Paths) -> dict[str, Task]:
    """Load the exported task definitions keyed by task id, empty when absent."""
    return _load_tasks_for_ui(str(paths.data_dir), dir_fingerprint(paths.data_dir))


@st.cache_data(show_spinner=False)
def _load_verdicts(judge_dir: str, _fingerprint: str) -> list[Verdict]:
    """Load verdicts from a judge cache, skipping quarantine and invalid files."""
    base = Path(judge_dir)
    verdicts: list[Verdict] = []
    skipped = 0
    for file in _iter_files(base):
        if _QUARANTINE in file.relative_to(base).parts:
            continue
        try:
            verdicts.append(Verdict.model_validate_json(file.read_text(encoding="utf-8")))
        except (ValueError, ValidationError):
            skipped += 1
    return verdicts


def load_verdicts(paths: Paths) -> list[Verdict]:
    """Load verdicts from the judge cache, skipping quarantined and invalid files.

    The cache holds both whole-trajectory and per-step verdicts in one
    content-addressed tree, so the returned list carries both kinds.
    """
    judge_dir = paths.cache_dir / "judge"
    return _load_verdicts(str(judge_dir), dir_fingerprint(judge_dir))


def verdicts_by_state(verdicts: list[Verdict]) -> dict[str, list[Verdict]]:
    """Group verdicts by the state hash they judged."""
    grouped: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        grouped.setdefault(verdict.state_hash, []).append(verdict)
    return grouped


@st.cache_data(show_spinner=False)
def _load_results_table(results_dir: str, name: str, _fingerprint: str) -> pd.DataFrame | None:
    """Read a results table CSV by name, returning None when it is absent or invalid."""
    path = Path(results_dir) / f"{name}.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (ValueError, OSError):
        return None


def load_results_table(paths: Paths, name: str) -> pd.DataFrame | None:
    """Return the named results table as a DataFrame, or None when it is absent."""
    return _load_results_table(str(paths.results_dir), name, dir_fingerprint(paths.results_dir))


def load_chart_path(paths: Paths, name: str) -> Path | None:
    """Return the path to the named results chart PNG, or None when it is absent."""
    path = paths.results_dir / f"{name}.png"
    return path if path.is_file() else None


@st.cache_data(show_spinner=False)
def _load_items_and_verdicts(
    cache_dir: str, profile: str, _fingerprint: str
) -> tuple[list[Item], list[Verdict]]:
    """Build items across both variants and the verdicts that judge them."""
    base = Path(cache_dir)
    items = pipeline.items_for_variants(
        base / "state", base / "agent", StateProfile(profile), list(_VARIANTS)
    )
    verdicts, _ = pipeline.load_verdicts(base / "judge")
    return items, pipeline.filter_verdicts_to_items(verdicts, items)


def load_items_and_verdicts(paths: Paths, profile: str) -> tuple[list[Item], list[Verdict]]:
    """Return items across both variants and the verdicts the interactive gates reduce."""
    return _load_items_and_verdicts(str(paths.cache_dir), profile, dir_fingerprint(paths.cache_dir))


@st.cache_data(show_spinner=False)
def _load_injected_items_and_verdicts(
    cache_dir: str, profile: str, _fingerprint: str
) -> tuple[list[Item], list[Verdict]]:
    """Build whole-trajectory and injected items and the verdicts that judge them."""
    base = Path(cache_dir)
    state_profile = StateProfile(profile)
    items = pipeline.items_for_variants(
        base / "state", base / "agent", state_profile, list(_VARIANTS)
    )
    for variant in _VARIANTS:
        records, _ = pipeline.load_agent_records(base / "agent", variant)
        injected = pipeline.read_injected_states(base / "state", variant, state_profile)
        items.extend(pipeline.items_from_states(list(injected.values()), records))
    verdicts, _ = pipeline.load_verdicts(base / "judge")
    return items, pipeline.filter_verdicts_to_items(verdicts, items)


def load_injected_items_and_verdicts(
    paths: Paths, profile: str
) -> tuple[list[Item], list[Verdict]]:
    """Return whole and injected items and their verdicts for the robustness gate.

    The items union the whole-trajectory states across both variants with the
    injected copies of the failing runs, so the G7 gate can compare each judge's
    original verdict against its verdicts on the injected copies.
    """
    return _load_injected_items_and_verdicts(
        str(paths.cache_dir), profile, dir_fingerprint(paths.cache_dir)
    )


def analysis_gates(paths: Paths) -> dict[str, Gate]:
    """Return the analysis-only gates constructed from the study and pricing."""
    study, pricing = load_study_and_pricing(paths)
    return pipeline.analysis_registry(study, pricing)


@st.cache_data(show_spinner=False)
def _load_ledger(spend_path: str, _fingerprint: str) -> Ledger | None:
    """Load the spend ledger from a path if it exists and validates."""
    path = Path(spend_path)
    if not path.is_file():
        return None
    try:
        return Ledger.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, ValidationError):
        return None


def load_ledger(paths: Paths) -> Ledger | None:
    """Load the spend ledger, returning None when it is absent or invalid."""
    return _load_ledger(str(paths.results_dir / "spend.json"), dir_fingerprint(paths.results_dir))


def _price_columns(pricing: PricingTable, model_id: str) -> tuple[object, object]:
    """Return input and output per-million-token prices, blank when unpriced."""
    price = pricing.models.get(model_id)
    if price is None:
        return _MISSING_PRICE, _MISSING_PRICE
    return price.input_per_mtok, price.output_per_mtok


def judge_roster(study: StudyConfig, pricing: PricingTable) -> pd.DataFrame:
    """Return the judges and their model ids and per-million-token prices."""
    models = study.models
    rows: list[dict[str, object]] = [
        {"judge": "code", "model_id": "none", "input_per_mtok": 0.0, "output_per_mtok": 0.0}
    ]
    for judge, model_id in (
        ("llm_cheap", models.llm_cheap),
        ("llm_strong", models.llm_strong),
        ("jev", models.jev),
        ("laya", models.laya.repo_id),
    ):
        input_price, output_price = _price_columns(pricing, model_id)
        rows.append(
            {
                "judge": judge,
                "model_id": model_id,
                "input_per_mtok": input_price,
                "output_per_mtok": output_price,
            }
        )
    return pd.DataFrame(rows, columns=["judge", "model_id", "input_per_mtok", "output_per_mtok"])


def spend_by_stage(ledger: Ledger) -> pd.DataFrame:
    """Return settled spend per stage as a two-column table."""
    rows = [{"stage": stage, "usd": usd} for stage, usd in sorted(ledger.per_stage.items())]
    return pd.DataFrame(rows, columns=["stage", "usd"])


def total_spend(ledger: Ledger | None) -> float:
    """Return the settled spend across all stages, zero when no ledger exists."""
    if ledger is None:
        return 0.0
    return sum(ledger.per_stage.values())


def is_local() -> bool:
    """Return whether the app runs in local mode that enables writing pages."""
    return os.environ.get("JUDGES_LOCAL") == "1"


def labels_store(paths: Paths) -> LabelStore:
    """Return the writable label store under the data directory.

    This is deliberately uncached: the store writes, so callers need the live
    file rather than a memoized snapshot.
    """
    return LabelStore.under(paths.data_dir)


def failing_trajectories(paths: Paths) -> list[tuple[str, str]]:
    """Return the ``(variant, task_id)`` pairs of runs that did not fully pass."""
    records = load_agent_records(paths)
    return sorted(key for key, record in records.items() if record.reward < 1.0)


def frontier_chart(paths: Paths) -> Path | None:
    """Return the cascade frontier chart PNG under results, if one is present."""
    for file in _iter_files(paths.results_dir):
        stem = file.stem.lower()
        if file.suffix == ".png" and "g5" in stem and "frontier" in stem:
            return file
    return None


def threats_text(paths: Paths) -> str:
    """Return the threats-to-validity text, or a note when it is not published."""
    threats = paths.repo_root / "docs" / "threats.md"
    if threats.is_file():
        return threats.read_text(encoding="utf-8")
    return _DEFAULT_THREATS
