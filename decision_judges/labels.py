"""Human taxonomy labels for failing trajectories and a JSONL store.

The store is append-only and intended for a single writer: each label is one
JSON line, so appends never rewrite prior lines and a crash mid-write leaves at
most one partial trailing line that ``load`` skips.
"""

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

TAXONOMY: tuple[str, ...] = (
    "wrong_or_missing_action",
    "unrequested_write",
    "skipped_confirmation",
    "identity_not_verified",
    "wrong_arguments",
    "premature_end",
    "policy_misapplied",
    "other",
)

TAXONOMY_DESCRIPTIONS: dict[str, str] = {
    "wrong_or_missing_action": (
        "The agent took an action the task did not call for, or skipped one it required."
    ),
    "unrequested_write": (
        "The agent changed an order, account, or record the customer never asked it to change."
    ),
    "skipped_confirmation": (
        "The agent made a change without first confirming the details with the customer."
    ),
    "identity_not_verified": (
        "The agent acted on the account without verifying the customer's identity first."
    ),
    "wrong_arguments": (
        "The agent called the right tool but with wrong arguments, such as the wrong order id "
        "or amount."
    ),
    "premature_end": (
        "The agent ended the conversation before the customer's request was resolved."
    ),
    "policy_misapplied": (
        "The agent applied the retail policy incorrectly, such as refunding an item outside its "
        "return window."
    ),
    "other": "The failure does not fit any of the specific categories above.",
}


def label_title(label: str) -> str:
    """Return a human title for a taxonomy label ('Wrong or missing action')."""
    return label.replace("_", " ").capitalize()


_STORE_RELATIVE = Path("labels") / "taxonomy.jsonl"


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class Label(BaseModel):
    """One human label attached to a ``(variant, task_id)`` trajectory."""

    variant: str
    task_id: str
    label: str
    note: str = ""
    labeler: str = "owner"
    created_at: str = Field(default_factory=_now_iso)

    @field_validator("label")
    @classmethod
    def _known_label(cls, value: str) -> str:
        """Reject a label that is not one of the taxonomy members."""
        if value not in TAXONOMY:
            raise ValueError(f"unknown label: {value!r}")
        return value


class LabelStore:
    """A JSONL-backed store of labels rooted at one file path."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def under(cls, data_dir: Path) -> "LabelStore":
        """Return a store at the default path under a data directory."""
        return cls(data_dir / _STORE_RELATIVE)

    def append(self, label: Label) -> None:
        """Append one label as a JSON line, creating the file when absent."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(label.model_dump_json() + "\n")
            handle.flush()

    def load(self) -> list[Label]:
        """Return every stored label in file order, skipping blank lines."""
        if not self.path.is_file():
            return []
        labels: list[Label] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labels.append(Label.model_validate_json(line))
        return labels

    def latest(self) -> dict[tuple[str, str], Label]:
        """Return the newest label per ``(variant, task_id)`` by ``created_at``."""
        newest: dict[tuple[str, str], Label] = {}
        for label in self.load():
            key = (label.variant, label.task_id)
            current = newest.get(key)
            if current is None or label.created_at >= current.created_at:
                newest[key] = label
        return newest

    def progress(self, target: int = 50) -> tuple[int, int]:
        """Return the count of distinct labeled pairs and the target."""
        return len(self.latest()), target


def label_counts(labels: Sequence[Label]) -> list[tuple[str, int]]:
    """Return each used taxonomy label as a title and its count, in taxonomy order."""
    counter = Counter(label.label for label in labels)
    return [(label_title(name), counter[name]) for name in TAXONOMY if counter[name]]
