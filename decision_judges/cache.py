"""Content-addressed JSON cache for reproducible judge calls."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_QUARANTINE = "_quarantine"


def cache_key(
    judge_id: str,
    model_id: str,
    prompt_version: str,
    state_hash: str,
    repeat: int,
) -> str:
    """Return the SHA-256 hex digest identifying a single judge call."""
    payload = "|".join([judge_id, model_id, prompt_version, state_hash, str(repeat)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    """Stores Pydantic models as stable JSON under a content-addressed tree."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        """Return the on-disk path for a key."""
        return self.root / key[:2] / f"{key}.json"

    def exists(self, key: str) -> bool:
        """Return whether a stored value exists for a key."""
        return self._path(key).is_file()

    def get_or_call(self, key: str, model_type: type[T], fn: Callable[[], T]) -> T:
        """Return the cached value for a key, computing and storing it on a miss.

        A stored file that fails JSON parsing or model validation is moved to
        quarantine and recomputed.
        """
        path = self._path(key)
        if path.is_file():
            try:
                return model_type.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, ValidationError):
                self._quarantine(key, path)

        value = fn()
        self._write(path, value)
        return value

    def peek(self, key: str, model_type: type[T]) -> T | None:
        """Return the cached value for a key without computing on a miss.

        Return None when no file exists or the stored file fails JSON parsing or
        model validation. Unlike ``get_or_call`` this never computes, quarantines,
        or writes, so a caller can inspect a stored value before deciding to keep
        or replace it.
        """
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, ValidationError):
            return None

    def put(self, key: str, value: BaseModel) -> None:
        """Write a value for a key atomically, replacing any existing file."""
        self._write(self._path(key), value)

    def iter_keys(self, prefix: str = "") -> Iterator[str]:
        """Yield stored keys, excluding quarantined files, matching an optional prefix."""
        if not self.root.is_dir():
            return
        for shard in sorted(self.root.iterdir()):
            if not shard.is_dir() or shard.name == _QUARANTINE:
                continue
            for entry in sorted(shard.glob("*.json")):
                key = entry.stem
                if key.startswith(prefix):
                    yield key

    def _quarantine(self, key: str, path: Path) -> None:
        """Move a corrupt or invalid file into the quarantine directory."""
        target_dir = self.root / _QUARANTINE
        target_dir.mkdir(parents=True, exist_ok=True)
        os.replace(path, target_dir / f"{key}.json")

    def _write(self, path: Path, value: BaseModel) -> None:
        """Write a model to a path atomically as stable, sorted JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp_name, path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
