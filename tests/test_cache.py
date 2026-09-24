"""Tests for the content-addressed JSON cache."""

from pathlib import Path

from pydantic import BaseModel

from decision_judges.cache import Cache, cache_key


class Sample(BaseModel):
    """Tiny model used to exercise the cache."""

    name: str
    value: int


def test_cache_key_is_deterministic() -> None:
    first = cache_key("j", "m", "v1", "s", 0)
    second = cache_key("j", "m", "v1", "s", 0)
    assert first == second
    assert len(first) == 64


def test_cache_key_changes_with_each_component() -> None:
    base = cache_key("j", "m", "v1", "s", 0)
    assert cache_key("J", "m", "v1", "s", 0) != base
    assert cache_key("j", "M", "v1", "s", 0) != base
    assert cache_key("j", "m", "v2", "s", 0) != base
    assert cache_key("j", "m", "v1", "S", 0) != base
    assert cache_key("j", "m", "v1", "s", 1) != base


def test_get_or_call_computes_once(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = cache_key("j", "m", "v1", "s", 0)
    calls = {"n": 0}

    def fn() -> Sample:
        calls["n"] += 1
        return Sample(name="x", value=1)

    first = cache.get_or_call(key, Sample, fn)
    second = cache.get_or_call(key, Sample, fn)
    assert first == second == Sample(name="x", value=1)
    assert calls["n"] == 1


def test_file_lands_at_expected_path(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = cache_key("j", "m", "v1", "s", 0)
    cache.get_or_call(key, Sample, lambda: Sample(name="x", value=1))
    assert (tmp_path / key[:2] / f"{key}.json").is_file()


def test_corrupted_json_is_quarantined_and_recomputed(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = cache_key("j", "m", "v1", "s", 0)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    calls = {"n": 0}

    def fn() -> Sample:
        calls["n"] += 1
        return Sample(name="fresh", value=2)

    result = cache.get_or_call(key, Sample, fn)
    assert result == Sample(name="fresh", value=2)
    assert calls["n"] == 1
    assert (tmp_path / "_quarantine" / f"{key}.json").is_file()
    assert path.is_file()


def test_valid_json_failing_validation_is_quarantined(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = cache_key("j", "m", "v1", "s", 0)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"unexpected": true}')

    result = cache.get_or_call(key, Sample, lambda: Sample(name="fresh", value=3))
    assert result == Sample(name="fresh", value=3)
    assert (tmp_path / "_quarantine" / f"{key}.json").is_file()


def test_exists(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = cache_key("j", "m", "v1", "s", 0)
    assert not cache.exists(key)
    cache.get_or_call(key, Sample, lambda: Sample(name="x", value=1))
    assert cache.exists(key)


def test_iter_keys(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key_a = cache_key("j", "m", "v1", "a", 0)
    key_b = cache_key("j", "m", "v1", "b", 0)
    cache.get_or_call(key_a, Sample, lambda: Sample(name="a", value=1))
    cache.get_or_call(key_b, Sample, lambda: Sample(name="b", value=2))

    assert set(cache.iter_keys()) == {key_a, key_b}
    assert list(cache.iter_keys(prefix=key_a[:6])) == [key_a]


def test_iter_keys_excludes_quarantine(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = cache_key("j", "m", "v1", "s", 0)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    cache.get_or_call(key, Sample, lambda: Sample(name="fresh", value=2))
    assert list(cache.iter_keys()) == [key]


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    key = cache_key("j", "m", "v1", "s", 0)
    cache.get_or_call(key, Sample, lambda: Sample(name="x", value=1))
    files = [p.name for p in (tmp_path / key[:2]).iterdir()]
    assert files == [f"{key}.json"]
