"""Unit tests for the data-loading module (config.py + src/data_loader.py).

Uses small temporary CSV fixtures for fast, deterministic tests of the core
``_read_split`` reader, plus checks against the real dataset for confirmed
dimensions and configured path resolution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from src.data_loader import (
    DataLoaderError,
    DatasetSplits,
    _read_split,
    load_all_splits,
    load_test,
    load_train,
    load_validation,
    resolve_dataset_dir,
    validate_structure,
)

TRAIN_CSV_STRAY = (
    "Column1,Column2\n"
    "text,labels\n"
    "ગુજરાતી સમાચાર,State\n"
    "hello world,Business\n"
)
SIMPLE_CSV = "text,labels\nગુજરાતી સમાચાર,State\nhello world,Business\n"


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _monkeypatch_paths(monkeypatch, tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    test = tmp_path / "test.csv"
    # Recovered CSVs start directly with the "text,labels" header (no stray
    # row), so all fixtures use the same simple layout.
    _write_csv(train, SIMPLE_CSV)
    _write_csv(valid, SIMPLE_CSV)
    _write_csv(test, SIMPLE_CSV)
    monkeypatch.setattr(config, "TRAIN_PATH", train)
    monkeypatch.setattr(config, "VALID_PATH", valid)
    monkeypatch.setattr(config, "TEST_PATH", test)


def test_read_split_loads_fixture(tmp_path):
    p = tmp_path / "d.csv"
    _write_csv(p, SIMPLE_CSV)
    df = _read_split(p, encoding="utf-8", header=0)
    assert list(df.columns) == ["text", "labels"]
    assert len(df) == 2
    assert df.iloc[0]["text"] == "ગુજરાતી સમાચાર"
    assert df.iloc[0]["labels"] == "State"


def test_read_split_missing_file():
    with pytest.raises(DataLoaderError, match="not found"):
        _read_split(Path("does_not_exist.csv"), encoding="utf-8", header=0)


def test_read_split_invalid_csv(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("text,labels\n\"unterminated,State\n", encoding="utf-8")
    with pytest.raises(DataLoaderError, match="parse error"):
        _read_split(p, encoding="utf-8", header=0)


def test_read_split_preserves_gujarati(tmp_path):
    p = tmp_path / "g.csv"
    _write_csv(p, SIMPLE_CSV)
    df = _read_split(p, encoding="utf-8", header=0)
    val = df.iloc[0]["text"]
    assert val == "ગુજરાતી સમાચાર"
    assert any(0x0A80 <= ord(c) <= 0x0AFF for c in val)


def test_read_split_train_stray_header(tmp_path):
    """The reader still supports a stray leading header row via header=1."""
    p = tmp_path / "t.csv"
    _write_csv(p, TRAIN_CSV_STRAY)
    df = _read_split(p, encoding="utf-8", header=1)
    assert list(df.columns) == ["text", "labels"]
    assert len(df) == 2
    assert df.iloc[0]["labels"] == "State"


def test_load_all_splits_with_fixtures(monkeypatch, tmp_path):
    _monkeypatch_paths(monkeypatch, tmp_path)
    splits = load_all_splits()
    assert isinstance(splits, DatasetSplits)
    assert list(splits.train.columns) == ["text", "labels"]
    assert list(splits.validation.columns) == ["text", "labels"]
    assert list(splits.test.columns) == ["text", "labels"]
    assert len(splits.train) == 2
    assert len(splits.validation) == 2
    assert len(splits.test) == 2


def test_individual_loaders(monkeypatch, tmp_path):
    _monkeypatch_paths(monkeypatch, tmp_path)
    assert len(load_train()) == 2
    assert len(load_validation()) == 2
    assert len(load_test()) == 2


def test_load_train_missing_file_error(monkeypatch, tmp_path):
    _monkeypatch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "TRAIN_PATH", tmp_path / "gone.csv")
    with pytest.raises(DataLoaderError, match="not found"):
        load_train()


def test_load_train_invalid_csv_error(monkeypatch, tmp_path):
    _monkeypatch_paths(monkeypatch, tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "Column1,Column2\ntext,labels\n\"unterminated,State\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "TRAIN_PATH", bad)
    with pytest.raises(DataLoaderError, match="parse error"):
        load_train()


def test_missing_value_reporting():
    df = pd.DataFrame(
        {"text": ["ગુજરાતી", None, ""], "labels": ["State", "Business", None]}
    )
    assert df["text"].isna().sum() == 1
    assert df["labels"].isna().sum() == 1
    assert int(df["text"].astype(str).str.strip().eq("").sum()) == 1


def test_category_counting():
    df = pd.DataFrame(
        {"text": ["a", "b", "c", "d"], "labels": ["State", "State", "Crime", "Auto"]}
    )
    vc = df["labels"].value_counts()
    assert vc["State"] == 2
    assert vc["Crime"] == 1
    assert vc["Auto"] == 1
    assert df["labels"].nunique() == 3


def test_validate_structure_passes():
    df = pd.DataFrame({"text": ["a"], "labels": ["State"]})
    validate_structure(df, "train")  # should not raise


def test_validate_structure_missing_column():
    df = pd.DataFrame({"text": ["a"]})
    with pytest.raises(DataLoaderError, match="missing expected column"):
        validate_structure(df, "train")


def test_configured_file_path_resolution():
    """Configured paths resolve to real files on disk."""
    for p in (config.TRAIN_PATH, config.VALID_PATH, config.TEST_PATH):
        assert p.exists(), f"configured path missing: {p}"
    assert resolve_dataset_dir() == config.TRAIN_PATH.parent


def test_real_dataset_dimensions():
    """Memory-conscious check: load real splits and confirm known row counts."""
    splits = load_all_splits()
    assert len(splits.train) == 79_416
    assert len(splits.validation) == 10_251
    assert len(splits.test) == 10_041
    for df in (splits.train, splits.validation, splits.test):
        assert list(df.columns) == ["text", "labels"]
