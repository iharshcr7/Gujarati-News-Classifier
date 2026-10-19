"""Unit tests for src/dataset_inspection.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from src.dataset_inspection import (
    GUJARATI_HI,
    GUJARATI_LO,
    has_gujarati,
    inspect_splits,
    run_inspection,
    write_report,
)
from src.data_loader import DatasetSplits


def _make_split(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["text", "labels"])


def test_has_gujarati_true():
    assert has_gujarati("ગુજરાતી") is True
    assert has_gujarati("abc ગુજરાતી xyz") is True


def test_has_gujarati_false():
    assert has_gujarati("hello world") is False
    assert has_gujarati("") is False


def test_gujarati_range_constants():
    assert GUJARATI_LO == 0x0A80
    assert GUJARATI_HI == 0x0AFF


def test_inspect_splits_contains_sections(monkeypatch, tmp_path):
    train = _make_split([["ગુજરાતી સમાચાર", "State"], ["hello", "Business"]])
    valid = _make_split([["પરીક્ષા", "State"], ["good", "International"]])
    test = _make_split([["આભાર", "Crime"], ["thanks", "Auto"]])
    splits = DatasetSplits(train=train, validation=valid, test=test)

    report = inspect_splits(splits)
    for section in [
        "1. DATASET DIMENSIONS AND ROW COUNTS",
        "2. COLUMN NAMES AND DATA TYPES",
        "3. SAMPLE RECORDS (first 3 per split)",
        "4. MISSING-VALUE COUNTS",
        "5. DUPLICATE-ROW COUNTS",
        "6. EMPTY / WHITESPACE-ONLY TEXT COUNTS",
        "7. CATEGORY DISTRIBUTION FOR LABEL COLUMN",
        "8. GUJARATI UNICODE INTEGRITY CHECK",
        "9. CROSS-SPLIT DUPLICATE AND CONFLICTING-LABEL CHECKS",
        "10. SHC COLUMN SUMMARY",
    ]:
        assert section in report, f"missing section: {section}"


def test_inspect_splits_reports_dimensions():
    train = _make_split([["a", "State"], ["b", "Business"]])
    valid = _make_split([["c", "State"]])
    test = _make_split([["d", "Crime"]])
    report = inspect_splits(DatasetSplits(train=train, validation=valid, test=test))
    assert "shape=(2, 2)" in report
    assert "shape=(1, 2)" in report


def test_inspect_splits_reports_missing_and_duplicates():
    train = _make_split([["a", "State"], [None, "Business"], ["a", "State"]])
    report = inspect_splits(DatasetSplits(train=train, validation=train[:0], test=train[:0]))
    assert "missing" in report.lower()
    assert "duplicate" in report.lower()


def test_write_report_creates_file(tmp_path, monkeypatch):
    report = "hello Gujarat ગુજરાતી"
    # point REPORTS_DIR at tmp
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    path = write_report(report, filename="test_report.txt")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == report


def test_run_inspection_writes_real_report():
    report, path = run_inspection()
    assert isinstance(report, str)
    assert path.exists()
    assert "DATASET DIMENSIONS" in report
    assert "SHC COLUMN SUMMARY" in report
