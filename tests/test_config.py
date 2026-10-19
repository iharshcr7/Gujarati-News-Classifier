"""Unit tests for config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import config


def test_project_root_is_parent_of_config():
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_dataset_dir_exists():
    assert config.DATASET_DIR.exists()
    assert config.DATASET_DIR.is_dir()


def test_all_dataset_paths_exist():
    for p in (config.TRAIN_PATH, config.VALID_PATH, config.TEST_PATH):
        assert p.exists(), f"missing: {p}"
        assert p.is_file()


def test_format_and_columns():
    assert config.DATA_FORMAT == "SHC"
    assert config.TEXT_COLUMN == "text"
    assert config.LABEL_COLUMN == "labels"


def test_category_labels():
    assert len(config.CATEGORY_LABELS) == config.NUM_CLASSES
    assert config.NUM_CLASSES == 10
    for label in ("State", "International", "Crime", "Auto"):
        assert label in config.CATEGORY_LABELS


def test_reports_dir_creatable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    assert config.REPORTS_DIR.exists()


def test_validate_paths_ok():
    config.validate_paths()  # should not raise


def test_validate_paths_missing(monkeypatch, tmp_path):
    missing = tmp_path / "gone.csv"
    monkeypatch.setattr(config, "TRAIN_PATH", missing)
    with pytest.raises(FileNotFoundError, match="not found"):
        config.validate_paths()
