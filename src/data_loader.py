"""Reusable data-loading utilities for the Gujarati news dataset.

Loads the train / validation / test CSV splits from the configured dataset
directory, validates their structure, and returns them as pandas DataFrames
without modifying the original source files.

Design goals:
- `pathlib.Path` throughout; configurable paths via `config.py` (read at call
  time so paths can be overridden in tests).
- Clear, actionable errors for missing files and invalid CSVs.
- Original data preserved during loading (no in-place cleaning/dedup).
- Schema validation once the column contract is established.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

import config


class DataLoaderError(RuntimeError):
    """Raised when dataset loading or validation fails."""


@dataclass(frozen=True)
class DatasetSplits:
    """Container for the three loaded splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def resolve_dataset_dir() -> Path:
    """Return the configured dataset directory, raising if it is missing."""
    dataset_dir = config.TRAIN_PATH.parent
    if not dataset_dir.exists():
        raise DataLoaderError(f"Dataset directory not found: {dataset_dir}")
    return dataset_dir


def _read_split(
    path: Path,
    *,
    encoding: str,
    header: int,
    encoding_errors: Optional[str] = None,
) -> pd.DataFrame:
    """Read a single CSV split into a DataFrame.

    The original file is not modified. Missing-file and parse errors are
    converted into :class:`DataLoaderError` with a clear message.
    """
    if not path.exists():
        raise DataLoaderError(f"Dataset file not found: {path}")

    read_kwargs = dict(
        encoding=encoding,
        header=header,
        dtype={config.TEXT_COLUMN: str, config.LABEL_COLUMN: str},
        keep_default_na=True,
    )
    if encoding_errors is not None:
        read_kwargs["encoding_errors"] = encoding_errors

    try:
        df = pd.read_csv(path, **read_kwargs)
    except FileNotFoundError as exc:
        raise DataLoaderError(f"Dataset file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataLoaderError(f"Dataset file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoaderError(
            f"Invalid CSV (parse error) in {path}: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataLoaderError(
            f"Encoding error reading {path} with encoding={encoding}: {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise DataLoaderError(f"Unexpected error reading {path}: {exc}") from exc

    return df


def load_train() -> pd.DataFrame:
    """Load the training split."""
    return _read_split(
        config.TRAIN_PATH,
        encoding=config.TRAIN_ENCODING,
        header=config.TRAIN_HEADER,
        encoding_errors=config.TRAIN_ENCODING_ERRORS,
    )


def load_validation() -> pd.DataFrame:
    """Load the validation split."""
    return _read_split(
        config.VALID_PATH,
        encoding=config.VALID_ENCODING,
        header=config.VALID_HEADER,
    )


def load_test() -> pd.DataFrame:
    """Load the test split."""
    return _read_split(
        config.TEST_PATH,
        encoding=config.TEST_ENCODING,
        header=config.TEST_HEADER,
    )


def load_all_splits() -> DatasetSplits:
    """Load train, validation, and test splits together."""
    return DatasetSplits(
        train=load_train(),
        validation=load_validation(),
        test=load_test(),
    )


def validate_structure(df: pd.DataFrame, source: str) -> None:
    """Validate that a loaded split has the expected columns and non-empty data.

    Does not modify ``df``. Raises :class:`DataLoaderError` on contract breach.
    """
    expected = {config.TEXT_COLUMN, config.LABEL_COLUMN}
    missing = expected - set(df.columns)
    if missing:
        raise DataLoaderError(
            f"{source}: missing expected column(s) {sorted(missing)}; "
            f"found {list(df.columns)}"
        )
    if df.empty:
        raise DataLoaderError(f"{source}: loaded DataFrame is empty")
    if df[config.TEXT_COLUMN].isna().all():
        raise DataLoaderError(f"{source}: '{config.TEXT_COLUMN}' is entirely missing")
    if df[config.LABEL_COLUMN].isna().all():
        raise DataLoaderError(f"{source}: '{config.LABEL_COLUMN}' is entirely missing")


def validate_all(splits: DatasetSplits) -> None:
    """Run structure validation on every split."""
    validate_structure(splits.train, "train")
    validate_structure(splits.validation, "validation")
    validate_structure(splits.test, "test")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    splits = load_all_splits()
    validate_all(splits)
    for name, df in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]:
        print(name, df.shape, list(df.columns))
