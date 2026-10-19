"""Dataset inspection for the Gujarati news classification project.

Produces a human-readable inspection report covering dimensions, schema,
samples, missing values, duplicates, empty text, category distributions,
Gujarati Unicode integrity, and cross-split duplicate/conflicting-label
checks. The original CSV files are never modified.

The report is written to ``reports/`` as UTF-8 so Gujarati text is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import (
    DATA_FORMAT,
    LABEL_COLUMN,
    REPORTS_DIR,
    TEXT_COLUMN,
)
from .data_loader import (
    DatasetSplits,
    load_all_splits,
)

GUJARATI_LO = 0x0A80
GUJARATI_HI = 0x0AFF


def has_gujarati(text: str) -> bool:
    """Return True if ``text`` contains at least one Gujarati Unicode char."""
    return any(GUJARATI_LO <= ord(ch) <= GUJARATI_HI for ch in str(text))


def _sample_block(df: pd.DataFrame, name: str, n: int = 3) -> List[str]:
    lines: List[str] = []
    lines.append(f"--- {name} sample records ---")
    sample = df.head(n)
    n_show = min(n, len(sample))
    lines.append(f"  (showing {n_show} of {len(df)} rows)")
    for idx, row in sample.iterrows():
        text = str(row[TEXT_COLUMN])
        label = str(row[LABEL_COLUMN])
        lines.append(f"  row {idx}: label={label!r} | text={text!r}")
    return lines


def _missing_block(df: pd.DataFrame, name: str) -> List[str]:
    lines: List[str] = []
    lines.append(f"--- {name} missing values ---")
    for col in df.columns:
        lines.append(f"  {col}: {int(df[col].isna().sum())} missing")
    return lines


def _category_block(df: pd.DataFrame, name: str) -> List[str]:
    lines: List[str] = []
    lines.append(f"--- {name} category distribution ('{LABEL_COLUMN}') ---")
    vc = df[LABEL_COLUMN].value_counts(dropna=False)
    for label, count in vc.items():
        lines.append(f"  {label!r}: {int(count)}")
    lines.append(f"  distinct labels: {int(df[LABEL_COLUMN].nunique(dropna=False))}")
    return lines


def _unicode_block(df: pd.DataFrame, name: str) -> List[str]:
    lines: List[str] = []
    lines.append(f"--- {name} Gujarati Unicode integrity ---")
    if df.empty:
        lines.append("  non-empty text rows: 0")
        lines.append("  rows containing Gujarati (U+0A80..U+0AFF): 0")
        lines.append("  Gujarati coverage: N/A (no rows)")
        return lines
    non_empty = df[TEXT_COLUMN].astype(str).str.strip().ne("")
    gujarati_mask = df[TEXT_COLUMN].map(has_gujarati)
    total_non_empty = int(non_empty.sum())
    gujarati_count = int(gujarati_mask.sum())
    lines.append(f"  non-empty text rows: {total_non_empty}")
    lines.append(f"  rows containing Gujarati (U+0A80..U+0AFF): {gujarati_count}")
    if total_non_empty:
        lines.append(
            f"  Gujarati coverage: {gujarati_count / total_non_empty * 100:.2f}%"
        )
    else:
        lines.append("  Gujarati coverage: N/A (no non-empty rows)")
    return lines


def inspect_splits(splits: DatasetSplits) -> str:
    """Build the full inspection report string from loaded splits."""
    train, valid, test = splits.train, splits.validation, splits.test

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GUJARATI NEWS DATASET INSPECTION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. DATASET DIMENSIONS AND ROW COUNTS")
    lines.append("-" * 72)
    for name, df in [("train", train), ("validation", valid), ("test", test)]:
        lines.append(
            f"  {name:12s}: rows={len(df):>7}  cols={df.shape[1]}  shape={df.shape}"
        )
    lines.append("")

    lines.append("2. COLUMN NAMES AND DATA TYPES")
    lines.append("-" * 72)
    for name, df in [("train", train), ("validation", valid), ("test", test)]:
        lines.append(f"  {name}:")
        for col in df.columns:
            lines.append(f"    - {col}: {df[col].dtype}")
    lines.append("")

    lines.append("3. SAMPLE RECORDS (first 3 per split)")
    lines.append("-" * 72)
    lines.extend(_sample_block(train, "train"))
    lines.append("")
    lines.extend(_sample_block(valid, "validation"))
    lines.append("")
    lines.extend(_sample_block(test, "test"))
    lines.append("")

    lines.append("4. MISSING-VALUE COUNTS")
    lines.append("-" * 72)
    lines.extend(_missing_block(train, "train"))
    lines.append("")
    lines.extend(_missing_block(valid, "validation"))
    lines.append("")
    lines.extend(_missing_block(test, "test"))
    lines.append("")

    lines.append("5. DUPLICATE-ROW COUNTS (within each split)")
    lines.append("-" * 72)
    for name, df in [("train", train), ("validation", valid), ("test", test)]:
        lines.append(f"  {name}: {int(df.duplicated().sum())} duplicate rows")
    lines.append("")

    lines.append("6. EMPTY / WHITESPACE-ONLY TEXT COUNTS")
    lines.append("-" * 72)
    for name, df in [("train", train), ("validation", valid), ("test", test)]:
        empty_count = int(df[TEXT_COLUMN].astype(str).str.strip().eq("").sum())
        lines.append(f"  {name}: {empty_count} empty/whitespace-only text rows")
    lines.append("")

    lines.append("7. CATEGORY DISTRIBUTION FOR LABEL COLUMN")
    lines.append("-" * 72)
    lines.extend(_category_block(train, "train"))
    lines.append("")
    lines.extend(_category_block(valid, "validation"))
    lines.append("")
    lines.extend(_category_block(test, "test"))
    lines.append("")

    lines.append("8. GUJARATI UNICODE INTEGRITY CHECK")
    lines.append("-" * 72)
    lines.extend(_unicode_block(train, "train"))
    lines.append("")
    lines.extend(_unicode_block(valid, "validation"))
    lines.append("")
    lines.extend(_unicode_block(test, "test"))
    lines.append("")

    lines.append("9. CROSS-SPLIT DUPLICATE AND CONFLICTING-LABEL CHECKS")
    lines.append("-" * 72)
    train_texts = set(train[TEXT_COLUMN].astype(str))
    valid_texts = set(valid[TEXT_COLUMN].astype(str))
    test_texts = set(test[TEXT_COLUMN].astype(str))

    train_valid_shared = train_texts & valid_texts
    train_test_shared = train_texts & test_texts
    valid_test_shared = valid_texts & test_texts

    lines.append(f"  train ∩ validation shared texts: {len(train_valid_shared)}")
    lines.append(f"  train ∩ test shared texts:       {len(train_test_shared)}")
    lines.append(f"  validation ∩ test shared texts:  {len(valid_test_shared)}")

    train_labels = train.set_index(TEXT_COLUMN)[LABEL_COLUMN].to_dict()
    valid_labels = valid.set_index(TEXT_COLUMN)[LABEL_COLUMN].to_dict()
    test_labels = test.set_index(TEXT_COLUMN)[LABEL_COLUMN].to_dict()

    def conflicts(shared, left_map, right_map) -> int:
        return sum(
            1 for txt in shared if left_map.get(txt) != right_map.get(txt)
        )

    lines.append(
        f"  train-validation conflicting labels: {conflicts(train_valid_shared, train_labels, valid_labels)}"
    )
    lines.append(
        f"  train-test conflicting labels:       {conflicts(train_test_shared, train_labels, test_labels)}"
    )
    lines.append(
        f"  validation-test conflicting labels:  {conflicts(valid_test_shared, valid_labels, test_labels)}"
    )
    lines.append("")

    lines.append("10. SHC COLUMN SUMMARY")
    lines.append("-" * 72)
    lines.append(f"  Selected dataset format: {DATA_FORMAT}")
    lines.append(
        f"  Text column used for SHC: {TEXT_COLUMN!r}"
    )
    lines.append(
        f"  Label column used for SHC: {LABEL_COLUMN!r}"
    )
    lines.append(
        "  NOTE: The CSV files expose only two columns (`text` and `labels`); "
        "there are no separate SHC/LDC/LPC fields despite the filenames. "
        "The `text` column is therefore used as the SHC text source."
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)

    return "\n".join(lines)


def write_report(report: str, filename: str = "dataset_inspection_report.txt") -> Path:
    """Write the report to ``reports/`` as UTF-8 and return the report path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / filename
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_inspection() -> Tuple[str, Path]:
    """Load splits, build the report, write it, and return (report, path)."""
    splits = load_all_splits()
    report = inspect_splits(splits)
    report_path = write_report(report)
    return report, report_path


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    report, path = run_inspection()
    print(f"Report written to: {path}")
