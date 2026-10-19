"""Entry-point script for Gujarati SHC preprocessing (Phase 2).

Run:
    python preprocess_dataset.py

The preprocessing report is written to reports/preprocessing_report.txt
(UTF-8, Gujarati-safe) and cleaned splits are written to data/processed/
without touching the original data/raw/ files.
"""

from __future__ import annotations

import sys

from src.preprocessing import run_preprocessing_pipeline

if __name__ == "__main__":
    # Ensure stdout can emit Gujarati/UTF-8 text on Windows consoles (cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    report, report_path, _ = run_preprocessing_pipeline(save_processed=True)
    print(report)
    print(f"\nReport saved to: {report_path}")
