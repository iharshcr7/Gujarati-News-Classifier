"""Entry-point script for dataset inspection.

Run:
    .venv\Scripts\python.exe inspect_dataset.py

The inspection report is written to reports/dataset_inspection_report.txt
(UTF-8, Gujarati-safe).
"""

from __future__ import annotations

from src.dataset_inspection import run_inspection

if __name__ == "__main__":
    report, report_path = run_inspection()
    print(report)
    print(f"\nReport saved to: {report_path}")
