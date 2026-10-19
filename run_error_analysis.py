"""Entry point: Phase 7 validation error analysis (Steps A, B and C).

Run:
    .venv\\Scripts\\python.exe run_error_analysis.py

Loads the frozen Phase 5 selected model read-only, evaluates it on the existing
validation features, writes the detailed error-analysis report and CSVs, the
error-cause investigation, and the baseline reproduction report. Nothing is
retrained; the test split and every existing artifact are left untouched.
"""

from __future__ import annotations

import sys

import config
from src.error_analysis import (
    build_baseline_reproduction_report,
    run_validation_error_analysis,
)
from src.text_features import EXPECTED_ROW_COUNTS


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    outcome = run_validation_error_analysis(
        out_dir=config.ERROR_ANALYSIS_DIR,
        expected_counts=EXPECTED_ROW_COUNTS,
    )

    # Step C: compare the reproduced metrics with the recorded Phase 5 metrics.
    artifact = outcome["artifact"]
    recorded = dict(artifact.get("metadata", {}).get("validation_metrics", {}))
    reproduced = outcome["metrics"]
    if recorded:
        report = build_baseline_reproduction_report(
            reproduced=reproduced,
            recorded=recorded,
            model_description=outcome["model_description"],
            artifact_path=outcome["artifact_path"],
        )
        baseline_path = config.ERROR_ANALYSIS_DIR / "baseline_reproduction.md"
        baseline_path.write_text(report, encoding="utf-8")
        print(f"baseline reproduction : {baseline_path}")

    metrics = outcome["metrics"]
    errors = int((~outcome["predictions"]["correct"].astype(bool)).sum())
    print(f"\nValidation examples : {metrics['support']}")
    print(f"Misclassified       : {errors} ({errors / metrics['support']:.4f})")
    print(f"Accuracy            : {metrics['accuracy']:.4f}")
    print(f"Macro-F1            : {metrics['macro_f1']:.4f}")
    print("\nArtifacts:")
    for name, path in outcome["paths"].items():
        print(f"  {name:16s}: {path}")
    print(f"  {'investigation':16s}: {outcome['investigation_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
