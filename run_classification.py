"""Entry-point script for Gujarati news classification (Phase 5).

Run:
    .venv\\Scripts\\python.exe run_classification.py

Trains and evaluates Logistic Regression, Linear SVM, Random Forest, and
Gradient Boosting on the existing CBOW / Skip-gram document vectors and TF-IDF
baselines. Selects the best configuration on the validation split by macro-F1,
then evaluates it once on the untouched test split.

Word2Vec is never retrained, document vectors are never regenerated, and the
original CSV files are never modified.
"""

from __future__ import annotations

import sys

from src.classification import run_classification

if __name__ == "__main__":
    # Ensure stdout can emit Gujarati/UTF-8 text on Windows consoles (cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    outcome = run_classification()

    df = outcome["dataframe"]
    print("Validation experiments (sorted by macro-F1):")
    print(
        df.sort_values("macro_f1", ascending=False)[
            ["index", "feature_type", "classifier", "macro_f1", "accuracy", "weighted_f1"]
        ].to_string(index=False)
    )

    best = outcome["best"].result
    test = outcome["test_metrics"]
    print(
        f"\nSelected: exp {best.index} - {best.feature_type} / {best.classifier}"
        f" (validation macro-F1={best.macro_f1:.4f})"
    )
    print(
        f"Test: accuracy={test['accuracy']:.4f}  macro-F1={test['macro_f1']:.4f}  "
        f"weighted-F1={test['weighted_f1']:.4f}"
    )
    print("\nArtifacts and reports:")
    for name, path in outcome["paths"].items():
        print(f"  {name:16s}: {path}")
