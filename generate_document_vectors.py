"""Entry-point script for Gujarati headline document-vector generation (Phase 4).

Run:
    .venv\\Scripts\\python.exe generate_document_vectors.py

Generates fixed-length (100-d) document vectors for every Gujarati SHC headline
by mean-pooling the embeddings of its in-vocabulary tokens, using the existing
Phase 3 CBOW and Skip-gram models. Vectors and metadata are written to
data/processed/ and an analysis report to reports/document_vectors_report.txt.

The Word2Vec models are loaded read-only and never retrained. The original
data/raw/ CSV files are never modified, and validation/test rows are never used
to train or update the embeddings.
"""

from __future__ import annotations

import sys

from src.document_vectors import generate_document_vectors

if __name__ == "__main__":
    # Ensure stdout can emit Gujarati/UTF-8 text on Windows consoles (cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    report, report_path, results = generate_document_vectors()
    print(report)
    print(f"\nReport saved to: {report_path}")
    for model_name, splits in results.items():
        for split, dvs in splits.items():
            print(
                f"{model_name:9s} {split:11s}: {dvs.vectors.shape} "
                f"fallback={dvs.fallback_count}"
            )
