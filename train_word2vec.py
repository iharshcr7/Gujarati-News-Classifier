"""Entry-point script for Word2Vec CBOW and Skip-gram training (Phase 3).

Run:
    .venv\Scripts\python.exe train_word2vec.py

Trains CBOW (sg=0) and Skip-gram (sg=1) Word2Vec models on the SHC training
split only (no data leakage), saves the Gensim artifacts to models/, and writes
a training analysis report to reports/word2vec_training_report.txt.

The original data/raw/ CSV files are never modified. Validation and test
headlines are never used to train the embeddings.
"""

from __future__ import annotations

import logging
import sys

from src.word2vec_trainer import run_training_pipeline

if __name__ == "__main__":
    # Ensure stdout can emit Gujarati/UTF-8 text on Windows consoles (cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Intentional retraining from the recovered/cleaned dataset. The previous
    # artifacts (trained on corrupted raw train text) are preserved under
    # models/backup/ before this overwrite.
    cbow, skipgram, report, report_path = run_training_pipeline(overwrite=True)
    print(report)
    print(f"\nReport saved to: {report_path}")
    print(f"CBOW model   : {cbow.model_path} ({cbow.model_file_size_bytes} bytes)")
    print(f"Skip-gram model: {skipgram.model_path} ({skipgram.model_file_size_bytes} bytes)")
