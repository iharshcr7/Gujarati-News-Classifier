"""Project configuration for Gujarati News Classification Using Word2Vec.

Paths are resolved relative to this file's location (the project root) so the
configuration stays portable across Windows machines. Do not hardcode
machine-specific absolute paths.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# NOTE: The files on disk are spelled "Gujurathi" (a spelling variant present in
# the repository). These are the actual filenames; do not "correct" them.
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATASET_DIR / "Gujarati_SHC_LDC_LPC_Train.csv"
VALID_PATH = DATASET_DIR / "Gujarati_SHC_LDC_LPC_Valid.csv"
TEST_PATH = DATASET_DIR / "Gujarati_SHC_LDC_LPC_Test.csv"

# Selected dataset format for the initial experiment: SHC (short headlines).
# IMPORTANT: the CSV files only expose two columns -- `text` and `labels`. There
# are no separate SHC/LDC/LPC fields despite the filenames. The `text` column is
# therefore the SHC text source.
DATA_FORMAT = "SHC"
TEXT_COLUMN = "text"
LABEL_COLUMN = "labels"

# Encoding notes:
#   - All three recovered CSV files are UTF-8-compatible (optionally with BOM)
#     and contain real Gujarati Unicode text; read all with utf-8-sig.
#   - No split has a stray header row: every file starts with "text,labels".
TRAIN_ENCODING = "utf-8-sig"
TRAIN_ENCODING_ERRORS = None
VALID_ENCODING = "utf-8-sig"
TEST_ENCODING = "utf-8-sig"

TRAIN_HEADER = 0   # 0-based row to use as header
VALID_HEADER = 0
TEST_HEADER = 0

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Word2Vec configuration (Phase 3)
# ---------------------------------------------------------------------------
# Shared hyperparameters for CBOW (sg=0) and Skip-gram (sg=1). Kept here so the
# experiment is reproducible and the parameters used are recorded in the report.
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = (os.cpu_count() or 1)
W2V_SEED = 42
W2V_EPOCHS = 10
W2V_SG_CBOW = 0
W2V_SG_SKIPGRAM = 1

# Native Gensim model artifacts (loadable later for inference / doc vectors).
W2V_MODEL_CBOW_PATH = MODELS_DIR / "word2vec_cbow.model"
W2V_MODEL_SKIPGRAM_PATH = MODELS_DIR / "word2vec_skipgram.model"

# Report written after training.
W2V_REPORT_FILENAME = "word2vec_training_report.txt"

# ---------------------------------------------------------------------------
# Document-vector configuration (Phase 4)
# ---------------------------------------------------------------------------
# Fixed-length headline vectors are produced by mean-pooling the embeddings of
# in-vocabulary tokens from the already-trained models above. Vector dimension
# must match W2V_VECTOR_SIZE; it is validated at runtime rather than assumed.
DOCVEC_REPORT_FILENAME = "document_vectors_report.txt"

# ---------------------------------------------------------------------------
# Classifier configuration (Phase 5)
# ---------------------------------------------------------------------------
# Trained classifiers and the fitted vectorizers they depend on are stored
# under models/classifiers/ so inference uses exactly the same transformation.
CLASSIFIERS_DIR = MODELS_DIR / "classifiers"

# Experiment tracking outputs.
CLASSIFICATION_RESULTS_CSV = "classification_experiments.csv"
CLASSIFICATION_RESULTS_JSON = "classification_experiments.json"
CLASSIFICATION_REPORT_FILENAME = "classification_report.txt"
CLASSIFICATION_TEST_REPORT_FILENAME = "classification_test_report.txt"
CLASSIFICATION_VALIDATION_TABLE_FILENAME = "classification_validation_table.md"
CLASSIFICATION_COMPARISON_PLOT_FILENAME = "classification_validation_macro_f1.png"
CLASSIFICATION_CONFUSION_PLOT_FILENAME = "classification_test_confusion_matrix.png"

# TF-IDF (word-level) baseline. Defaults chosen after inspecting the data:
# ~78.8k training headlines, ~153 tokens/headline median 105, so a 50k capped
# 1-2 gram vocabulary keeps the matrix sparse (measured ~0.23% density).
TFIDF_WORD_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 50000,
    "sublinear_tf": True,
    "lowercase": False,
}

# TF-IDF (character) baseline. Character n-grams on long headlines are far
# larger, so the vocabulary is capped and n-grams kept moderate.
TFIDF_CHAR_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 5,
    "max_features": 100000,
    "sublinear_tf": True,
    "lowercase": False,
}

# Feature configurations evaluated by the pipeline. ``enabled`` controls
# whether a configuration is included in a default run; disabled entries carry
# a ``disabled_reason`` that is recorded in the report (never skipped until
# they are explicitly requested, so nothing is silently dropped).
FEATURE_CONFIGURATIONS = (
    {
        "key": "cbow",
        "kind": "docvec",
        "model_name": "cbow",
        "label": "CBOW doc vectors (100d)",
        "enabled": True,
    },
    {
        "key": "skipgram",
        "kind": "docvec",
        "model_name": "skipgram",
        "label": "Skip-gram doc vectors (100d)",
        "enabled": True,
    },
    {
        "key": "tfidf_word",
        "kind": "tfidf",
        "analyzer": "word",
        "label": "TF-IDF word n-grams",
        "enabled": True,
    },
    {
        "key": "tfidf_char",
        "kind": "tfidf",
        "analyzer": "char",
        "label": "TF-IDF char n-grams",
        "enabled": False,
        "disabled_reason": (
            "Character n-grams were measured on the full training split: "
            "fitting + transforming took ~459s and produced a ~74.5M-nonzero "
            "(78806 x 100000) CSR matrix taking >1.2 GB, on a machine with "
            "7.2 GB total RAM. Training four classifiers on top of that is not "
            "practical here, so the word-level baseline is used instead. Set "
            "enabled=True (and expect a long run) to attempt it."
        ),
    },
)

# Tree ensembles cannot consume the large sparse TF-IDF matrices directly
# without excessive memory; they are trained on an SVD-reduced dense view.
CLASSIFIER_SVD_COMPONENTS = 256

# Classifier defaults. Kept small and documented; no exhaustive search.
CLASSIFIER_DEFAULTS = {
    "logistic_regression": {"max_iter": 1000, "solver": "lbfgs"},
    "linear_svm": {"C": 1.0},
    "random_forest": {"n_estimators": 200, "n_jobs": -1},
    "gradient_boosting": {"max_iter": 200, "learning_rate": 0.1},
}

# ---------------------------------------------------------------------------
# Error analysis & targeted experiments (Phase 7)
# ---------------------------------------------------------------------------
# New candidate classifiers from Phase 7 tuning are stored separately from the
# frozen Phase 5 selected model so experimentation can never overwrite it.
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
# Dedicated report locations for the Phase 7 validation error analysis and the
# unified validation comparison. The historical Phase 5 reports are left intact.
ERROR_ANALYSIS_DIR = REPORTS_DIR / "error_analysis"
EXPERIMENTS_REPORT_DIR = REPORTS_DIR / "experiments"

# Minimum absolute validation macro-F1 improvement required before replacing the
# Phase 5 selected model (conservative default; see Phase 7 plan).
REPLACEMENT_THRESHOLD = 0.005

# ---------------------------------------------------------------------------
# Category labels (confirmed from dataset inspection)
# ---------------------------------------------------------------------------
CATEGORY_LABELS = [
    "State",
    "International",
    "Entertainment",
    "Business",
    "Nation",
    "Technology",
    "Sports",
    "Astro",
    "Crime",
    "Auto",
]

NUM_CLASSES = len(CATEGORY_LABELS)


def validate_paths() -> None:
    """Raise FileNotFoundError if any configured dataset path is missing."""
    missing = [str(p) for p in (TRAIN_PATH, VALID_PATH, TEST_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset file(s) not found: " + ", ".join(missing)
        )


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("TRAIN_PATH :", TRAIN_PATH, "exists=", TRAIN_PATH.exists())
    print("VALID_PATH :", VALID_PATH, "exists=", VALID_PATH.exists())
    print("TEST_PATH  :", TEST_PATH, "exists=", TEST_PATH.exists())
    print("REPORTS_DIR:", REPORTS_DIR)
