"""Feature construction for Gujarati news classification (Phase 5).

Two kinds of features are supported, both reusing existing Phase 2/4 artifacts:

- **Document vectors** (dense, 100-d): loaded read-only from the Phase 4
  ``.npz`` archives (``X``/``labels``/``row_ids``). Never regenerated here.
- **TF-IDF** (sparse): fitted on the *training* text only using the shared
  Phase 2 preprocessing, then used to transform validation/test text. The
  original raw CSVs are never modified.

Data-leakage prevention
-----------------------
The cleaned train/validation/test splits are rebuilt with exactly the same
``clean_splits`` + ``preprocess`` pipeline used in Phase 4, so row membership
and row order match the document-vector artifacts. The rebuilt labels are
checked positionally against the doc-vector labels before any comparison.

Design goals
------------
- Never fit a vectorizer on validation/test text.
- Preserve row order; expose the same ``row_id`` (positional index) contract.
- Validate feature shape, finiteness, and label alignment on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.data_loader import load_all_splits, validate_all
from src.document_vectors import load_document_vectors
from src.preprocessing import clean_splits, preprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCVEC_MODEL_NAMES: Tuple[str, ...] = ("cbow", "skipgram")
SPLITS: Tuple[str, ...] = ("train", "validation", "test")

# Row counts produced by the completed Phase 2/3/4 pipeline. Used to confirm
# the artifacts have not drifted before training classifiers.
EXPECTED_ROW_COUNTS: Dict[str, int] = {
    "train": 78806,
    "validation": 10175,
    "test": 9972,
}

# TF-IDF matrix dtype. float64 keeps sklearn's SVD/logistic solvers happy and
# the matrices are sparse, so memory stays proportional to non-zeros.
TFIDF_DTYPE = np.float64


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    """A single split's feature matrix with aligned labels and row ids."""

    X: object                      # ndarray (dense) or scipy.sparse matrix
    y: np.ndarray                  # string labels
    row_ids: np.ndarray            # positional index within the cleaned split
    feature_type: str
    num_features: int
    is_sparse: bool
    class_names: Tuple[str, ...]
    vectorizer: Optional[TfidfVectorizer] = None
    svd: Optional[TruncatedSVD] = None


@dataclass
class FeatureBundle:
    """Train/validation/test matrices for one feature configuration."""

    key: str                       # e.g. "cbow", "tfidf_word"
    label: str                     # human-readable configuration name
    feature_type: str              # e.g. "docvec:cbow", "tfidf_word"
    feature_family: str            # "docvec" or "tfidf"
    is_sparse: bool
    num_features: int
    class_names: Tuple[str, ...]
    X: Dict[str, object] = field(default_factory=dict)
    y: Dict[str, np.ndarray] = field(default_factory=dict)
    row_ids: Dict[str, np.ndarray] = field(default_factory=dict)
    vectorizer: Optional[TfidfVectorizer] = None
    model_name: Optional[str] = None
    warnings: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Split <-> artifact filename mapping
# ---------------------------------------------------------------------------

def _split_stem(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    return {
        "train": config.TRAIN_PATH.stem,
        "validation": config.VALID_PATH.stem,
        "test": config.TEST_PATH.stem,
    }[split]


def docvec_path(
    model_name: str,
    split: str,
    processed_dir: Path = config.PROCESSED_DIR,
) -> Path:
    """Return the ``.npz`` path for a (model, split) document-vector file."""
    if model_name not in DOCVEC_MODEL_NAMES:
        raise ValueError(
            f"Unknown document-vector model {model_name!r}; "
            f"expected one of {DOCVEC_MODEL_NAMES}"
        )
    return Path(processed_dir) / f"{_split_stem(split)}.{model_name}.docvecs.npz"


# ---------------------------------------------------------------------------
# Document-vector loading
# ---------------------------------------------------------------------------

def load_docvec_split(
    model_name: str,
    split: str,
    *,
    processed_dir: Path = config.PROCESSED_DIR,
    class_names: Optional[Sequence[str]] = None,
) -> FeatureMatrix:
    """Load and validate one document-vector split.

    Raises ``FileNotFoundError`` if the archive is missing and ``ValueError``
    if it is empty, non-2-D, non-finite, misaligned, or has non-sequential
    ``row_ids``.
    """
    path = docvec_path(model_name, split, processed_dir)
    if not path.exists():
        raise FileNotFoundError(f"Document-vector artifact not found: {path}")

    data = load_document_vectors(path)
    for required in ("X", "labels", "row_ids"):
        if required not in data:
            raise ValueError(f"{path.name}: missing array {required!r}")

    X = np.asarray(data["X"])
    y = np.asarray(data["labels"]).astype(str)
    row_ids = np.asarray(data["row_ids"])

    if X.ndim != 2:
        raise ValueError(f"{path.name}: X must be 2-D, got shape {X.shape}")
    n = X.shape[0]
    if n == 0:
        raise ValueError(f"{path.name}: X is empty")
    if len(y) != n or len(row_ids) != n:
        raise ValueError(
            f"{path.name}: length mismatch X={n} labels={len(y)} "
            f"row_ids={len(row_ids)}"
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{path.name}: X contains non-finite values")
    if not np.array_equal(row_ids, np.arange(n, dtype=np.int64)):
        raise ValueError(
            f"{path.name}: row_ids are not the sequential positional index"
        )

    classes = (
        tuple(str(c) for c in class_names)
        if class_names is not None
        else tuple(sorted(set(y.tolist())))
    )
    return FeatureMatrix(
        X=X,
        y=y,
        row_ids=row_ids,
        feature_type=f"docvec:{model_name}",
        num_features=int(X.shape[1]),
        is_sparse=False,
        class_names=classes,
    )


def build_docvec_bundle(
    config_entry: Mapping[str, object],
    *,
    processed_dir: Path = config.PROCESSED_DIR,
    class_names: Optional[Sequence[str]] = None,
) -> FeatureBundle:
    """Load all three splits for a document-vector configuration."""
    model_name = str(config_entry["model_name"])
    bundle = FeatureBundle(
        key=str(config_entry["key"]),
        label=str(config_entry.get("label", model_name)),
        feature_type=f"docvec:{model_name}",
        feature_family="docvec",
        is_sparse=False,
        num_features=0,
        class_names=tuple(class_names) if class_names else (),
        model_name=model_name,
    )
    for split in SPLITS:
        fm = load_docvec_split(
            model_name, split, processed_dir=processed_dir, class_names=class_names
        )
        bundle.X[split] = fm.X
        bundle.y[split] = fm.y
        bundle.row_ids[split] = fm.row_ids
        bundle.num_features = fm.num_features
    bundle.class_names = (
        tuple(class_names) if class_names else fm.class_names
    )
    return bundle


# ---------------------------------------------------------------------------
# Cleaned text (same pipeline used for the document vectors)
# ---------------------------------------------------------------------------

def load_cleaned_texts(
    *,
    text_column: str = config.TEXT_COLUMN,
    expected_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Rebuild the Phase 2 cleaned splits and return ``(cleaned_texts, labels)``.

    Uses exactly ``clean_splits`` + ``preprocess`` (stopwords disabled) as the
    Phase 3/4 pipeline. If ``expected_counts`` is supplied, a mismatch raises
    ``ValueError`` so a drifted split can never silently enter a comparison.
    """
    config.validate_paths()
    raw = load_all_splits()
    validate_all(raw)
    cleaned, _ = clean_splits(raw)

    frames = {
        "train": cleaned.train,
        "validation": cleaned.validation,
        "test": cleaned.test,
    }
    if expected_counts is not None:
        for split, df in frames.items():
            expected = expected_counts.get(split)
            if expected is not None and len(df) != expected:
                raise ValueError(
                    f"{split}: expected {expected} rows from the completed "
                    f"pipeline but found {len(df)}"
                )

    out: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for split, df in frames.items():
        if text_column not in df.columns:
            raise ValueError(
                f"{split}: text column {text_column!r} not found in "
                f"{list(df.columns)}"
            )
        texts = [preprocess(value).cleaned for value in df[text_column].astype(str)]
        labels = df[config.LABEL_COLUMN].astype(str).to_numpy()
        out[split] = (texts, labels)
    return out


def labels_match_docvec(labels: Sequence[str], feature: FeatureMatrix) -> bool:
    """True when ``labels`` align positionally with a doc-vector split."""
    arr = np.asarray(list(labels), dtype=str)
    return arr.shape == feature.y.shape and bool(np.array_equal(arr, feature.y))


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TfidfConfig:
    """Configuration for the TF-IDF baselines."""

    analyzer: str = "word"
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_features: int = 50000
    sublinear_tf: bool = True
    lowercase: bool = False

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> "TfidfConfig":
        return cls(
            analyzer=str(params.get("analyzer", "word")),
            ngram_range=tuple(params.get("ngram_range", (1, 2))),  # type: ignore[arg-type]
            min_df=params.get("min_df", 2),  # type: ignore[arg-type]
            max_features=params.get("max_features", 50000),  # type: ignore[arg-type]
            sublinear_tf=bool(params.get("sublinear_tf", True)),
            lowercase=bool(params.get("lowercase", False)),
        )


def build_tfidf_features(
    cleaned_texts: Mapping[str, Sequence[str]],
    tfidf_config: TfidfConfig,
    *,
    fit_split: str = "train",
    splits: Sequence[str] = SPLITS,
) -> Tuple[TfidfVectorizer, Dict[str, sparse.spmatrix]]:
    """Fit TF-IDF on ``fit_split`` only and transform every split.

    The vectorizer is fitted **exclusively** on ``cleaned_texts[fit_split]``
    (the training split); validation/test are only transformed, never used to
    learn the vocabulary or IDF weights.
    """
    if fit_split not in cleaned_texts:
        raise KeyError(f"fit_split {fit_split!r} not present in cleaned_texts")
    missing = [s for s in splits if s not in cleaned_texts]
    if missing:
        raise KeyError(f"cleaned_texts missing splits: {missing}")

    vectorizer = TfidfVectorizer(
        analyzer=tfidf_config.analyzer,
        ngram_range=tuple(tfidf_config.ngram_range),
        min_df=tfidf_config.min_df,
        max_features=tfidf_config.max_features,
        sublinear_tf=tfidf_config.sublinear_tf,
        lowercase=tfidf_config.lowercase,
        dtype=TFIDF_DTYPE,
    )
    vectorizer.fit([str(text) for text in cleaned_texts[fit_split]])
    matrices = {
        split: vectorizer.transform([str(text) for text in cleaned_texts[split]])
        for split in splits
    }
    return vectorizer, matrices


def build_tfidf_bundle(
    config_entry: Mapping[str, object],
    cleaned_texts: Mapping[str, Sequence[str]],
    labels: Mapping[str, np.ndarray],
    *,
    fit_split: str = "train",
    class_names: Optional[Sequence[str]] = None,
) -> FeatureBundle:
    """Fit a TF-IDF configuration and package it as a :class:`FeatureBundle`."""
    analyzer = str(config_entry.get("analyzer", "word"))
    params = (
        config.TFIDF_CHAR_PARAMS if analyzer == "char" else config.TFIDF_WORD_PARAMS
    )
    tfidf_config = TfidfConfig.from_params(params)
    vectorizer, matrices = build_tfidf_features(
        cleaned_texts, tfidf_config, fit_split=fit_split
    )

    bundle = FeatureBundle(
        key=str(config_entry["key"]),
        label=str(config_entry.get("label", config_entry["key"])),
        feature_type=str(config_entry["key"]),
        feature_family="tfidf",
        is_sparse=True,
        num_features=int(matrices[fit_split].shape[1]),
        class_names=tuple(class_names) if class_names else (),
        vectorizer=vectorizer,
    )
    for split in SPLITS:
        bundle.X[split] = matrices[split]
        bundle.y[split] = np.asarray(labels[split]).astype(str)
        bundle.row_ids[split] = np.arange(len(bundle.y[split]), dtype=np.int64)
    return bundle


# ---------------------------------------------------------------------------
# Optional dimensionality reduction for tree ensembles on sparse features
# ---------------------------------------------------------------------------

def svd_reduce(
    X,
    n_components: int,
    seed: int = config.RANDOM_SEED,
) -> Tuple[TruncatedSVD, np.ndarray]:
    """Fit ``TruncatedSVD`` on ``X`` (train only) and return (svd, dense)."""
    max_components = min(X.shape) - 1
    if max_components < 2:
        raise ValueError(
            f"Cannot reduce {X.shape} with TruncatedSVD: need at least "
            "2 samples/features"
        )
    n_components = int(min(n_components, max_components))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    dense = np.asarray(svd.fit_transform(X), dtype=np.float32)
    return svd, dense
