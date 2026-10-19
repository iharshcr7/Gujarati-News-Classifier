"""Gujarati headline document-vector generation (Phase 4).

Builds fixed-length (100-dimensional) document vectors for every Gujarati SHC
headline using the CBOW and Skip-gram Word2Vec models trained in Phase 3.

Method
------
For each headline the existing Phase 2 preprocessing
(:func:`src.preprocessing.preprocess`, whitespace tokenization, stopword
removal disabled) produces the token list -- the *same* pipeline used to train
the embeddings. Each in-vocabulary token's embedding is averaged
(mean-pooling); out-of-vocabulary tokens are ignored. A headline with no
in-vocabulary tokens (empty token list or all tokens out of vocabulary) gets a
documented zero-vector fallback.

Design goals
------------
- No data leakage: embeddings were trained on the training split only; this
  module only *reads* the saved models and never trains or updates them.
- Validation and test splits stay separate; train/validation/test are never
  combined or repartitioned, and the original CSVs are never modified.
- Row order, a stable ``row_id`` (positional index in the cleaned split), and
  the source ``labels`` are preserved exactly.
- All output vectors are finite and have exactly ``model.vector_size`` columns.
- Both models are loaded once and each split is tokenized once.

Outputs (``data/processed/``)
-----------------------------
- ``<split-stem>.<model>.docvecs.npz``  arrays: ``X`` (n x dim float32),
  ``labels`` (unicode), ``row_ids`` (int64). Compressed NumPy archive.
- ``<split-stem>.<model>.docvecs.meta.json`` metadata (embedding model, vector
  dimension, preprocessing method, source split, fallback counts, ...).
Report: ``reports/document_vectors_report.txt`` (UTF-8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

import config
from src.data_loader import DatasetSplits, load_all_splits, validate_all
from src.preprocessing import clean_splits, preprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Human-readable preprocessing description recorded in metadata/report.
DEFAULT_PREPROCESSING_METHOD = (
    "src.preprocessing.preprocess: URL/HTML/email removal, punctuation to "
    "space, whitespace-tokenization, stopword removal disabled "
    "(identical to the Word2Vec training pipeline)"
)

# Embedding model names, in report order. Paths are resolved from config at
# call time (not import time) so tests can monkeypatch config cleanly.
DOCVEC_MODEL_NAMES: Tuple[str, ...] = ("cbow", "skipgram")

SPLIT_ORDER: Tuple[str, ...] = ("train", "validation", "test")


def _resolve_default_models() -> Tuple[Tuple[str, Path], ...]:
    """Return the configured (model_name, path) pairs, resolved at call time."""
    return (
        ("cbow", config.W2V_MODEL_CBOW_PATH),
        ("skipgram", config.W2V_MODEL_SKIPGRAM_PATH),
    )


# ---------------------------------------------------------------------------
# Core pooling
# ---------------------------------------------------------------------------

def mean_pool_tokens(
    tokens: Sequence[str],
    key_to_index: Mapping[str, int],
    vectors: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Mean-pool the embeddings of ``tokens``; ignore out-of-vocabulary ones.

    Parameters
    ----------
    tokens:
        Token list for a single headline.
    key_to_index:
        Mapping from vocabulary token to its row index in ``vectors``
        (``model.wv.key_to_index``).
    vectors:
        Embedding matrix of shape ``(vocab_size, dim)`` (``model.wv.vectors``).

    Returns
    -------
    (vector, in_vocab_count)
        ``vector`` is a ``float32`` array of width ``vectors.shape[1]``. When no
        token is in vocabulary (including an empty token list) the zero vector
        is returned and the count is ``0``.
    """
    dim = vectors.shape[1]
    accumulator = np.zeros(dim, dtype=np.float64)
    in_vocab = 0
    for token in tokens:
        index = key_to_index.get(token)
        if index is not None:
            accumulator += vectors[index]
            in_vocab += 1
    if in_vocab == 0:
        return np.zeros(dim, dtype=np.float32), 0
    return (accumulator / in_vocab).astype(np.float32), in_vocab


def vectorize_tokens(
    token_lists: Sequence[Sequence[str]],
    model: Word2Vec,
    *,
    model_name: str,
    split: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Vectorize many headlines with a single model, returning (X, stats).

    ``X`` has shape ``(len(token_lists), model.vector_size)`` and ``float32``
    dtype. ``stats`` records ``fallback_count`` (rows with no in-vocabulary
    token), ``empty_token_count``, ``total_oov_tokens``, and
    ``total_in_vocab_tokens``.
    """
    key_to_index = model.wv.key_to_index
    vectors = model.wv.vectors
    dim = int(vectors.shape[1])
    num_rows = len(token_lists)

    X = np.zeros((num_rows, dim), dtype=np.float32)
    fallback_count = 0
    empty_token_count = 0
    total_oov_tokens = 0
    total_in_vocab_tokens = 0

    for row, tokens in enumerate(token_lists):
        if not tokens:
            empty_token_count += 1
        vector, in_vocab = mean_pool_tokens(tokens, key_to_index, vectors)
        X[row] = vector
        total_in_vocab_tokens += in_vocab
        total_oov_tokens += len(tokens) - in_vocab
        if in_vocab == 0:
            fallback_count += 1

    stats = {
        "fallback_count": fallback_count,
        "empty_token_count": empty_token_count,
        "total_oov_tokens": total_oov_tokens,
        "total_in_vocab_tokens": total_in_vocab_tokens,
    }
    return X, stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class DocumentVectorSet:
    """Document vectors for one (model, split) pair."""

    vectors: np.ndarray          # (num_rows, dim) float32, row order preserved
    labels: np.ndarray           # source labels, aligned by row
    row_ids: np.ndarray          # positional index in the cleaned split
    dim: int
    model_name: str
    split: str
    num_rows: int
    fallback_count: int
    empty_token_count: int
    total_oov_tokens: int
    total_in_vocab_tokens: int
    mean_in_vocab_tokens: float


def build_document_vector_set(
    token_lists: Sequence[Sequence[str]],
    labels: Sequence[str],
    model: Word2Vec,
    *,
    model_name: str,
    split: str,
    row_ids: Optional[Sequence[int]] = None,
) -> DocumentVectorSet:
    """Build a :class:`DocumentVectorSet` from pre-tokenized headlines.

    Labels and row ids stay aligned with ``token_lists`` by position. If
    ``row_ids`` is ``None`` the positional index ``0..n-1`` is used, which is
    the stable identifier of a row within the cleaned split.
    """
    if len(token_lists) != len(labels):
        raise ValueError(
            f"token_lists ({len(token_lists)}) and labels ({len(labels)}) "
            "must have the same length"
        )

    X, stats = vectorize_tokens(
        token_lists, model, model_name=model_name, split=split
    )
    num_rows = len(token_lists)
    if row_ids is None:
        row_ids_arr = np.arange(num_rows, dtype=np.int64)
    else:
        row_ids_arr = np.asarray(row_ids, dtype=np.int64)

    mean_in_vocab = (
        stats["total_in_vocab_tokens"] / num_rows if num_rows else 0.0
    )

    return DocumentVectorSet(
        vectors=X,
        labels=np.asarray(list(labels), dtype=str),
        row_ids=row_ids_arr,
        dim=int(model.wv.vectors.shape[1]),
        model_name=model_name,
        split=split,
        num_rows=num_rows,
        fallback_count=stats["fallback_count"],
        empty_token_count=stats["empty_token_count"],
        total_oov_tokens=stats["total_oov_tokens"],
        total_in_vocab_tokens=stats["total_in_vocab_tokens"],
        mean_in_vocab_tokens=mean_in_vocab,
    )


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_split(
    df: pd.DataFrame, text_column: str = config.TEXT_COLUMN
) -> List[List[str]]:
    """Tokenize every headline in ``df`` with the shared preprocessing pipeline.

    Returns one token list per row, in original row order. Invalid/missing text
    yields an empty list.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"text_column {text_column!r} not found in {list(df.columns)}"
        )
    return [preprocess(text).tokens for text in df[text_column]]


# ---------------------------------------------------------------------------
# Model loading / validation
# ---------------------------------------------------------------------------

def load_embedding_model(path: Path) -> Word2Vec:
    """Load a saved Gensim Word2Vec model (read-only)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Word2Vec model not found: {path}")
    return Word2Vec.load(str(path))


def validate_embedding_dimension(model: Word2Vec, expected: int) -> None:
    """Raise ValueError unless the model's embedding dimension is ``expected``."""
    model_dim = int(model.vector_size)
    wv_dim = int(model.wv.vector_size)
    if model_dim != expected or wv_dim != expected:
        raise ValueError(
            f"Unexpected embedding dimension: model.vector_size={model_dim}, "
            f"wv.vector_size={wv_dim}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def document_vector_filename(stem: str, model_name: str) -> str:
    """Return the ``.npz`` filename for a split stem and embedding model."""
    return f"{stem}.{model_name}.docvecs.npz"


def _metadata_filename(stem: str, model_name: str) -> str:
    return f"{stem}.{model_name}.docvecs.meta.json"


def save_document_vectors(dvs: DocumentVectorSet, path: Path) -> Path:
    """Persist vectors, labels, and row ids to a compressed ``.npz`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=dvs.vectors,
        labels=dvs.labels,
        row_ids=dvs.row_ids,
    )
    return path


def load_document_vectors(path: Path) -> Dict[str, np.ndarray]:
    """Load a document-vector ``.npz`` archive into a dict of arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_metadata(
    dvs: DocumentVectorSet,
    *,
    model_path: Path,
    model_sg: int,
    vocab_size: int,
    preprocessing_method: str = DEFAULT_PREPROCESSING_METHOD,
    source_file: str = "",
) -> Dict[str, object]:
    """Build the metadata dict recorded next to each vector file."""
    fallback_fraction = (
        dvs.fallback_count / dvs.num_rows if dvs.num_rows else 0.0
    )
    return {
        "embedding_model": dvs.model_name,
        "model_sg": int(model_sg),
        "vector_dimension": int(dvs.dim),
        "preprocessing_method": preprocessing_method,
        "source_split": dvs.split,
        "source_file": source_file,
        "num_rows": int(dvs.num_rows),
        "fallback_count": int(dvs.fallback_count),
        "fallback_fraction": float(fallback_fraction),
        "empty_token_count": int(dvs.empty_token_count),
        "out_of_vocabulary_tokens": int(dvs.total_oov_tokens),
        "in_vocabulary_tokens": int(dvs.total_in_vocab_tokens),
        "mean_in_vocabulary_tokens": float(dvs.mean_in_vocab_tokens),
        "vocab_size": int(vocab_size),
        "model_path": str(model_path),
    }


def write_metadata(meta: Mapping[str, object], path: Path) -> Path:
    """Write metadata as UTF-8 JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_document_vector_report(
    entries: Sequence[Mapping[str, object]],
    *,
    input_counts: Mapping[str, int],
    model_info: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> str:
    """Build the human-readable document-vector generation report (UTF-8 safe).

    ``entries`` is a sequence of mappings with keys ``model``, ``split``,
    ``dvs`` (a :class:`DocumentVectorSet`), and optional ``meta``,
    ``output_file``, ``metadata_file``.
    """
    model_info = model_info or {}
    sep = "-" * 72
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GUJARATI SHC DOCUMENT VECTOR GENERATION REPORT (PHASE 4)")
    lines.append("=" * 72)
    lines.append("")

    lines.append("1. EXPERIMENT SCOPE AND DATA LEAKAGE PREVENTION")
    lines.append(sep)
    lines.append("  Dataset format: SHC")
    lines.append(f"  Text column:    {config.TEXT_COLUMN!r}")
    lines.append(f"  Label column:   {config.LABEL_COLUMN!r} (preserved, not used to build vectors)")
    lines.append("  Embeddings:     loaded read-only from Phase 3 artifacts")
    lines.append("  Word2Vec retraining: NO")
    lines.append("  Classifier training: NO")
    lines.append("  Vectorization:  mean-pooling of in-vocabulary token embeddings")
    lines.append("  Validation/test rows: never used to train or update Word2Vec")
    lines.append("  Original CSV files: never modified")
    lines.append("")

    lines.append("2. EMBEDDING MODELS (VALIDATED DIMENSIONS)")
    lines.append(sep)
    if model_info:
        for name, info in model_info.items():
            lines.append(f"  --- {name} ---")
            lines.append(f"    model path:        {info.get('path')}")
            lines.append(f"    sg:                {info.get('sg')}")
            lines.append(f"    vector dimension:  {info.get('vector_size')}")
            lines.append(f"    vocabulary size:   {info.get('vocab_size')}")
    else:
        seen = {}
        for entry in entries:
            dvs = entry["dvs"]
            seen[dvs.model_name] = dvs.dim
        for name, dim in seen.items():
            lines.append(f"  {name}: dimension={dim}")
    lines.append("")

    lines.append("3. ROW COUNTS (PER SPLIT AND MODEL)")
    lines.append(sep)
    for split in SPLIT_ORDER:
        lines.append(f"  --- {split} (input rows={input_counts.get(split, 'N/A')}) ---")
        for entry in entries:
            if entry["split"] != split:
                continue
            dvs = entry["dvs"]
            match = "OK" if input_counts.get(split) == dvs.num_rows else "MISMATCH"
            lines.append(
                f"    {dvs.model_name:9s}: vector rows={dvs.num_rows:>7}  "
                f"input rows={input_counts.get(split, 'N/A')}  [{match}]"
            )
    lines.append("")

    lines.append("4. ZERO-VECTOR FALLBACK RATES (NO IN-VOCABULARY TOKENS)")
    lines.append(sep)
    for entry in entries:
        dvs = entry["dvs"]
        pct = (dvs.fallback_count / dvs.num_rows * 100) if dvs.num_rows else 0.0
        lines.append(
            f"  {dvs.model_name:9s} {dvs.split:11s}: fallback={dvs.fallback_count:>6} / "
            f"{dvs.num_rows} ({pct:.4f}%)  empty_token_rows={dvs.empty_token_count}  "
            f"OOV tokens={dvs.total_oov_tokens}  in-vocab tokens={dvs.total_in_vocab_tokens}  "
            f"mean in-vocab/headline={dvs.mean_in_vocab_tokens:.2f}"
        )
    lines.append("")

    lines.append("5. DIMENSION AND NUMERICAL FINITENESS CHECKS")
    lines.append(sep)
    all_ok = True
    for entry in entries:
        dvs = entry["dvs"]
        finite = bool(np.isfinite(dvs.vectors).all())
        dim_ok = dvs.vectors.ndim == 2 and dvs.vectors.shape[1] == dvs.dim
        all_ok = all_ok and finite and dim_ok
        lines.append(
            f"  {dvs.model_name:9s} {dvs.split:11s}: dim={dvs.dim}  "
            f"shape={dvs.vectors.shape}  finite={finite}  dimension_ok={dim_ok}"
        )
    lines.append(f"  ALL CHECKS PASSED: {all_ok}")
    lines.append("")

    lines.append("6. LABEL AND ROW-ORDER ALIGNMENT CHECKS")
    lines.append(sep)
    for entry in entries:
        dvs = entry["dvs"]
        n = dvs.num_rows
        rows_ok = bool(np.array_equal(dvs.row_ids, np.arange(n, dtype=np.int64)))
        labels_ok = len(dvs.labels) == n
        lines.append(
            f"  {dvs.model_name:9s} {dvs.split:11s}: rows={n}  "
            f"row_ids_sequential={rows_ok}  labels_aligned={labels_ok}"
        )
    lines.append("")

    lines.append("7. OUTPUT FILES")
    lines.append(sep)
    for entry in entries:
        dvs = entry["dvs"]
        lines.append(
            f"  {dvs.model_name:9s} {dvs.split:11s}: "
            f"{entry.get('output_file', '(in-memory)')}"
        )
        if entry.get("metadata_file"):
            lines.append(f"    metadata: {entry['metadata_file']}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF DOCUMENT VECTOR GENERATION REPORT")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_report(
    report: str, filename: str = config.DOCVEC_REPORT_FILENAME
) -> Path:
    """Write the report to ``reports/`` as UTF-8 and return the report path."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / filename
    report_path.write_text(report, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _splits_with_names(
    cleaned: DatasetSplits,
) -> List[Tuple[str, pd.DataFrame]]:
    return [
        ("train", cleaned.train),
        ("validation", cleaned.validation),
        ("test", cleaned.test),
    ]


def generate_document_vectors(
    *,
    output_dir: Path = config.PROCESSED_DIR,
    report_filename: str = config.DOCVEC_REPORT_FILENAME,
    text_column: str = config.TEXT_COLUMN,
    models: Optional[Sequence[Tuple[str, Path]]] = None,
) -> Tuple[str, Path, Dict[str, Dict[str, DocumentVectorSet]]]:
    """Generate document vectors for both models and every split.

    Loads the configured SHC splits, removes invalid rows (identical to Phase 2
    / Phase 3 cleaning), tokenizes each split once with the shared preprocessing
    pipeline, then mean-pools with each embedding model. Outputs are written to
    ``output_dir`` and the report to ``reports/``.

    Returns ``(report, report_path, results)`` where ``results`` maps
    ``model_name -> split -> DocumentVectorSet``.
    """
    config.validate_paths()
    raw_splits = load_all_splits()
    validate_all(raw_splits)
    cleaned, _ = clean_splits(raw_splits)

    if models is None:
        models = _resolve_default_models()

    named_splits = _splits_with_names(cleaned)
    input_counts = {name: len(df) for name, df in named_splits}

    # Tokenize each split exactly once; reuse for every embedding model.
    token_lists: Dict[str, List[List[str]]] = {
        name: tokenize_split(df, text_column) for name, df in named_splits
    }
    labels: Dict[str, List[str]] = {
        name: [str(value) for value in df[config.LABEL_COLUMN]]
        for name, df in named_splits
    }

    stem_map = {
        "train": config.TRAIN_PATH.stem,
        "validation": config.VALID_PATH.stem,
        "test": config.TEST_PATH.stem,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, DocumentVectorSet]] = {}
    entries: List[Dict[str, object]] = []
    model_info: Dict[str, Dict[str, object]] = {}

    for model_name, model_path in models:
        model = load_embedding_model(model_path)
        validate_embedding_dimension(model, config.W2V_VECTOR_SIZE)
        model_info[model_name] = {
            "path": str(model_path),
            "sg": int(model.sg),
            "vector_size": int(model.vector_size),
            "vocab_size": len(model.wv),
        }
        results[model_name] = {}

        for name, _df in named_splits:
            dvs = build_document_vector_set(
                token_lists[name],
                labels[name],
                model,
                model_name=model_name,
                split=name,
            )
            out_file = output_dir / document_vector_filename(
                stem_map[name], model_name
            )
            save_document_vectors(dvs, out_file)

            meta = build_metadata(
                dvs,
                model_path=model_path,
                model_sg=int(model.sg),
                vocab_size=len(model.wv),
                source_file=f"{stem_map[name]}.csv",
            )
            meta_file = output_dir / _metadata_filename(stem_map[name], model_name)
            write_metadata(meta, meta_file)

            results[model_name][name] = dvs
            entries.append(
                {
                    "model": model_name,
                    "split": name,
                    "dvs": dvs,
                    "meta": meta,
                    "output_file": str(out_file),
                    "metadata_file": str(meta_file),
                }
            )

    report = build_document_vector_report(
        entries, input_counts=input_counts, model_info=model_info
    )
    report_path = write_report(report, filename=report_filename)
    return report, report_path, results


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    report, report_path, _ = generate_document_vectors()
    print(report)
    print(f"\nReport saved to: {report_path}")
