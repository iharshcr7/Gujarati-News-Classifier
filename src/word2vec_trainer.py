"""Word2Vec CBOW and Skip-gram training for Gujarati SHC headlines (Phase 3).

Trains two Gensim Word2Vec models on the SHC *training* split only:

    Model A - CBOW        (sg=0)
    Model B - Skip-gram   (sg=1)

Shared hyperparameters live in :mod:`config` and are recorded in the training
report so the experiment is reproducible.

Design goals
------------
- No data leakage: embeddings are trained on training headlines only. Validation
  and test headlines are never seen by Word2Vec. Category labels are never used.
- Original CSV files are never modified.
- The existing Phase 2 preprocessing (:func:`src.preprocessing.preprocess`) is
  used to tokenize training headlines consistently.
- Empty token lists and out-of-vocabulary words are handled gracefully.
- Models are saved with Gensim's native ``save`` so they can be reloaded for
  inference and document-vector generation in later phases.

Only the standard library, Gensim, and the project's own modules are used.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gensim.models import Word2Vec
from gensim.models.word2vec import FAST_VERSION

import config
from src.data_loader import DatasetSplits, load_all_splits, load_train, validate_all
from src.preprocessing import (
    GUJARATI_LO,
    GUJARATI_HI,
    clean_splits,
    has_gujarati,
    preprocess,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostic sampling
# ---------------------------------------------------------------------------

DIAGNOSTIC_SAMPLE_HEADLINES = 50
DIAGNOSTIC_EXAMPLE_WORDS = 8

# Fail-fast thresholds.
MAX_EMPTY_TOKEN_FRACTION = 0.50  # abort if more than half of headlines tokenize to nothing
MIN_GUJARATI_VOCAB_WORDS = 5     # abort if fewer Gujarati words are learned


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of training a single Word2Vec model."""

    model: Word2Vec
    sg: int
    vector_size: int
    window: int
    min_count: int
    workers: int
    seed: int
    epochs: int
    num_headlines: int
    num_tokenized_headlines: int
    empty_headlines: int
    empty_fraction: float
    vocab_size: int
    embedding_dim: int
    discarded_tokens_below_min_count: int
    elapsed_seconds: float
    model_path: Path
    model_file_size_bytes: int


# ---------------------------------------------------------------------------
# Token extraction (training text only)
# ---------------------------------------------------------------------------

def extract_train_sentences(
    splits: Optional[DatasetSplits] = None,
) -> List[List[str]]:
    """Tokenize the SHC training headlines for Word2Vec training.

    Only the ``train`` split is used; validation and test headlines are never
    tokenized here, which prevents embedding data leakage. Missing or invalid
    text yields an empty token list (kept in the sequence so headline counts
    stay aligned with the original split).

    Parameters
    ----------
    splits:
        Loaded splits. When ``None`` the configured splits are loaded via
        :func:`src.data_loader.load_all_splits`.

    Returns
    -------
    list of list of str
        One token list per training headline, in original row order.
    """
    if splits is None:
        splits = load_all_splits()

    sentences: List[List[str]] = []
    for text in splits.train[config.TEXT_COLUMN]:
        prepared = preprocess(text)  # stopword removal disabled by default
        sentences.append(prepared.tokens)
    return sentences


def count_discarded_tokens(
    sentences: Sequence[Sequence[str]], min_count: int
) -> int:
    """Return the number of distinct tokens dropped by ``min_count``.

    A token is "discarded" when its total corpus frequency is strictly below
    ``min_count``. This is measurable directly from the tokenized corpus.
    """
    frequencies: Counter[str] = Counter()
    for tokens in sentences:
        frequencies.update(tokens)
    return sum(1 for count in frequencies.values() if count < min_count)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _gujarati_char_count(series: pd.Series) -> int:
    """Total Gujarati Unicode chars (U+0A80..U+0AFF) in a text column."""
    import pandas as pd  # noqa: F401  (pd already imported via preprocessing path)

    return int(
        series.fillna("").astype(str).map(
            lambda t: sum(1 for c in t if GUJARATI_LO <= ord(c) <= GUJARATI_HI)
        ).sum()
    )


def _empty_fraction(sentences: Sequence[Sequence[str]]) -> float:
    if not sentences:
        return 1.0
    return sum(1 for s in sentences if not s) / len(sentences)


def validate_training_readiness(
    raw_texts: Sequence[str], sentences: Sequence[Sequence[str]]
) -> None:
    """Fail fast if Gujarati text is missing or tokenization is severely broken.

    Raises ValueError with a clear diagnostic so training never silently
    continues on question-mark-filled or empty text.
    """
    gujarati_rows = sum(1 for t in raw_texts if has_gujarati(str(t)))
    gujarati_chars = _gujarati_char_count(pd.Series(list(raw_texts)))
    if gujarati_rows == 0 or gujarati_chars == 0:
        raise ValueError(
            "Training aborted: no Gujarati Unicode characters (U+0A80..U+0AFF) "
            "found in the training text. Tokenization is broken or the data is "
            "corrupted; do not train on question-mark-filled text."
        )
    empty_frac = _empty_fraction(sentences)
    if empty_frac > MAX_EMPTY_TOKEN_FRACTION:
        raise ValueError(
            f"Training aborted: {empty_frac*100:.1f}% of training headlines "
            "tokenize to an empty list. Tokenization is severely broken."
        )
    logger.info(
        "Training readiness OK: Gujarati-bearing rows=%d, Gujarati chars=%d, "
        "empty-token fraction=%.4f",
        gujarati_rows,
        gujarati_chars,
        empty_frac,
    )


def train_word2vec(
    sentences: Sequence[Sequence[str]],
    *,
    sg: int,
    vector_size: int = config.W2V_VECTOR_SIZE,
    window: int = config.W2V_WINDOW,
    min_count: int = config.W2V_MIN_COUNT,
    workers: int = config.W2V_WORKERS,
    seed: int = config.W2V_SEED,
    epochs: int = config.W2V_EPOCHS,
) -> Tuple[Word2Vec, float]:
    """Train a single Word2Vec model and return it with elapsed seconds.

    Gensim trains for ``epochs`` passes over the corpus. The ``seed`` makes the
    result reproducible as far as Gensim and the execution environment allow.

    Empty token lists are skipped by Gensim automatically and never raise.
    """
    if sg not in (0, 1):
        raise ValueError(f"sg must be 0 (CBOW) or 1 (Skip-gram), got {sg!r}")

    start = time.perf_counter()
    # Build without passing `sentences` so auto-training does not run; this lets
    # us handle an all-empty corpus (no vocabulary) without raising.
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
        sg=sg,
        epochs=epochs,
    )
    model.build_vocab(corpus_iterable=list(sentences), update=False)
    if len(model.wv) > 0:
        model.train(
            corpus_iterable=list(sentences),
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=model.epochs,
        )
    elapsed = time.perf_counter() - start
    logger.info(
        "Word2Vec (sg=%s) trained: vocab=%d, dim=%d, epochs=%d, "
        "workers=%d, elapsed=%.2fs",
        sg,
        len(model.wv),
        model.vector_size,
        epochs,
        workers,
        elapsed,
    )
    return model, elapsed


# ---------------------------------------------------------------------------
# Native Gensim artifact saving (with existing-artifact guard)
# ---------------------------------------------------------------------------

def _is_loadable_word2vec(path: Path) -> bool:
    """Return True if ``path`` holds a loadable Gensim Word2Vec model."""
    try:
        Word2Vec.load(str(path))
        return True
    except Exception:  # corrupt / incompatible / not a Gensim artifact
        return False


def save_model(
    model: Word2Vec,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save ``model`` with Gensim's native ``save``.

    If an artifact already exists at ``path`` and is a loadable Word2Vec model,
    it is preserved (not overwritten) unless ``overwrite=True``. An existing
    artifact that cannot be loaded is treated as incompatible and is replaced,
    with the fact recorded in the returned log message.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        if _is_loadable_word2vec(path):
            logger.info(
                "Preserving existing loadable Word2Vec artifact: %s (not overwritten)",
                path,
            )
            return path
        logger.warning(
            "Existing artifact at %s is not a loadable Word2Vec model; replacing it.",
            path,
        )

    model.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Diagnostic examples
# ---------------------------------------------------------------------------

def _pick_example_words(model: Word2Vec, n: int) -> List[str]:
    """Pick up to ``n`` vocabulary words for nearest-neighbor diagnostics.

    Gujarati-script words are preferred (to demonstrate the embedding preserves
    Gujarati), falling back to the remaining vocabulary in order.
    """
    vocab = list(model.wv.index_to_key)
    gujarati = [
        w for w in vocab if any(0x0A80 <= ord(ch) <= 0x0AFF for ch in w)
    ]
    others = [w for w in vocab if w not in set(gujarati)]
    chosen = gujarati[:n]
    if len(chosen) < n:
        chosen.extend(others[: n - len(chosen)])
    return chosen


def _nearest_neighbors(model: Word2Vec, word: str, topn: int = 5) -> List[Tuple[str, float]]:
    """Return ``(neighbor, similarity)`` for ``word`` or an empty list if OOV."""
    if word not in model.wv:
        return []
    return [
        (neighbor, float(sim))
        for neighbor, sim in model.wv.most_similar(word, topn=topn)
    ]


def _headline_oov_stats(
    model: Word2Vec,
    sentences: Sequence[Sequence[str]],
    sample_size: int,
) -> Tuple[int, int]:
    """Count empty and fully-out-of-vocabulary headlines in a diagnostic sample."""
    sample = list(sentences[:sample_size])
    empty = 0
    fully_oov = 0
    for tokens in sample:
        if not tokens:
            empty += 1
            continue
        if all(token not in model.wv for token in tokens):
            fully_oov += 1
    return empty, fully_oov


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def _gujarati_vocab_words(model: Word2Vec, limit: int = 20) -> List[str]:
    """Gujarati-script words in the model vocabulary (up to ``limit``)."""
    return [
        w for w in model.wv.index_to_key
        if any(GUJARATI_LO <= ord(ch) <= GUJARATI_HI for ch in w)
    ][:limit]


def _is_gujarati_token(token: str) -> bool:
    return any(GUJARATI_LO <= ord(ch) <= GUJARATI_HI for ch in token)


def _top_gujarati_tokens(
    sentences: Sequence[Sequence[str]], n: int = 15
) -> List[Tuple[str, int]]:
    """Most frequent Gujarati tokens in the training corpus."""
    from collections import Counter

    freq: Counter[str] = Counter()
    for toks in sentences:
        for t in toks:
            if _is_gujarati_token(t):
                freq[t] += 1
    return freq.most_common(n)


def _gujarati_neighbor_fraction(
    model: Word2Vec, word: str, topn: int = 5
) -> Tuple[int, int]:
    """Return ``(gujarati_neighbors, total_neighbors)`` for ``word``."""
    neighbors = _nearest_neighbors(model, word, topn=topn)
    if not neighbors:
        return 0, 0
    guj = sum(1 for nbr, _ in neighbors if _is_gujarati_token(nbr))
    return guj, len(neighbors)


def build_training_report(
    cbow: TrainingResult,
    skipgram: TrainingResult,
    sentences: Sequence[Sequence[str]],
    *,
    example_word_topn: int = 5,
    raw_gujarati_chars: int = 0,
    gujarati_rows: int = 0,
) -> str:
    """Build the full Word2Vec training analysis report string (UTF-8 safe)."""
    lines: List[str] = []
    sep = "-" * 72

    lines.append("=" * 72)
    lines.append("GUJARATI SHC WORD2VEC TRAINING REPORT (PHASE 3)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. EXPERIMENT SCOPE AND DATA LEAKAGE PREVENTION")
    lines.append(sep)
    lines.append("  Dataset format: SHC")
    lines.append(f"  Text column:    {config.TEXT_COLUMN!r}")
    lines.append(f"  Label column:   {config.LABEL_COLUMN!r} (unused by Word2Vec)")
    lines.append("  Train rows:     training text only")
    lines.append("  Validation rows: NOT used for embedding training")
    lines.append("  Test rows:      NOT used for embedding training")
    lines.append(
        "  Category labels: not used to train Word2Vec (unsupervised)."
    )
    lines.append(
        "  Original CSV files: never modified."
    )
    lines.append("")
    lines.append("2. TRAINING DATA")
    lines.append(sep)
    lines.append(f"  Training headlines processed: {cbow.num_headlines}")
    lines.append(f"  Tokenized headlines:          {cbow.num_tokenized_headlines}")
    lines.append(f"  Empty token lists:            {cbow.empty_headlines}  ({cbow.empty_fraction*100:.2f}%)")
    lines.append(
        f"  Discarded tokens (freq < min_count={cbow.min_count}): "
        f"{cbow.discarded_tokens_below_min_count}"
    )
    lines.append(
        f"  Gujarati-bearing rows:        {gujarati_rows}"
    )
    lines.append(
        f"  Gujarati Unicode chars (pre-tokenization): {raw_gujarati_chars}"
    )
    lines.append("")
    lines.append("3. SHARED TRAINING PARAMETERS")
    lines.append(sep)
    for label, result in [("CBOW (sg=0)", cbow), ("Skip-gram (sg=1)", skipgram)]:
        lines.append(f"  --- {label} ---")
        lines.append(f"    sg:               {result.sg}")
        lines.append(f"    vector_size:      {result.vector_size}")
        lines.append(f"    window:           {result.window}")
        lines.append(f"    min_count:        {result.min_count}")
        lines.append(f"    workers:          {result.workers}")
        lines.append(f"    seed:             {result.seed}")
        lines.append(f"    epochs:           {result.epochs}")
        lines.append(f"    elapsed time:     {result.elapsed_seconds:.2f} seconds")
        lines.append(f"    gensim fast version: {FAST_VERSION}")
        lines.append("")
    lines.append("4. MODEL DIAGNOSTICS")
    lines.append(sep)
    for label, result in [("CBOW (sg=0)", cbow), ("Skip-gram (sg=1)", skipgram)]:
        lines.append(f"  --- {label} ---")
        lines.append(f"    Vocabulary size:            {result.vocab_size}")
        lines.append(f"    Embedding dimensions:       {result.embedding_dim}")
        lines.append(f"    Model artifact path:        {result.model_path}")
        lines.append(f"    Model artifact size (bytes): {result.model_file_size_bytes}")
        lines.append("")
    lines.append("5. COMPARISON: CBOW vs Skip-gram")
    lines.append(sep)
    lines.append(
        f"  Vocabulary size   : CBOW={cbow.vocab_size}  "
        f"Skip-gram={skipgram.vocab_size}"
    )
    lines.append(
        f"  Training time (s) : CBOW={cbow.elapsed_seconds:.2f}  "
        f"Skip-gram={skipgram.elapsed_seconds:.2f}"
    )
    lines.append(
        f"  Embedding dims    : CBOW={cbow.embedding_dim}  "
        f"Skip-gram={skipgram.embedding_dim}"
    )
    lines.append(
        "  Note: Word2Vec training loss is not used to claim classification "
        "superiority; classifiers are implemented in a later phase."
    )
    lines.append("")

    # Nearest-neighbor examples for both models.
    cbow_model = cbow.model
    skipgram_model = skipgram.model
    cbow_words = _pick_example_words(cbow_model, DIAGNOSTIC_EXAMPLE_WORDS)
    skipgram_words = _pick_example_words(skipgram_model, DIAGNOSTIC_EXAMPLE_WORDS)

    lines.append("6. NEAREST-NEIGHBOR EXAMPLES - CBOW (sg=0)")
    lines.append(sep)
    for word in cbow_words:
        neighbors = _nearest_neighbors(cbow_model, word, topn=example_word_topn)
        if neighbors:
            nbr_str = "  |  ".join(
                f"{nbr} ({sim:.3f})" for nbr, sim in neighbors
            )
            lines.append(f"  '{word}' -> {nbr_str}")
        else:
            lines.append(f"  '{word}' -> (in vocabulary, no neighbors computed)")
    lines.append("")

    lines.append("7. NEAREST-NEIGHBOR EXAMPLES - SKIP-GRAM (sg=1)")
    lines.append(sep)
    for word in skipgram_words:
        neighbors = _nearest_neighbors(skipgram_model, word, topn=example_word_topn)
        if neighbors:
            nbr_str = "  |  ".join(
                f"{nbr} ({sim:.3f})" for nbr, sim in neighbors
            )
            lines.append(f"  '{word}' -> {nbr_str}")
        else:
            lines.append(f"  '{word}' -> (in vocabulary, no neighbors computed)")
    lines.append("")

    # Diagnostic sample OOV stats.
    cbow_empty, cbow_fully_oov = _headline_oov_stats(
        cbow_model, sentences, DIAGNOSTIC_SAMPLE_HEADLINES
    )
    sg_empty, sg_fully_oov = _headline_oov_stats(
        skipgram_model, sentences, DIAGNOSTIC_SAMPLE_HEADLINES
    )
    lines.append("8. DIAGNOSTIC SAMPLE (first %d training headlines)" % DIAGNOSTIC_SAMPLE_HEADLINES)
    lines.append(sep)
    lines.append(f"  CBOW     : empty={cbow_empty}  fully out-of-vocabulary={cbow_fully_oov}")
    lines.append(f"  Skip-gram: empty={sg_empty}  fully out-of-vocabulary={sg_fully_oov}")
    lines.append("")
    lines.append("9. GUJARATI TEXT PRESERVATION AND VOCABULARY CHECKS")
    lines.append(sep)
    gujarati_vocab_cbow = _gujarati_vocab_words(cbow_model)
    gujarati_vocab_skipgram = _gujarati_vocab_words(skipgram_model)
    lines.append(
        f"  Gujarati words in CBOW vocab (sample): {gujarati_vocab_cbow}"
    )
    lines.append(
        f"  Gujarati words in Skip-gram vocab (sample): {gujarati_vocab_skipgram}"
    )
    lines.append(
        f"  Total Gujarati words in CBOW vocab: {len(_gujarati_vocab_words(cbow_model, limit=10_000))}"
    )
    lines.append(
        f"  Total Gujarati words in Skip-gram vocab: {len(_gujarati_vocab_words(skipgram_model, limit=10_000))}"
    )
    lines.append("")

    lines.append("  Top Gujarati tokens in training corpus and in-vocabulary check:")
    top_gujarati = _top_gujarati_tokens(sentences, n=15)
    for token, count in top_gujarati:
        in_cbow = token in cbow_model.wv
        in_sg = token in skipgram_model.wv
        lines.append(
            f"    {token!r} (freq={count})  "
            f"in CBOW={in_cbow}  in Skip-gram={in_sg}"
        )
    lines.append("")

    lines.append("  Nearest-neighbor Gujarati-script fraction (Gujarati query words):")
    cbow_query = [w for w, _ in top_gujarati[:3] if w in cbow_model.wv]
    sg_query = [w for w, _ in top_gujarati[:3] if w in skipgram_model.wv]
    if cbow_query:
        total = 0
        guj_total = 0
        for w in cbow_query:
            g, n = _gujarati_neighbor_fraction(cbow_model, w)
            guj_total += g
            total += n
        lines.append(
            f"    CBOW : {guj_total}/{total} neighbors are Gujarati-script "
            f"({guj_total/total*100:.0f}% if total else N/A)"
        )
    if sg_query:
        total = 0
        guj_total = 0
        for w in sg_query:
            g, n = _gujarati_neighbor_fraction(skipgram_model, w)
            guj_total += g
            total += n
        lines.append(
            f"    Skip-gram: {guj_total}/{total} neighbors are Gujarati-script "
            f"({guj_total/total*100:.0f}% if total else N/A)"
        )
    lines.append("")

    min_gujarati = min(
        len(_gujarati_vocab_words(cbow_model, limit=10_000)),
        len(_gujarati_vocab_words(skipgram_model, limit=10_000)),
    )
    if min_gujarati < MIN_GUJARATI_VOCAB_WORDS:
        lines.append(
            f"  WARNING: only {min_gujarati} Gujarati words learned; expected "
            f">= {MIN_GUJARATI_VOCAB_WORDS}. Gujarati coverage may be limited."
        )
    else:
        lines.append(
            f"  Gujarati vocab coverage adequate: >= {MIN_GUJARATI_VOCAB_WORDS} "
            f"Gujarati words learned per model."
        )
    lines.append(
        "  Validation and test Gujarati text is NOT used for embedding training."
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF WORD2VEC TRAINING REPORT")
    lines.append("=" * 72)

    return "\n".join(lines)


def write_report(report: str, filename: str = config.W2V_REPORT_FILENAME) -> Path:
    """Write the report to ``reports/`` as UTF-8 and return the report path."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / filename
    report_path.write_text(report, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def train_both_models(
    *,
    splits: Optional[DatasetSplits] = None,
    overwrite: bool = False,
) -> Tuple[TrainingResult, TrainingResult, str, Path]:
    """Train CBOW and Skip-gram models, save artifacts, and write the report.

    Returns ``(cbow_result, skipgram_result, report, report_path)``.
    """
    if splits is None:
        raw_splits = load_all_splits()
        validate_all(raw_splits)
        # Fit embeddings on the cleaned *training* split only. Invalid rows are
        # removed; train/validation/test are never combined or repartitioned.
        splits, _ = clean_splits(raw_splits)

    sentences = extract_train_sentences(splits)
    raw_texts = list(splits.train[config.TEXT_COLUMN].fillna("").astype(str))
    validate_training_readiness(raw_texts, sentences)

    gujarati_rows = sum(1 for t in raw_texts if has_gujarati(t))
    raw_gujarati_chars = _gujarati_char_count(
        pd.Series(raw_texts)
    )

    num_headlines = len(splits.train)
    num_tokenized = len(sentences)
    empty_headlines = sum(1 for tokens in sentences if not tokens)
    empty_fraction = empty_headlines / num_headlines if num_headlines else 1.0
    discarded = count_discarded_tokens(sentences, config.W2V_MIN_COUNT)

    results: List[TrainingResult] = []
    for sg, name in [
        (config.W2V_SG_CBOW, "CBOW"),
        (config.W2V_SG_SKIPGRAM, "Skip-gram"),
    ]:
        logger.info("Training Word2Vec %s (sg=%s) ...", name, sg)
        model, elapsed = train_word2vec(
            sentences,
            sg=sg,
            vector_size=config.W2V_VECTOR_SIZE,
            window=config.W2V_WINDOW,
            min_count=config.W2V_MIN_COUNT,
            workers=config.W2V_WORKERS,
            seed=config.W2V_SEED,
            epochs=config.W2V_EPOCHS,
        )
        if sg == config.W2V_SG_CBOW:
            model_path = config.W2V_MODEL_CBOW_PATH
        else:
            model_path = config.W2V_MODEL_SKIPGRAM_PATH
        save_model(model, model_path, overwrite=overwrite)
        file_size = model_path.stat().st_size if model_path.exists() else 0
        results.append(
            TrainingResult(
                model=model,
                sg=sg,
                vector_size=config.W2V_VECTOR_SIZE,
                window=config.W2V_WINDOW,
                min_count=config.W2V_MIN_COUNT,
                workers=config.W2V_WORKERS,
                seed=config.W2V_SEED,
                epochs=config.W2V_EPOCHS,
                num_headlines=num_headlines,
                num_tokenized_headlines=num_tokenized,
                empty_headlines=empty_headlines,
                empty_fraction=empty_fraction,
                vocab_size=len(model.wv),
                embedding_dim=model.vector_size,
                discarded_tokens_below_min_count=discarded,
                elapsed_seconds=elapsed,
                model_path=model_path,
                model_file_size_bytes=file_size,
            )
        )

    cbow, skipgram = results
    report = build_training_report(
        cbow,
        skipgram,
        sentences,
        raw_gujarati_chars=raw_gujarati_chars,
        gujarati_rows=gujarati_rows,
    )
    report_path = write_report(report)
    return cbow, skipgram, report, report_path


def run_training_pipeline(
    *,
    overwrite: bool = False,
    report_filename: Optional[str] = None,
) -> Tuple[TrainingResult, TrainingResult, str, Path]:
    """Convenience entry point mirroring the Phase 2 pipeline conventions."""
    config.validate_paths()
    cbow, skipgram, report, report_path = train_both_models(
        splits=None, overwrite=overwrite
    )
    if report_filename is not None and report_filename != config.W2V_REPORT_FILENAME:
        # Re-write under an alternate name if requested.
        report_path = write_report(report, filename=report_filename)
    return cbow, skipgram, report, report_path


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cbow, skipgram, report, report_path = run_training_pipeline()
    print(report)
    print(f"\nReport saved to: {report_path}")
    print(f"CBOW model   : {cbow.model_path} ({cbow.model_file_size_bytes} bytes)")
    print(
        f"Skip-gram model: {skipgram.model_path} ({skipgram.model_file_size_bytes} bytes)"
    )
