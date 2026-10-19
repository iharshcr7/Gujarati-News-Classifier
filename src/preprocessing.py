"""Gujarati text preprocessing and tokenization for SHC headlines.

Provides deterministic, reusable, Unicode-aware preprocessing for Gujarati
news headlines. Designed so the same functions can be applied at training
time and at inference time without any behavior drift.

Design goals
------------
- Never crash on missing, null, or non-string input.
- Preserve Gujarati characters (U+0A80..U+0AFF) and vowel signs
  (matras, e.g. U+0ABE..U+0AC5) exactly; no stemming, no transliteration.
- Remove URLs, HTML tags, and email addresses; replace other punctuation/
  symbols with whitespace so unrelated words are never joined.
- Collapse repeated whitespace; tokenize on whitespace.
- Optional, configurable Gujarati stopword removal (disabled by default).
- Original text and labels are always preserved by the DataFrame helpers.

Only the standard library plus pandas are used; no new dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from .data_loader import DatasetSplits, load_all_splits, validate_all

# ---------------------------------------------------------------------------
# Unicode ranges
# ---------------------------------------------------------------------------

GUJARATI_LO = 0x0A80
GUJARATI_HI = 0x0AFF

# ---------------------------------------------------------------------------
# Regular expressions (all deterministic, no look-around magic)
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# Keep ASCII letters/digits, Gujarati script characters, and whitespace;
# everything else (punctuation, symbols, corrupted bytes such as "?") becomes
# a space so words stay separated rather than glued together.
NON_TOKEN_RE = re.compile(r"[^a-zA-Z0-9\u0A80-\u0AFF\s]")
MULTISPACE_RE = re.compile(r"\s+")


def has_gujarati(text: str) -> bool:
    """Return True if ``text`` contains at least one Gujarati Unicode char."""
    return any(GUJARATI_LO <= ord(ch) <= GUJARATI_HI for ch in str(text))


def _strip_preserving_gujarati(text: str, pattern: "re.Pattern[str]") -> str:
    """Replace ``pattern`` matches with a space, re-emitting Gujarati chars.

    URLs, HTML tags, and email addresses are removed as noise, but any
    Gujarati-script characters embedded inside a removed span are re-inserted
    (separated by spaces) so no Gujarati character is silently dropped.
    """

    def repl(match) -> str:
        span = match.group(0)
        kept = "".join(ch for ch in span if GUJARATI_LO <= ord(ch) <= GUJARATI_HI)
        return f" {kept} " if kept else " "

    return pattern.sub(repl, text)


# ---------------------------------------------------------------------------
# Default Gujarati stopword list (documented; stopword removal is opt-in)
# ---------------------------------------------------------------------------
#
# Single-token Gujarati function words only. Multi-word expressions are
# deliberately excluded because they would never match a single token after
# whitespace tokenization. This list is a starting point to be evaluated
# against the data; it is NOT applied unless the caller opts in.

DEFAULT_GUJARATI_STOPWORDS: frozenset[str] = frozenset({
    # Conjunctions / particles
    "અને",     # and
    "અથવા",   # or
    "પણ",      # but
    "જ",       # emphatic particle
    "કે",      # that (complementizer)
    # Pronouns
    "તે",      # he / she / it / that
    "આ",       # this
    "તમે",     # you
    "હું",     # I
    "અમે",     # we
    "આપણે",   # we (inclusive)
    # Possessives
    "મારું", "મારા", "મારે",   # my
    "તમારું", "તમારા", "તમારે",  # your
    # Be / do verbs (single-token forms)
    "છે", "હશે", "હતું", "છું", "છો", "છીએ",
    "કરી", "કરે",
    # Postpositions / case markers
    "માં", "ના", "ની", "ને", "થી", "પર", "તરીકે", "દ્વારા",
    # Quantifiers / time adverbs
    "એક", "ખૂબ", "વધુ", "પછી", "હવે",
})


# ---------------------------------------------------------------------------
# Text-level cleaning
# ---------------------------------------------------------------------------

def is_valid_text(value) -> bool:
    """Return True if ``value`` is a string with at least one non-space char.

    Handles None, NaN, numbers, and other non-string input safely.
    """
    if not isinstance(value, str):
        return False
    return bool(value.strip())


def clean_text(
    text: Optional[str],
    *,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_emails: bool = True,
) -> str:
    """Clean a single Gujarati text string.

    Missing, null, or non-string input returns ``""``. URLs, HTML tags, and
    email addresses are removed (when enabled). Every remaining character
    outside the allowed set (ASCII alphanumerics, Gujarati script,
    whitespace) is replaced by a single space, and whitespace is collapsed.
    Gujarati vowel signs/diacritics are preserved untouched.
    """
    if not isinstance(text, str):
        return ""

    result = text
    if remove_urls:
        result = _strip_preserving_gujarati(result, URL_RE)
    if remove_html:
        result = _strip_preserving_gujarati(result, HTML_TAG_RE)
    if remove_emails:
        result = _strip_preserving_gujarati(result, EMAIL_RE)
    result = NON_TOKEN_RE.sub(" ", result)
    return normalize_whitespace(result)


def normalize_whitespace(text: str) -> str:
    """Collapse any run of whitespace into a single space and strip ends."""
    return MULTISPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenize cleaned Gujarati text into words.

    Splits on whitespace (Unicode-aware via ``str.split``), which preserves
    Gujarati vowel signs and never joins unrelated words. Empty or blank
    input returns an empty list.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


# ---------------------------------------------------------------------------
# Stopword removal
# ---------------------------------------------------------------------------

def remove_stopwords(
    tokens: Sequence[str],
    stopwords: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return ``tokens`` with stopwords removed.

    By default uses :data:`DEFAULT_GUJARATI_STOPWORDS`. Pass ``stopwords=set()``
    to disable filtering. Matching is exact and case-sensitive.
    """
    if stopwords is None:
        stopwords = DEFAULT_GUJARATI_STOPWORDS
    stop_set = set(stopwords)
    return [token for token in tokens if token not in stop_set]


# ``preprocess`` has a parameter named ``remove_stopwords`` which shadows this
# function inside its own scope, so keep an alias for internal calls.
_remove_stopwords = remove_stopwords


# ---------------------------------------------------------------------------
# Combined preprocessing result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessedText:
    """Result of preprocessing a single text value."""

    original: str       # original input (non-str input becomes "")
    cleaned: str        # cleaned text
    tokens: List[str]   # tokenized words
    num_tokens: int     # len(tokens)
    is_empty: bool      # True when no tokens remain


def preprocess(
    text,
    *,
    remove_stopwords: bool = False,
    stopwords: Optional[Iterable[str]] = None,
) -> PreprocessedText:
    """Clean and tokenize a single text value.

    Parameters
    ----------
    text:
        Raw input; None/NaN/non-string yields an empty result.
    remove_stopwords:
        If True, remove stopwords from the token list (disabled by default).
    stopwords:
        Custom stopword iterable; defaults to
        :data:`DEFAULT_GUJARATI_STOPWORDS` when removal is enabled.
    """
    original = text if isinstance(text, str) else ""
    cleaned = clean_text(original)
    tokens = tokenize(cleaned)
    if remove_stopwords:
        tokens = _remove_stopwords(tokens, stopwords=stopwords)
    return PreprocessedText(
        original=original,
        cleaned=cleaned,
        tokens=tokens,
        num_tokens=len(tokens),
        is_empty=len(tokens) == 0,
    )


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing (original text and labels preserved)
# ---------------------------------------------------------------------------

def add_preprocessed_columns(
    df: pd.DataFrame,
    text_column: str = config.TEXT_COLUMN,
    *,
    cleaned_column: str = "text_clean",
    tokens_column: str = "tokens",
    num_tokens_column: str = "num_tokens",
    empty_column: str = "text_is_empty",
    remove_stopwords: bool = False,
    stopwords: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with preprocessed columns added.

    The input DataFrame is not modified; the original ``text`` column and all
    other columns (including ``labels``) are preserved untouched.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"text_column {text_column!r} not found in DataFrame columns "
            f"{list(df.columns)}"
        )

    out = df.copy()
    prepared = out[text_column].map(
        lambda value: preprocess(
            value, remove_stopwords=remove_stopwords, stopwords=stopwords
        )
    )
    out[cleaned_column] = [p.cleaned for p in prepared]
    out[tokens_column] = [p.tokens for p in prepared]
    out[num_tokens_column] = [p.num_tokens for p in prepared]
    out[empty_column] = [p.is_empty for p in prepared]
    return out


def preprocess_splits(
    splits: DatasetSplits,
    *,
    remove_stopwords: bool = False,
    stopwords: Optional[Iterable[str]] = None,
    **column_kwargs,
) -> Dict[str, pd.DataFrame]:
    """Preprocess train, validation, and test splits; labels preserved."""
    return {
        "train": add_preprocessed_columns(
            splits.train,
            remove_stopwords=remove_stopwords,
            stopwords=stopwords,
            **column_kwargs,
        ),
        "validation": add_preprocessed_columns(
            splits.validation,
            remove_stopwords=remove_stopwords,
            stopwords=stopwords,
            **column_kwargs,
        ),
        "test": add_preprocessed_columns(
            splits.test,
            remove_stopwords=remove_stopwords,
            stopwords=stopwords,
            **column_kwargs,
        ),
    }


# ---------------------------------------------------------------------------
# Invalid-row removal (null / whitespace-only text or labels)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitCleaningStats:
    """Bookkeeping for rows removed from one split during cleaning."""

    name: str
    rows_before: int
    missing_text: int          # text column is NaN / None
    whitespace_text: int       # text present but blank after stripping
    missing_label: int         # label column is NaN / None
    whitespace_label: int      # label present but blank after stripping
    rows_removed: int          # unique rows dropped (union of the above)
    rows_after: int

    @property
    def kept_fraction(self) -> float:
        if self.rows_before == 0:
            return 1.0
        return self.rows_after / self.rows_before


def _blank_mask(series: pd.Series) -> pd.Series:
    """True for NaN or whitespace-only values (non-strings become "")."""
    return series.fillna("").astype(str).str.strip().eq("")


def drop_invalid_rows(
    df: pd.DataFrame,
    *,
    text_column: str = config.TEXT_COLUMN,
    label_column: str = config.LABEL_COLUMN,
    name: str = "split",
) -> Tuple[pd.DataFrame, SplitCleaningStats]:
    """Return ``df`` with invalid rows removed, plus per-reason removal stats.

    A row is invalid when ``text`` or ``labels`` is missing (NaN/None) or
    whitespace-only. The input frame is not modified; row order is preserved
    and the index is reset. Source CSVs are never touched.
    """
    if text_column not in df.columns:
        raise ValueError(f"text_column {text_column!r} not found in {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"label_column {label_column!r} not found in {list(df.columns)}")

    text_blank = _blank_mask(df[text_column])
    label_blank = _blank_mask(df[label_column])

    missing_text = int(df[text_column].isna().sum())
    whitespace_text = int((~df[text_column].isna() & text_blank).sum())
    missing_label = int(df[label_column].isna().sum())
    whitespace_label = int((~df[label_column].isna() & label_blank).sum())

    invalid = text_blank | label_blank
    clean = df.loc[~invalid].copy().reset_index(drop=True)

    stats = SplitCleaningStats(
        name=name,
        rows_before=len(df),
        missing_text=missing_text,
        whitespace_text=whitespace_text,
        missing_label=missing_label,
        whitespace_label=whitespace_label,
        rows_removed=int(invalid.sum()),
        rows_after=len(clean),
    )
    return clean, stats


def clean_splits(
    splits: DatasetSplits,
) -> Tuple[DatasetSplits, Dict[str, SplitCleaningStats]]:
    """Remove invalid rows from every split independently.

    Train, validation, and test are cleaned separately; splits are never
    combined or repartitioned. Original text and labels of kept rows are
    preserved untouched.
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    stats: Dict[str, SplitCleaningStats] = {}
    for name, df in [
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ]:
        clean_df, split_stats = drop_invalid_rows(
            df,
            text_column=config.TEXT_COLUMN,
            label_column=config.LABEL_COLUMN,
            name=name,
        )
        cleaned[name] = clean_df
        stats[name] = split_stats
    return (
        DatasetSplits(
            train=cleaned["train"],
            validation=cleaned["validation"],
            test=cleaned["test"],
        ),
        stats,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _token_statistics(tokens_column: pd.Series) -> Dict[str, float]:
    counts = [len(tokens) for tokens in tokens_column]
    if not counts:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    arr = np.asarray(counts, dtype=float)
    return {
        "count": int(len(counts)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def _gujarati_char_overlap(original: str, cleaned: str) -> Tuple[set, set]:
    orig_gujarati = {c for c in original if GUJARATI_LO <= ord(c) <= GUJARATI_HI}
    cleaned_gujarati = {c for c in cleaned if GUJARATI_LO <= ord(c) <= GUJARATI_HI}
    return orig_gujarati, cleaned_gujarati


def _gujarati_char_count(series: pd.Series) -> int:
    """Total number of Gujarati Unicode chars (U+0A80..U+0AFF) in a column."""
    return int(
        series.fillna("").astype(str).map(
            lambda t: sum(1 for c in t if GUJARATI_LO <= ord(c) <= GUJARATI_HI)
        ).sum()
    )


def build_preprocessing_report(
    splits: DatasetSplits,
    preprocessed: Dict[str, pd.DataFrame],
    *,
    remove_stopwords: bool = False,
    cleaning_stats: Optional[Dict[str, SplitCleaningStats]] = None,
    raw_gujarati_char_counts: Optional[Dict[str, int]] = None,
) -> str:
    """Build a full preprocessing analysis report string (UTF-8 safe).

    ``splits`` are the cleaned splits (invalid rows removed, original text and
    labels preserved). ``preprocessed`` holds the cleaned/tokenized frames.
    """
    cleaning_stats = cleaning_stats or {}
    raw_gujarati_char_counts = raw_gujarati_char_counts or {}
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GUJARATI SHC PREPROCESSING ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. PREPROCESSING CONFIGURATION")
    lines.append("-" * 72)
    lines.append(f"  Text column:    {config.TEXT_COLUMN!r}")
    lines.append(f"  Label column:   {config.LABEL_COLUMN!r}")
    lines.append(f"  Encoding:       utf-8-sig (train/validation/test)")
    lines.append(f"  Stopword removal: {remove_stopwords}")
    if remove_stopwords:
        lines.append(f"  Stopword list: default Gujarati list ({len(DEFAULT_GUJARATI_STOPWORDS)} tokens)")
    lines.append("")
    lines.append("2. DATASET DIMENSIONS AND ROW COUNTS (before -> after cleaning)")
    lines.append("-" * 72)
    for name, df in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]:
        st = cleaning_stats.get(name)
        before = st.rows_before if st else len(df)
        after = st.rows_after if st else len(df)
        lines.append(f"  {name:12s}: rows_before={before:>7}  rows_after={after:>7}")
    lines.append("")

    lines.append("3. INVALID ROW REMOVAL (missing / whitespace-only text or labels)")
    lines.append("-" * 72)
    for name, st in cleaning_stats.items():
        empty_clean = int(preprocessed[name]["text_is_empty"].sum())
        lines.append(
            f"  {name:12s}: rows_before={st.rows_before:>7}  "
            f"missing_text={st.missing_text:>6}  whitespace_text={st.whitespace_text:>6}  "
            f"missing_label={st.missing_label:>6}  whitespace_label={st.whitespace_label:>6}  "
            f"rows_removed={st.rows_removed:>7}  rows_after={st.rows_after:>7}  "
            f"kept={st.kept_fraction*100:.2f}%  empty_after_clean={empty_clean}"
        )
    lines.append("")

    lines.append("4. LABEL DISTRIBUTION AFTER CLEANING (per split)")
    lines.append("-" * 72)
    for name, df in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]:
        lines.append(f"  --- {name} (rows={len(df)}) ---")
        vc = df[config.LABEL_COLUMN].value_counts(dropna=False)
        for label, count in vc.items():
            lines.append(f"    {label!r}: {int(count)}")
        lines.append(f"    distinct labels: {int(df[config.LABEL_COLUMN].nunique(dropna=False))}")
    lines.append("")

    lines.append("5. TOKEN COUNT STATISTICS (per split, after cleaning)")
    lines.append("-" * 72)
    for name, df in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]:
        stats = _token_statistics(preprocessed[name]["tokens"])
        lines.append(
            f"  {name:12s}: examples={stats['count']:>7}  "
            f"mean={stats['mean']:.2f}  median={stats['median']:.1f}  "
            f"min={stats['min']}  max={stats['max']}"
        )
    lines.append("")

    lines.append("6. GUJARATI CHARACTER COUNTS AND UNICODE INTEGRITY")
    lines.append("-" * 72)
    for name, df in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]:
        raw_gujarati = raw_gujarati_char_counts.get(name, 0)
        kept_gujarati = _gujarati_char_count(df[config.TEXT_COLUMN])
        lost_total = 0
        checked = 0
        gujarati_rows = 0
        for orig, cleaned in zip(
            df[config.TEXT_COLUMN].fillna("").astype(str),
            preprocessed[name]["text_clean"],
        ):
            orig_gujarati_set, cleaned_gujarati_set = _gujarati_char_overlap(orig, cleaned)
            if orig_gujarati_set:
                gujarati_rows += 1
                checked += 1
                lost_total += len(orig_gujarati_set - cleaned_gujarati_set)
        lines.append(
            f"  {name:12s}: Gujarati-bearing rows={gujarati_rows:>7}  "
            f"Gujarati chars raw={raw_gujarati:>9}  kept={kept_gujarati:>9}  "
            f"lost in cleaning={lost_total}"
        )
    lines.append("")

    lines.append("7. BEFORE-AFTER EXAMPLES (first 3 non-empty rows per split)")
    lines.append("-" * 72)
    for name, df in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]:
        pdf = preprocessed[name]
        shown = 0
        lines.append(f"  --- {name} ---")
        for _, row in pdf.iterrows():
            if shown >= 3:
                break
            if row["text_is_empty"]:
                continue
            lines.append(f"    original : {row[config.TEXT_COLUMN]!r}")
            lines.append(f"    cleaned  : {row['text_clean']!r}")
            lines.append(f"    tokens   : {row['tokens']}")
            lines.append("")
            shown += 1

    if remove_stopwords:
        lines.append("7. OPTIONAL STOPWORD REMOVAL COMPARISON (first 3 rows per split)")
        lines.append("-" * 72)
        for name, df in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]:
            pdf = preprocessed[name]
            shown = 0
            lines.append(f"  --- {name} ---")
            for _, row in pdf.iterrows():
                if shown >= 3:
                    break
                base = preprocess(row[config.TEXT_COLUMN], remove_stopwords=False)
                lines.append(
                    f"    without stopwords: {base.tokens}"
                )
                lines.append(
                    f"    with stopwords   : {row['tokens']}"
                )
                lines.append("")
                shown += 1
    else:
        lines.append("7. OPTIONAL STOPWORD REMOVAL COMPARISON")
        lines.append("-" * 72)
        lines.append("  Disabled by default (enable with remove_stopwords=True).")
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)

    return "\n".join(lines)


def write_report(report: str, filename: str = "preprocessing_report.txt") -> Path:
    """Write the report to ``reports/`` as UTF-8 and return the report path."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / filename
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_preprocessing_pipeline(
    *,
    remove_stopwords: bool = False,
    stopwords: Optional[Iterable[str]] = None,
    save_processed: bool = False,
    processed_dir: Optional[Path] = None,
    report_filename: str = "preprocessing_report.txt",
) -> Tuple[str, Path, Dict[str, pd.DataFrame]]:
    """Run the full Phase 2 preprocessing pipeline on the real SHC splits.

    Loads the configured splits (originals never modified), applies
    preprocessing to the verified SHC text column, writes the analysis report,
    and optionally saves the processed splits under ``data/processed/``.

    Returns (report, report_path, preprocessed_splits).
    """
    config.validate_paths()
    raw_splits = load_all_splits()
    validate_all(raw_splits)

    # Gujarati character totals in the raw text (before invalid-row removal).
    raw_gujarati_char_counts = {
        name: _gujarati_char_count(df[config.TEXT_COLUMN])
        for name, df in [
            ("train", raw_splits.train),
            ("validation", raw_splits.validation),
            ("test", raw_splits.test),
        ]
    }

    # Remove invalid rows per split; splits stay separate.
    splits, cleaning_stats = clean_splits(raw_splits)

    preprocessed = preprocess_splits(
        splits, remove_stopwords=remove_stopwords, stopwords=stopwords
    )

    report = build_preprocessing_report(
        splits,
        preprocessed,
        remove_stopwords=remove_stopwords,
        cleaning_stats=cleaning_stats,
        raw_gujarati_char_counts=raw_gujarati_char_counts,
    )
    report_path = write_report(report, filename=report_filename)

    if save_processed:
        out_dir = processed_dir or config.PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_map = {
            "train": config.TRAIN_PATH.stem,
            "validation": config.VALID_PATH.stem,
            "test": config.TEST_PATH.stem,
        }
        for name, df in preprocessed.items():
            save_df = df.copy()
            save_df["tokens"] = save_df["tokens"].map(lambda t: " ".join(t))
            save_df.to_csv(
                out_dir / f"{stem_map[name]}.preprocessed.csv",
                index=False,
                encoding="utf-8",
            )

    return report, report_path, preprocessed
