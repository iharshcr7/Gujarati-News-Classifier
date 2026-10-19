"""Unit tests for src/preprocessing.py (Gujarati text preprocessing).

Written first (TDD): each test names the behavior it expects from the not-yet
implemented `src.preprocessing` module. Run with:

    See the project README or run: python -m pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import math
from typing import List

import pandas as pd
import pytest

import hashlib

import config
from src.data_loader import DatasetSplits, load_all_splits
from src.preprocessing import (
    DEFAULT_GUJARATI_STOPWORDS,
    GUJARATI_HI,
    GUJARATI_LO,
    PreprocessedText,
    add_preprocessed_columns,
    clean_splits,
    clean_text,
    drop_invalid_rows,
    is_valid_text,
    normalize_whitespace,
    preprocess,
    preprocess_splits,
    remove_stopwords,
    run_preprocessing_pipeline,
    tokenize,
)


# ---------------------------------------------------------------------------
# Gujarati text preservation
# ---------------------------------------------------------------------------

def test_clean_preserves_gujarati_headline():
    assert clean_text("ગુજરાતી સમાચાર") == "ગુજરાતી સમાચાર"


def test_clean_preserves_mixed_gujarati_and_english():
    assert clean_text("UPI-PayNow: ભારત અને સિંગાપોર") == "UPI PayNow ભારત અને સિંગાપોર"


# ---------------------------------------------------------------------------
# Gujarati vowel signs and diacritics (must survive untouched)
# ---------------------------------------------------------------------------

def test_clean_preserves_vowel_signs():
    # કી = ક + ી (U+0AC0), કુ = ક + ુ (U+0AC1), કૌ = ક + ૌ (U+0AC4)
    assert clean_text("કી કુ કૌ") == "કી કુ કૌ"


def test_vowel_signs_survive_full_pipeline():
    result = preprocess("કી કુ કૌ સમાચાર")
    assert result.cleaned == "કી કુ કૌ સમાચાર"
    assert result.tokens == ["કી", "કુ", "કૌ", "સમાચાર"]


def test_gujarati_unicode_range_constants():
    assert GUJARATI_LO == 0x0A80
    assert GUJARATI_HI == 0x0AFF


def test_cleaning_keeps_every_original_gujarati_character():
    text = "અરએક ટુકો! https://example.com નવીનીતિ"
    result = preprocess(text)
    original_gujarati = {c for c in text if GUJARATI_LO <= ord(c) <= GUJARATI_HI}
    cleaned_gujarati = {c for c in result.cleaned if GUJARATI_LO <= ord(c) <= GUJARATI_HI}
    assert original_gujarati <= cleaned_gujarati, f"lost: {original_gujarati - cleaned_gujarati}"


# ---------------------------------------------------------------------------
# URLs, HTML, emails, punctuation
# ---------------------------------------------------------------------------

def test_clean_removes_http_urls():
    assert clean_text("આ વેબસાઇટ https://example.com/path છે") == "આ વેબસાઇટ છે"
    assert clean_text("http://x.in આ") == "આ"


def test_clean_removes_www_urls():
    assert clean_text("જુઓ www.abc.in અને આ") == "જુઓ અને આ"


def test_clean_removes_html_tags():
    assert clean_text("<b>ગુજરાતી</b> સમાચાર") == "ગુજરાતી સમાચાર"
    assert clean_text('<div class="x">તે</div>') == "તે"


def test_clean_removes_email_addresses():
    assert clean_text("સંપર્ક abc@example.com કરો") == "સંપર્ક કરો"


def test_clean_removes_punctuation_without_joining_words():
    # Commas/periods/exclamation become spaces; adjacent words must stay separate.
    assert clean_text("આ,તે. જ!") == "આ તે જ"
    assert clean_text("અને...પણ") == "અને પણ"


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

def test_normalize_whitespace_collapses_tabs_and_newlines():
    assert normalize_whitespace("ગુજરાતી    સમાચાર\n\tઆ\nહવે") == "ગુજરાતી સમાચાર આ હવે"


def test_clean_normalizes_whitespace():
    assert clean_text("ગુજરાતી    સમાચાર") == "ગુજરાતી સમાચાર"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def test_tokenize_splits_gujarati_sentence():
    assert tokenize("ગુજરાતી સમાચાર વાંચો") == ["ગુજરાતી", "સમાચાર", "વાંચો"]


def test_tokenize_returns_list_of_strings():
    tokens = tokenize("આ એક પરીક્ષા છે")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)


def test_tokenize_empty_and_blank():
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("\n\t") == []


def test_tokenize_keeps_english_and_digits():
    # Tokenization splits on whitespace only; hyphenated words stay joined.
    assert tokenize("UPI-PayNow 2024 ભારત") == ["UPI-PayNow", "2024", "ભારત"]


# ---------------------------------------------------------------------------
# Missing values, None, non-string, empty strings
# ---------------------------------------------------------------------------

def test_clean_none_returns_empty():
    assert clean_text(None) == ""


def test_clean_nan_returns_empty():
    assert clean_text(float("nan")) == ""


def test_clean_pd_na_returns_empty():
    assert clean_text(pd.NA) == ""


def test_clean_non_string_returns_empty():
    assert clean_text(123) == ""
    assert clean_text(["ગુજરાતી"]) == ""


def test_clean_empty_string_returns_empty():
    assert clean_text("") == ""


def test_is_valid_text_false_for_invalid():
    assert is_valid_text(None) is False
    assert is_valid_text(float("nan")) is False
    assert is_valid_text("") is False
    assert is_valid_text("   ") is False
    assert is_valid_text(123) is False


def test_is_valid_text_true_for_gujarati():
    assert is_valid_text("ગુજરાતી") is True
    assert is_valid_text("  ગુજરાતી  ") is True


def test_preprocess_on_invalid_yields_empty_result():
    result = preprocess(None)
    assert isinstance(result, PreprocessedText)
    assert result.original == ""
    assert result.cleaned == ""
    assert result.tokens == []
    assert result.num_tokens == 0
    assert result.is_empty is True


# ---------------------------------------------------------------------------
# Optional stopword removal (disabled by default)
# ---------------------------------------------------------------------------

def test_default_stopword_set_is_documented_and_non_empty():
    assert isinstance(DEFAULT_GUJARATI_STOPWORDS, frozenset)
    assert len(DEFAULT_GUJARATI_STOPWORDS) > 0
    assert "અને" in DEFAULT_GUJARATI_STOPWORDS
    assert "તે" in DEFAULT_GUJARATI_STOPWORDS


def test_stopword_removal_disabled_by_default():
    stopwords = {"અને", "તે", "સમાચાર"}
    result = preprocess("અને તે સમાચાર", stopwords=stopwords)
    assert result.tokens == ["અને", "તે", "સમાચાર"]


def test_stopword_removal_when_enabled():
    stopwords = {"અને", "તે"}
    result = preprocess("અને તે સમાચાર", remove_stopwords=True, stopwords=stopwords)
    assert result.tokens == ["સમાચાર"]
    assert result.num_tokens == 1


def test_remove_stopwords_function_direct():
    assert remove_stopwords(["અને", "તે", "સમાચાર"], stopwords={"અને", "તે"}) == ["સમાચાર"]
    assert remove_stopwords(["અને", "તે"], stopwords=set()) == ["અને", "તે"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_preprocess_is_deterministic_across_repeated_calls():
    text = "ગુજરાતી https://example.com સમાચાર,!  આ   તે"
    first = preprocess(text)
    second = preprocess(text)
    assert first == second


def test_clean_text_is_deterministic():
    assert clean_text("આ, તે! https://x.in") == clean_text("આ, તે! https://x.in")


# ---------------------------------------------------------------------------
# DataFrame wrapper: original text and labels preserved
# ---------------------------------------------------------------------------

def test_add_preprocessed_columns_preserves_original_text_and_labels():
    df = pd.DataFrame(
        {"text": ["ગુજરાતી સમાચાર", "hello world"], "labels": ["State", "Business"]}
    )
    out = add_preprocessed_columns(df)
    assert list(out.columns) == [
        "text",
        "labels",
        "text_clean",
        "tokens",
        "num_tokens",
        "text_is_empty",
    ]
    assert list(out["text"]) == ["ગુજરાતી સમાચાર", "hello world"]
    assert list(out["labels"]) == ["State", "Business"]
    assert list(out["text_clean"]) == ["ગુજરાતી સમાચાર", "hello world"]
    assert list(out["num_tokens"]) == [2, 2]
    assert list(out["text_is_empty"]) == [False, False]
    # input frame not modified
    assert "text_clean" not in df.columns


def test_add_preprocessed_columns_handles_missing_text():
    df = pd.DataFrame({"text": ["ગુજરાતી", None, ""], "labels": ["State", "Business", "Crime"]})
    out = add_preprocessed_columns(df)
    assert list(out["num_tokens"]) == [1, 0, 0]
    assert list(out["text_is_empty"]) == [False, True, True]


def test_preprocess_splits_keeps_labels_and_counts():
    train = pd.DataFrame({"text": ["આ એક પરીક્ષા છે", "બીજી સમાચાર"], "labels": ["State", "Business"]})
    valid = pd.DataFrame({"text": ["ત્રીજી"], "labels": ["Crime"]})
    test = pd.DataFrame({"text": ["ચોથી"], "labels": ["Auto"]})
    splits = DatasetSplits(train=train, validation=valid, test=test)

    from src.preprocessing import preprocess_splits

    out = preprocess_splits(splits)
    assert set(out.keys()) == {"train", "validation", "test"}
    assert list(out["train"]["labels"]) == ["State", "Business"]
    assert list(out["validation"]["num_tokens"]) == [1]
    assert list(out["test"]["text_clean"]) == ["ચોથી"]


# ---------------------------------------------------------------------------
# Compatibility with the actual SHC dataset
# ---------------------------------------------------------------------------

def test_preprocess_real_shc_sample():
    splits = load_all_splits()
    sample = splits.train.head(100)
    out = add_preprocessed_columns(sample)
    assert len(out) == 100
    assert list(out["labels"]) == list(sample["labels"])
    assert list(out["text"]) == list(sample["text"])
    assert (out["num_tokens"] >= 0).all()
    # Gujarati coverage: the recovered train file contains real Gujarati text.
    train_guj_rows = int(
        out["text"].map(lambda t: any(GUJARATI_LO <= ord(c) <= GUJARATI_HI for c in str(t))).sum()
    )
    valid_sample = splits.validation.head(100)
    valid_out = add_preprocessed_columns(valid_sample)
    valid_guj_rows = int(
        valid_out["text"].map(lambda t: any(GUJARATI_LO <= ord(c) <= GUJARATI_HI for c in str(t))).sum()
    )
    assert train_guj_rows > 0
    assert valid_guj_rows > 0


# ---------------------------------------------------------------------------
# UTF-8 Gujarati reading and preservation (recovered dataset)
# ---------------------------------------------------------------------------

def test_real_dataset_read_utf8_gujarati_preserved():
    splits = load_all_splits()
    for name in ["train", "validation", "test"]:
        df = getattr(splits, name)
        mask = df[config.TEXT_COLUMN].map(
            lambda t: isinstance(t, str)
            and any(GUJARATI_LO <= ord(c) <= GUJARATI_HI for c in t)
        )
        assert mask.sum() > 0, f"split {name} has no Gujarati text"
    sample = splits.validation.loc[
        splits.validation[config.TEXT_COLUMN].map(
            lambda t: isinstance(t, str) and "ગુજરાતી" in t
        ),
        config.TEXT_COLUMN,
    ].iloc[0]
    original_gujarati = {c for c in sample if GUJARATI_LO <= ord(c) <= GUJARATI_HI}
    cleaned_gujarati = {c for c in clean_text(sample) if GUJARATI_LO <= ord(c) <= GUJARATI_HI}
    assert original_gujarati <= cleaned_gujarati, f"lost: {original_gujarati - cleaned_gujarati}"


# ---------------------------------------------------------------------------
# Missing and whitespace-only text / labels
# ---------------------------------------------------------------------------

def test_drop_invalid_rows_removes_missing_and_blank_text_and_labels():
    df = pd.DataFrame(
        {
            "text": ["ગુજરાતી સમાચાર", None, "   ", "valid text", "also valid"],
            "labels": ["State", "Business", "Crime", None, "   "],
        }
    )
    clean, stats = drop_invalid_rows(df)
    # Only row 0 has both non-blank text and non-blank label.
    assert len(clean) == 1
    assert list(clean["text"]) == ["ગુજરાતી સમાચાર"]
    assert list(clean["labels"]) == ["State"]
    assert stats.rows_before == 5
    assert stats.missing_text == 1   # row 1
    assert stats.whitespace_text == 1  # row 2
    assert stats.missing_label == 1   # row 3
    assert stats.whitespace_label == 1  # row 4
    assert stats.rows_removed == 4  # union, no overlap here
    assert stats.rows_after == 1
    # input frame untouched
    assert len(df) == 5


def test_drop_invalid_rows_input_not_modified():
    df = pd.DataFrame({"text": ["ગુજરાતી", None], "labels": ["State", "Business"]})
    before = df.copy()
    drop_invalid_rows(df)
    pd.testing.assert_frame_equal(df, before)


def test_drop_invalid_rows_raises_on_missing_column():
    df = pd.DataFrame({"text": ["a"], "label": ["State"]})
    with pytest.raises(ValueError, match="label_column"):
        drop_invalid_rows(df)
    df2 = pd.DataFrame({"text": ["a"], "labels": ["State"]})
    with pytest.raises(ValueError, match="text_column"):
        drop_invalid_rows(df2.rename(columns={"text": "txt"}))


def test_clean_splits_keeps_splits_separate():
    train = pd.DataFrame({"text": ["ગુજરાતી", None], "labels": ["State", "State"]})
    valid = pd.DataFrame({"text": [" valid "], "labels": ["Business"]})
    test = pd.DataFrame({"text": ["x"], "labels": [None]})
    splits = DatasetSplits(train=train, validation=valid, test=test)
    cleaned, stats = clean_splits(splits)
    assert len(cleaned.train) == 1
    assert len(cleaned.validation) == 1
    assert len(cleaned.test) == 0
    assert cleaned.train.iloc[0]["text"] == "ગુજરાતી"
    assert set(stats) == {"train", "validation", "test"}
    assert stats["train"].rows_removed == 1
    assert stats["validation"].rows_removed == 0
    assert stats["test"].rows_removed == 1


# ---------------------------------------------------------------------------
# Gujarati tokenization and punctuation handling
# ---------------------------------------------------------------------------

def test_tokenize_gujarati_with_punctuation_and_digits():
    # tokenize splits on whitespace only; punctuation is handled by clean_text.
    assert tokenize(clean_text("ગુજરાતી સમાચાર, હવે! 2024.")) == [
        "ગુજરાતી",
        "સમાચાર",
        "હવે",
        "2024",
    ]
    assert tokenize("ગુજરાતી સમાચાર, હવે! 2024.") == [
        "ગુજરાતી",
        "સમાચાર,",
        "હવે!",
        "2024.",
    ]


def test_clean_text_mixed_english_gujarati_punctuation():
    assert clean_text("UPI-PayNow: ભારત અને સિંગાપોર વચ્ચે કરાર થયો.") == (
        "UPI PayNow ભારત અને સિંગાપોર વચ્ચે કરાર થયો"
    )
    # Negation and meaningful words are kept, not stripped.
    assert clean_text("પૂજા-પાઠ કરવા નહીં") == "પૂજા પાઠ કરવા નહીં"


# ---------------------------------------------------------------------------
# Empty-token-list diagnostics
# ---------------------------------------------------------------------------

def test_preprocess_no_tokens_yields_empty_diagnostics():
    # Digits survive cleaning; only pure punctuation/symbols yield no tokens.
    result = preprocess("!@# ,,, ...")
    assert result.tokens == []
    assert result.num_tokens == 0
    assert result.is_empty is True
    assert result.cleaned == ""


def test_preprocess_on_invalid_yields_empty_diagnostics():
    result = preprocess(None)
    assert result.tokens == []
    assert result.num_tokens == 0
    assert result.is_empty is True
    result = preprocess(float("nan"))
    assert result.is_empty is True


# ---------------------------------------------------------------------------
# Separate train / validation / test outputs
# ---------------------------------------------------------------------------

def test_preprocess_splits_output_keys_and_labels():
    train = pd.DataFrame({"text": ["આ એક પરીક્ષા છે"], "labels": ["State"]})
    valid = pd.DataFrame({"text": ["બીજી સમાચાર"], "labels": ["Business"]})
    test = pd.DataFrame({"text": ["ચોથી"], "labels": ["Crime"]})
    splits = DatasetSplits(train=train, validation=valid, test=test)
    out = preprocess_splits(splits)
    assert set(out) == {"train", "validation", "test"}
    assert list(out["train"]["labels"]) == ["State"]
    assert list(out["validation"]["text_clean"]) == ["બીજી સમાચાર"]
    assert list(out["test"]["tokens"]) == [["ચોથી"]]


# ---------------------------------------------------------------------------
# Source CSV files must never be modified
# ---------------------------------------------------------------------------

def test_pipeline_run_does_not_modify_source_csvs(tmp_path, monkeypatch):
    raw_paths = [config.TRAIN_PATH, config.VALID_PATH, config.TEST_PATH]
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths
    }
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    report, report_path, out = run_preprocessing_pipeline(save_processed=True)
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths
    }
    assert before == after, "source CSV files were modified"
    assert set(out) == {"train", "validation", "test"}
    assert "INVALID ROW REMOVAL" in report
    assert "LABEL DISTRIBUTION AFTER CLEANING" in report
    assert "GUJARATI CHARACTER COUNTS" in report
    processed_dir = tmp_path / "processed"
    for name, stem in [
        ("train", config.TRAIN_PATH.stem),
        ("validation", config.VALID_PATH.stem),
        ("test", config.TEST_PATH.stem),
    ]:
        out_file = processed_dir / f"{stem}.preprocessed.csv"
        assert out_file.exists(), f"missing processed output: {out_file}"
        saved = pd.read_csv(out_file, encoding="utf-8")
        assert list(saved.columns) == [
            "text", "labels", "text_clean", "tokens", "num_tokens", "text_is_empty"
        ]
        assert len(saved) == len(out[name])
