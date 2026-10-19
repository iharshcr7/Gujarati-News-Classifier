"""Unit tests for src/word2vec_trainer.py (Phase 3: Word2Vec CBOW/Skip-gram).

Fast tests use small synthetic Gujarati corpora so the suite stays quick. The
actual project models are trained separately via train_word2vec.py on the
configured SHC training split.

Run with:

    python -m pytest tests/test_word2vec_trainer.py -v
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pytest

import config
from src.data_loader import DatasetSplits
from src.preprocessing import clean_splits, has_gujarati, preprocess
from src.word2vec_trainer import (
    DIAGNOSTIC_EXAMPLE_WORDS,
    DIAGNOSTIC_SAMPLE_HEADLINES,
    _gujarati_vocab_words,
    _headline_oov_stats,
    _is_loadable_word2vec,
    _nearest_neighbors,
    _pick_example_words,
    _top_gujarati_tokens,
    count_discarded_tokens,
    extract_train_sentences,
    save_model,
    train_word2vec,
    validate_training_readiness,
)

# Small synthetic Gujarati corpus used for fast unit tests.
SYNTHETIC_SENTENCES: List[List[str]] = [
    ["ગુજરાતી", "સમાચાર", "આજ"],
    ["ગુજરાતી", "સમાચાર", "કાલ"],
    ["આજ", "ગુજરાતી", "સમાચાર", "વાંચો"],
    ["કાલ", "સમાચાર", "બ્યુઝનેસ"],
    ["ગુજરાતી", "ક્રિકેટ", "મેચ"],
    ["ક્રિકેટ", "મેચ", "આજ"],
    ["ગુજરાતી", "સમાચાર"],  # repeated to raise frequency above min_count
    ["ગુજરાતી", "સમાચાર"],
    [],  # empty token list (must be handled gracefully)
]


def test_train_cbow_sg_zero():
    model, elapsed = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=0,
        vector_size=16,
        window=3,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    assert model.sg == 0
    assert model.vector_size == 16
    assert "ગુજરાતી" in model.wv
    assert elapsed >= 0


def test_train_skipgram_sg_one():
    model, elapsed = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        window=3,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    assert model.sg == 1
    assert model.vector_size == 16
    assert "ગુજરાતી" in model.wv


def test_invalid_sg_raises():
    with pytest.raises(ValueError, match="sg must be"):
        train_word2vec(
            SYNTHETIC_SENTENCES,
            sg=2,
            vector_size=16,
            min_count=1,
            workers=1,
            seed=42,
            epochs=1,
        )


def test_gujarati_vocab_lookup():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    assert "ગુજરાતી" in model.wv
    assert "સમાચાર" in model.wv
    assert "ક્રિકેટ" in model.wv


def test_vector_dimensionality():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=0,
        vector_size=32,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    vec = model.wv["ગુજરાતી"]
    assert len(vec) == 32
    assert model.vector_size == 32


def test_similarity_query_for_known_word():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=5,
    )
    neighbors = model.wv.most_similar("ગુજરાતી", topn=3)
    assert len(neighbors) == 3
    for _, sim in neighbors:
        assert -1.0 <= sim <= 1.0
        assert not math.isnan(sim)


def test_out_of_vocabulary_word_graceful():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    assert "અમારતુરુંઅમુકશબ્દ" not in model.wv
    assert _nearest_neighbors(model, "અમારતુરુંઅમુકશબ્દ", topn=5) == []


def test_empty_token_list_handled_gracefully():
    # Corpus that contains only empty token lists should still train without
    # raising (Gensim skips empty sentences; vocab may be empty).
    model, _ = train_word2vec(
        [[], [], []],
        sg=0,
        vector_size=8,
        min_count=1,
        workers=1,
        seed=42,
        epochs=2,
    )
    assert model.vector_size == 8


def test_save_and_reload_model(tmp_path):
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    path = tmp_path / "test_model.model"
    save_model(model, path)
    assert path.exists()

    reloaded = __import__("gensim.models", fromlist=["Word2Vec"]).Word2Vec.load(
        str(path)
    )
    assert reloaded.sg == model.sg == 1
    assert reloaded.vector_size == model.vector_size == 16
    assert "ગુજરાતી" in reloaded.wv
    assert reloaded.wv["ગુજરાતી"].shape == model.wv["ગુજરાતી"].shape


def test_save_model_preserves_existing_loadable_artifact(tmp_path):
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=0,
        vector_size=8,
        min_count=1,
        workers=1,
        seed=42,
        epochs=2,
    )
    path = tmp_path / "keep.model"
    save_model(model, path)
    first_size = path.stat().st_size

    # Different model; saving without overwrite must not replace it.
    other, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=32,
        min_count=1,
        workers=1,
        seed=42,
        epochs=2,
    )
    save_model(other, path, overwrite=False)
    assert path.stat().st_size == first_size, "existing loadable artifact was overwritten"

    reloaded = __import__("gensim.models", fromlist=["Word2Vec"]).Word2Vec.load(
        str(path)
    )
    assert reloaded.sg == 0  # original CBOW model preserved


def test_save_model_replaces_incompatible_artifact(tmp_path):
    path = tmp_path / "bad.model"
    path.write_text("not a gensim model", encoding="utf-8")
    assert _is_loadable_word2vec(path) is False

    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=0,
        vector_size=8,
        min_count=1,
        workers=1,
        seed=42,
        epochs=2,
    )
    save_model(model, path, overwrite=False)
    assert _is_loadable_word2vec(path) is True


def test_count_discarded_tokens_below_min_count():
    # "આજ" appears 3 times, "ગુજરાતી" 4, "સમાચાર" 4, "કાલ" 2, rest once.
    # With min_count=2, singletons are discarded: બ્યુઝનેસ, વાંચો, મેચ(2 -> kept),
    # Let's just verify counting logic against min_count=2 vs min_count=100.
    discarded_min2 = count_discarded_tokens(SYNTHETIC_SENTENCES, min_count=2)
    discarded_min100 = count_discarded_tokens(SYNTHETIC_SENTENCES, min_count=100)
    assert discarded_min100 > discarded_min2
    assert discarded_min2 >= 0


def test_config_parameters_recorded():
    assert config.W2V_VECTOR_SIZE == 100
    assert config.W2V_WINDOW == 5
    assert config.W2V_MIN_COUNT == 2
    assert config.W2V_SEED == 42
    assert config.W2V_EPOCHS == 10
    assert config.W2V_SG_CBOW == 0
    assert config.W2V_SG_SKIPGRAM == 1
    assert isinstance(config.W2V_WORKERS, int) and config.W2V_WORKERS >= 1
    assert config.W2V_MODEL_CBOW_PATH.name == "word2vec_cbow.model"
    assert config.W2V_MODEL_SKIPGRAM_PATH.name == "word2vec_skipgram.model"
    assert config.W2V_MODEL_CBOW_PATH.parent == config.MODELS_DIR
    assert config.W2V_REPORT_FILENAME == "word2vec_training_report.txt"


def test_pick_example_words_prefers_gujarati():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    chosen = _pick_example_words(model, DIAGNOSTIC_EXAMPLE_WORDS)
    assert len(chosen) <= DIAGNOSTIC_EXAMPLE_WORDS
    assert all(w in model.wv for w in chosen)
    # Gujarati words should be preferred when present.
    assert "ગુજરાતી" in chosen


def test_headline_oov_stats_empty_and_fully_oov():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    empty, fully_oov = _headline_oov_stats(
        model, SYNTHETIC_SENTENCES, len(SYNTHETIC_SENTENCES)
    )
    # SYNTHETIC_SENTENCES[8] is empty -> empty >= 1
    assert empty >= 1
    assert empty + fully_oov <= 5


def test_extract_train_sentences_returns_token_lists(monkeypatch, tmp_path):
    # Use tiny fixtures so we don't depend on real dataset row counts.
    train_csv = "text,labels\nગુજરાતી સમાચાર,State\nhello world,Business\n"
    valid_csv = "text,labels\nવValidation,State\n"
    test_csv = "text,labels\nવValidation,State\n"
    train_p = tmp_path / "train.csv"
    valid_p = tmp_path / "valid.csv"
    test_p = tmp_path / "test.csv"
    train_p.write_text(train_csv, encoding="utf-8")
    valid_p.write_text(valid_csv, encoding="utf-8")
    test_p.write_text(test_csv, encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_PATH", train_p)
    monkeypatch.setattr(config, "VALID_PATH", valid_p)
    monkeypatch.setattr(config, "TEST_PATH", test_p)
    monkeypatch.setattr(config, "TRAIN_HEADER", 0)
    monkeypatch.setattr(config, "VALID_HEADER", 0)
    monkeypatch.setattr(config, "TEST_HEADER", 0)
    monkeypatch.setattr(config, "TRAIN_ENCODING", "utf-8")
    monkeypatch.setattr(config, "VALID_ENCODING", "utf-8")
    monkeypatch.setattr(config, "TEST_ENCODING", "utf-8")
    monkeypatch.setattr(config, "TRAIN_ENCODING_ERRORS", None)

    sentences = extract_train_sentences()
    assert isinstance(sentences, list)
    assert len(sentences) == 2
    assert all(isinstance(t, list) for t in sentences)
    assert all(isinstance(tok, str) for sent in sentences for tok in sent)
    # "ગુજરાતી" and "સમાચાર" preserved through preprocessing tokenization.
    assert "ગુજરાતી" in sentences[0]


def test_preprocess_integration_for_word2vec():
    result = preprocess("ગુજરાતી https://example.com સમાચાર,!  આ   તે")
    assert result.tokens == ["ગુજરાતી", "સમાચાર", "આ", "તે"]
    assert result.num_tokens == 4
    assert result.is_empty is False


def test_preprocess_on_invalid_yields_empty_tokens():
    result = preprocess(None)
    assert result.tokens == []
    assert result.is_empty is True


# ---------------------------------------------------------------------------
# Gujarati Unicode preservation
# ---------------------------------------------------------------------------

def test_trained_model_keeps_gujarati_vocab():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    gujarati = [w for w in model.wv.index_to_key if has_gujarati(w)]
    assert "ગુજરાતી" in gujarati
    assert "સમાચાર" in gujarati
    assert len(gujarati) >= 2


def test_preprocessing_preserves_gujarati_through_pipeline():
    result = preprocess("ગુજરાતી સમાચાર વાંચો")
    assert result.tokens == ["ગુજરાતી", "સમાચાર", "વાંચો"]
    assert all(has_gujarati(t) for t in result.tokens)
    assert not result.is_empty


# ---------------------------------------------------------------------------
# Correct use of preprocessing.py
# ---------------------------------------------------------------------------

def test_extract_train_sentences_uses_preprocessing(monkeypatch, tmp_path):
    train_csv = "text,labels\nગુજરાતી સમાચાર,State\nhello world,Business\n"
    train_p = tmp_path / "train.csv"
    train_p.write_text(train_csv, encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_PATH", train_p)
    monkeypatch.setattr(config, "TRAIN_HEADER", 0)
    monkeypatch.setattr(config, "TRAIN_ENCODING", "utf-8")
    monkeypatch.setattr(config, "TRAIN_ENCODING_ERRORS", None)

    sentences = extract_train_sentences()
    expected = preprocess("ગુજરાતી સમાચાર").tokens
    assert sentences[0] == expected


# ---------------------------------------------------------------------------
# CBOW and Skip-gram configuration
# ---------------------------------------------------------------------------

def test_cbow_and_skipgram_use_config_values():
    for sg, expected in [
        (config.W2V_SG_CBOW, 0),
        (config.W2V_SG_SKIPGRAM, 1),
    ]:
        model, _ = train_word2vec(
            SYNTHETIC_SENTENCES,
            sg=sg,
            vector_size=config.W2V_VECTOR_SIZE,
            window=config.W2V_WINDOW,
            min_count=config.W2V_MIN_COUNT,
            workers=1,
            seed=config.W2V_SEED,
            epochs=config.W2V_EPOCHS,
        )
        assert model.sg == expected
        assert model.vector_size == config.W2V_VECTOR_SIZE
        assert model.window == config.W2V_WINDOW


# ---------------------------------------------------------------------------
# Training-only corpus usage (no validation/test leakage)
# ---------------------------------------------------------------------------

def test_extract_train_sentences_uses_train_only(monkeypatch, tmp_path):
    train_csv = "text,labels\nગુજરાતી સમાચાર,State\n"
    valid_csv = "text,labels\nVALIDATION_UNIQUE_TOKEN_XYZ,International\n"
    test_csv = "text,labels\nTEST_UNIQUE_TOKEN_XYZ,Crime\n"
    train_p = tmp_path / "train.csv"
    valid_p = tmp_path / "valid.csv"
    test_p = tmp_path / "test.csv"
    train_p.write_text(train_csv, encoding="utf-8")
    valid_p.write_text(valid_csv, encoding="utf-8")
    test_p.write_text(test_csv, encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_PATH", train_p)
    monkeypatch.setattr(config, "VALID_PATH", valid_p)
    monkeypatch.setattr(config, "TEST_PATH", test_p)
    monkeypatch.setattr(config, "TRAIN_HEADER", 0)
    monkeypatch.setattr(config, "VALID_HEADER", 0)
    monkeypatch.setattr(config, "TEST_HEADER", 0)
    monkeypatch.setattr(config, "TRAIN_ENCODING", "utf-8")
    monkeypatch.setattr(config, "VALID_ENCODING", "utf-8")
    monkeypatch.setattr(config, "TEST_ENCODING", "utf-8")
    monkeypatch.setattr(config, "TRAIN_ENCODING_ERRORS", None)

    sentences = extract_train_sentences()
    assert len(sentences) == 1
    all_tokens = [t for s in sentences for t in s]
    assert "VALIDATION_UNIQUE_TOKEN_XYZ" not in all_tokens
    assert "TEST_UNIQUE_TOKEN_XYZ" not in all_tokens
    assert "ગુજરાતી" in all_tokens
    assert "સમાચાર" in all_tokens


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def test_saved_model_reloads_with_gujarati(tmp_path):
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    path = tmp_path / "reload_test.model"
    save_model(model, path)
    reloaded = __import__("gensim.models", fromlist=["Word2Vec"]).Word2Vec.load(
        str(path)
    )
    assert reloaded.sg == 1
    assert "ગુજરાતી" in reloaded.wv
    assert reloaded.wv["ગુજરાતી"].shape == (16,)


# ---------------------------------------------------------------------------
# Vocabulary checks and diagnostics
# ---------------------------------------------------------------------------

def test_gujarati_vocab_checks_detect_presence():
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=1,
        vector_size=16,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    gujarati_words = _gujarati_vocab_words(model)
    assert "ગુજરાતી" in gujarati_words
    assert len(gujarati_words) >= 1
    # Top Gujarati corpus tokens should be learnable.
    top = _top_gujarati_tokens(SYNTHETIC_SENTENCES, n=5)
    assert any(token == "ગુજરાતી" for token, _ in top)


# ---------------------------------------------------------------------------
# Fail-fast validation
# ---------------------------------------------------------------------------

def test_validate_training_readiness_raises_on_no_gujarati():
    raw_texts = ["hello world", "foo bar", "english only news"]
    sentences = [["hello", "world"], ["foo", "bar"], ["english", "only", "news"]]
    with pytest.raises(ValueError, match="no Gujarati"):
        validate_training_readiness(raw_texts, sentences)


def test_validate_training_readiness_raises_on_all_empty_tokens():
    raw_texts = ["ગુજરાતી સમાચાર", "ગુજરાતી સમાચાર"]
    sentences = [[], []]
    with pytest.raises(ValueError, match="tokenize to an empty"):
        validate_training_readiness(raw_texts, sentences)


def test_validate_training_readiness_passes_on_clean_gujarati():
    raw_texts = ["ગુજરાતી સમાચાર", "સમાચાર વાંચો"]
    sentences = [["ગુજરાતી", "સમાચાર"], ["સમાચાર", "વાંચો"]]
    # Should not raise.
    validate_training_readiness(raw_texts, sentences)
