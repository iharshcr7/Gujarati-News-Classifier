"""Unit tests for src/document_vectors.py (Phase 4: document vectors).

The tests build tiny synthetic Word2Vec models and tiny CSV fixtures so the
suite stays quick; the real CBOW/Skip-gram artifacts are used by the
``generate_document_vectors.py`` entry point on the configured SHC splits.

Run with:

    python -m pytest tests/test_document_vectors.py -v
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

import config
from src.document_vectors import (
    DocumentVectorSet,
    build_document_vector_report,
    build_document_vector_set,
    build_metadata,
    document_vector_filename,
    generate_document_vectors,
    load_document_vectors,
    mean_pool_tokens,
    save_document_vectors,
    tokenize_split,
    vectorize_tokens,
    write_metadata,
)
from src.preprocessing import preprocess
from src.word2vec_trainer import save_model, train_word2vec

# Small synthetic Gujarati corpus used to train fast unit-test models.
SYNTHETIC_SENTENCES: List[List[str]] = [
    ["ગુજરાતી", "સમાચાર", "આજ"],
    ["ગુજરાતી", "સમાચાર", "કાલ"],
    ["આજ", "ગુજરાતી", "સમાચાર", "વાંચો"],
    ["કાલ", "સમાચાર", "બ્યુઝનેસ"],
    ["ગુજરાતી", "ક્રિકેટ", "મેચ"],
    ["ક્રિકેટ", "મેચ", "આજ"],
    ["ગુજરાતી", "સમાચાર"],
    ["ગુજરાતી", "સમાચાર"],
]


def _tiny_model(sg: int = 0, dim: int = 16):
    model, _ = train_word2vec(
        SYNTHETIC_SENTENCES,
        sg=sg,
        vector_size=dim,
        window=3,
        min_count=1,
        workers=1,
        seed=42,
        epochs=3,
    )
    return model


# ---------------------------------------------------------------------------
# mean_pool_tokens
# ---------------------------------------------------------------------------

def test_mean_pool_averages_only_in_vocabulary_tokens():
    key_to_index = {"a": 0, "b": 1, "c": 2}
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    vec, n = mean_pool_tokens(["a", "c", "ઓઓવી_શબ્દ"], key_to_index, vectors)
    assert n == 2
    assert vec.shape == (3,)
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec, [0.5, 0.0, 0.5])


def test_mean_pool_oov_only_token_is_ignored():
    key_to_index = {"a": 0}
    vectors = np.array([[2.0, 4.0]], dtype=np.float32)
    vec, n = mean_pool_tokens(["a", "zzz"], key_to_index, vectors)
    assert n == 1
    np.testing.assert_allclose(vec, [2.0, 4.0])


def test_mean_pool_empty_tokens_returns_zero_vector():
    key_to_index = {"a": 0}
    vectors = np.array([[1.0, 1.0]], dtype=np.float32)
    vec, n = mean_pool_tokens([], key_to_index, vectors)
    assert n == 0
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, np.zeros(2, dtype=np.float32))


def test_mean_pool_all_oov_returns_zero_vector():
    key_to_index = {"a": 0}
    vectors = np.array([[1.0, 1.0]], dtype=np.float32)
    vec, n = mean_pool_tokens(["x", "y"], key_to_index, vectors)
    assert n == 0
    np.testing.assert_array_equal(vec, np.zeros(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# tokenize_split (reuses src.preprocessing)
# ---------------------------------------------------------------------------

def test_tokenize_split_uses_preprocessing_pipeline():
    df = pd.DataFrame(
        {"text": ["ગુજરાતી https://example.com સમાચાર,!", None, ""], "labels": ["State", "B", "C"]}
    )
    tokens = tokenize_split(df)
    assert tokens[0] == preprocess("ગુજરાતી https://example.com સમાચાર, ").tokens
    assert tokens[0] == ["ગુજરાતી", "સમાચાર"]
    assert tokens[1] == []
    assert tokens[2] == []


# ---------------------------------------------------------------------------
# vectorize_tokens / build_document_vector_set
# ---------------------------------------------------------------------------

def test_vectorize_tokens_shapes_and_counts():
    model = _tiny_model(sg=0)
    dim = model.vector_size
    token_lists = [
        ["ગુજરાતી", "સમાચાર"],
        ["પૂરેપૂરું_ઓઓવી_ટોકન"],
        [],
    ]
    X, stats = vectorize_tokens(token_lists, model, model_name="cbow", split="train")
    assert X.shape == (3, dim)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert stats["fallback_count"] == 2  # all-OOV row + empty row
    assert stats["empty_token_count"] == 1
    assert stats["total_oov_tokens"] == 1
    assert stats["total_in_vocab_tokens"] == 2


def test_build_document_vector_set_alignment_and_values():
    model = _tiny_model(sg=0)
    token_lists = [["ગુજરાતી", "સમાચાર"], ["પૂરેપૂરું_ઓઓવી_ટોકન"], []]
    labels = ["State", "Business", "Crime"]
    dvs = build_document_vector_set(
        token_lists, labels, model, model_name="cbow", split="train"
    )
    assert isinstance(dvs, DocumentVectorSet)
    assert dvs.vectors.shape == (3, model.vector_size)
    assert dvs.dim == model.vector_size
    assert dvs.model_name == "cbow"
    assert dvs.split == "train"
    assert dvs.num_rows == 3
    assert dvs.fallback_count == 2
    assert dvs.empty_token_count == 1
    # Stable row identifier = positional index, labels aligned with source.
    np.testing.assert_array_equal(dvs.row_ids, np.arange(3))
    assert list(dvs.labels) == labels
    # Fallback rows are exact zero vectors and finite.
    np.testing.assert_array_equal(dvs.vectors[1], np.zeros(model.vector_size, dtype=np.float32))
    np.testing.assert_array_equal(dvs.vectors[2], np.zeros(model.vector_size, dtype=np.float32))
    # In-vocabulary row is the mean of the known token vectors.
    expected = np.mean(
        [model.wv["ગુજરાતી"], model.wv["સમાચાર"]], axis=0
    ).astype(np.float32)
    np.testing.assert_allclose(dvs.vectors[0], expected, rtol=1e-5, atol=1e-6)


def test_build_document_vector_set_all_finite():
    model = _tiny_model(sg=1)
    dvs = build_document_vector_set(
        [["ગુજરાતી"], ["ક્રિકેટ", "મેચ"]], ["A", "B"], model,
        model_name="skipgram", split="validation",
    )
    assert dvs.vectors.shape[1] == model.vector_size
    assert np.isfinite(dvs.vectors).all()
    assert dvs.fallback_count == 0


# ---------------------------------------------------------------------------
# Model is never mutated (no retraining / no vector updates)
# ---------------------------------------------------------------------------

def test_generation_does_not_modify_model_vectors():
    model = _tiny_model(sg=0)
    before = model.wv.vectors.copy()
    build_document_vector_set(
        [["ગુજરાતી", "સમાચાર"], ["ક્રિકેટ"]], ["A", "B"], model,
        model_name="cbow", split="train",
    )
    np.testing.assert_array_equal(model.wv.vectors, before)


# ---------------------------------------------------------------------------
# Persistence + metadata
# ---------------------------------------------------------------------------

def test_document_vector_filename_convention():
    assert (
        document_vector_filename("Gujarati_SHC_LDC_LPC_Train", "cbow")
        == "Gujarati_SHC_LDC_LPC_Train.cbow.docvecs.npz"
    )
    assert (
        document_vector_filename("Gujarati_SHC_LDC_LPC_Test", "skipgram")
        == "Gujarati_SHC_LDC_LPC_Test.skipgram.docvecs.npz"
    )


def test_save_and_load_document_vectors_roundtrip(tmp_path):
    model = _tiny_model(sg=0)
    dvs = build_document_vector_set(
        [["ગુજરાતી"], ["ઓઓવી"]], ["State", "Crime"], model,
        model_name="cbow", split="test",
    )
    path = tmp_path / "x.docvecs.npz"
    save_document_vectors(dvs, path)
    assert path.exists()
    loaded = load_document_vectors(path)
    np.testing.assert_array_equal(loaded["X"], dvs.vectors)
    assert list(loaded["labels"]) == list(dvs.labels)
    np.testing.assert_array_equal(loaded["row_ids"], dvs.row_ids)


def test_build_metadata_records_required_fields():
    model = _tiny_model(sg=1)
    dvs = build_document_vector_set(
        [["ગુજરાતી"], ["ઓઓવી"]], ["State", "Crime"], model,
        model_name="skipgram", split="validation",
    )
    meta = build_metadata(
        dvs,
        model_path=Path("models/word2vec_skipgram.model"),
        model_sg=1,
        vocab_size=len(model.wv),
        preprocessing_method="src.preprocessing.preprocess (whitespace tokenization, stopword removal disabled)",
        source_file="Gujarati_SHC_LDC_LPC_Valid.csv",
    )
    required = {
        "embedding_model",
        "model_sg",
        "vector_dimension",
        "preprocessing_method",
        "source_split",
        "source_file",
        "num_rows",
        "fallback_count",
        "fallback_fraction",
        "empty_token_count",
        "out_of_vocabulary_tokens",
        "in_vocabulary_tokens",
        "vocab_size",
        "model_path",
    }
    assert required <= set(meta)
    assert meta["embedding_model"] == "skipgram"
    assert meta["model_sg"] == 1
    assert meta["vector_dimension"] == model.vector_size
    assert meta["source_split"] == "validation"
    assert meta["fallback_count"] == 1
    assert meta["num_rows"] == 2


def test_write_metadata_json(tmp_path):
    meta = {"embedding_model": "cbow", "vector_dimension": 100}
    path = write_metadata(meta, tmp_path / "meta.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["embedding_model"] == "cbow"
    assert data["vector_dimension"] == 100


def test_build_report_contains_required_sections():
    model = _tiny_model(sg=0)
    dvs = build_document_vector_set(
        [["ગુજરાતી"], ["ઓઓવી"]], ["State", "Crime"], model,
        model_name="cbow", split="train",
    )
    report = build_document_vector_report(
        [{"model": "cbow", "split": "train", "dvs": dvs, "meta": {"fallback_fraction": 0.5}}],
        input_counts={"train": 2, "validation": 0, "test": 0},
    )
    for header in (
        "DOCUMENT VECTOR GENERATION REPORT",
        "DATA LEAKAGE",
        "ROW COUNTS",
        "FALLBACK",
        "DIMENSION",
        "ALIGNMENT",
    ):
        assert header in report.upper()


# ---------------------------------------------------------------------------
# Full pipeline on tiny fixtures (no real artifacts touched)
# ---------------------------------------------------------------------------

def _write_fixture_csv(path: Path, text_label_pairs) -> None:
    df = pd.DataFrame(text_label_pairs, columns=["text", "labels"])
    df.to_csv(path, index=False, encoding="utf-8")


def _patch_config_paths(monkeypatch, tmp_path, train_p, valid_p, test_p):
    monkeypatch.setattr(config, "TRAIN_PATH", train_p)
    monkeypatch.setattr(config, "VALID_PATH", valid_p)
    monkeypatch.setattr(config, "TEST_PATH", test_p)
    for attr in ("TRAIN_HEADER", "VALID_HEADER", "TEST_HEADER"):
        monkeypatch.setattr(config, attr, 0)
    for attr in ("TRAIN_ENCODING", "VALID_ENCODING", "TEST_ENCODING"):
        monkeypatch.setattr(config, attr, "utf-8")
    monkeypatch.setattr(config, "TRAIN_ENCODING_ERRORS", None)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")


def test_generate_document_vectors_pipeline(monkeypatch, tmp_path):
    train_p = tmp_path / "Gujarati_SHC_LDC_LPC_Train.csv"
    valid_p = tmp_path / "Gujarati_SHC_LDC_LPC_Valid.csv"
    test_p = tmp_path / "Gujarati_SHC_LDC_LPC_Test.csv"
    _write_fixture_csv(
        train_p,
        [
            ("ગુજરાતી સમાચાર આજ", "State"),
            ("ગુજરાતી ક્રિકેટ મેચ", "Sports"),
            ("ગુજરાતી સમાચાર કાલ", "State"),
        ],
    )
    _write_fixture_csv(
        valid_p,
        [("ગુજરાતી ક્રિકેટ", "Sports"), ("એકદમ_અજાણ્યો_શબ્દ", "Culture")],
    )
    _write_fixture_csv(test_p, [("ગુજરાતી સમાચાર", "State")])

    _patch_config_paths(monkeypatch, tmp_path, train_p, valid_p, test_p)

    # Train and save tiny models (in tmp_path; real models/ untouched).
    train_tokens = [preprocess(t).tokens for t in pd.read_csv(train_p)["text"]]
    cbow, _ = train_word2vec(train_tokens, sg=0, vector_size=16, window=3, min_count=1, workers=1, seed=42, epochs=3)
    sg, _ = train_word2vec(train_tokens, sg=1, vector_size=16, window=3, min_count=1, workers=1, seed=42, epochs=3)
    cbow_path = tmp_path / "word2vec_cbow.model"
    sg_path = tmp_path / "word2vec_skipgram.model"
    save_model(cbow, cbow_path, overwrite=True)
    save_model(sg, sg_path, overwrite=True)
    monkeypatch.setattr(config, "W2V_MODEL_CBOW_PATH", cbow_path)
    monkeypatch.setattr(config, "W2V_MODEL_SKIPGRAM_PATH", sg_path)
    monkeypatch.setattr(config, "W2V_VECTOR_SIZE", 16)

    src_hashes_before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_p, valid_p, test_p)
    }
    model_hashes_before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cbow_path, sg_path)
    }

    report, report_path, result = generate_document_vectors(
        output_dir=tmp_path / "processed",
        report_filename="document_vectors_report.txt",
    )

    assert set(result) == {"cbow", "skipgram"}
    for model_name in ("cbow", "skipgram"):
        assert set(result[model_name]) == {"train", "validation", "test"}
        for split in ("train", "validation", "test"):
            dvs = result[model_name][split]
            assert dvs.vectors.shape[1] == 16
            assert np.isfinite(dvs.vectors).all()
            assert dvs.vectors.shape[0] == dvs.num_rows

    # Per-model per-split artifacts + metadata exist.
    processed = tmp_path / "processed"
    for stem in ("Gujarati_SHC_LDC_LPC_Train", "Gujarati_SHC_LDC_LPC_Valid", "Gujarati_SHC_LDC_LPC_Test"):
        for model_name in ("cbow", "skipgram"):
            npz = processed / document_vector_filename(stem, model_name)
            meta = processed / document_vector_filename(stem, model_name).replace(".npz", ".meta.json")
            assert npz.exists()
            assert meta.exists()
            data = json.loads(meta.read_text(encoding="utf-8"))
            assert data["embedding_model"] == model_name
            assert data["vector_dimension"] == 16
            assert "source_split" in data
    assert report_path.exists()
    assert "DOCUMENT VECTOR GENERATION REPORT" in report

    # Source CSVs and model files must be untouched.
    src_hashes_after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_p, valid_p, test_p)
    }
    model_hashes_after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cbow_path, sg_path)
    }
    assert src_hashes_before == src_hashes_after
    assert model_hashes_before == model_hashes_after


def test_generate_document_vectors_validation_labels_aligned(monkeypatch, tmp_path):
    train_p = tmp_path / "Gujarati_SHC_LDC_LPC_Train.csv"
    valid_p = tmp_path / "Gujarati_SHC_LDC_LPC_Valid.csv"
    test_p = tmp_path / "Gujarati_SHC_LDC_LPC_Test.csv"
    _write_fixture_csv(train_p, [("ગુજરાતી સમાચાર આજ", "State"), ("ગુજરાતી ક્રિકેટ મેચ", "Sports")])
    _write_fixture_csv(valid_p, [("ગુજરાતી સમાચાર", "Business")])
    _write_fixture_csv(test_p, [("ગુજરાતી સમાચાર", "Crime")])
    _patch_config_paths(monkeypatch, tmp_path, train_p, valid_p, test_p)

    train_tokens = [preprocess(t).tokens for t in pd.read_csv(train_p)["text"]]
    model, _ = train_word2vec(train_tokens, sg=0, vector_size=8, window=3, min_count=1, workers=1, seed=42, epochs=3)
    cbow_path = tmp_path / "word2vec_cbow.model"
    sg_path = tmp_path / "word2vec_skipgram.model"
    save_model(model, cbow_path, overwrite=True)
    save_model(model, sg_path, overwrite=True)
    monkeypatch.setattr(config, "W2V_MODEL_CBOW_PATH", cbow_path)
    monkeypatch.setattr(config, "W2V_MODEL_SKIPGRAM_PATH", sg_path)
    monkeypatch.setattr(config, "W2V_VECTOR_SIZE", 8)

    _, _, result = generate_document_vectors(output_dir=tmp_path / "processed")
    assert list(result["cbow"]["validation"].labels) == ["Business"]
    assert list(result["cbow"]["test"].labels) == ["Crime"]
    np.testing.assert_array_equal(result["cbow"]["validation"].row_ids, np.arange(1))
