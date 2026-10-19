"""Tests for src/text_features.py (Phase 5 feature construction)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import config
from src.text_features import (
    SPLITS,
    TfidfConfig,
    build_tfidf_features,
    docvec_path,
    labels_match_docvec,
    load_docvec_split,
    svd_reduce,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_docvec(
    path,
    n=10,
    dim=100,
    labels=None,
    row_ids=None,
    X=None,
):
    if X is None:
        X = np.random.RandomState(0).rand(n, dim).astype(np.float32)
    if labels is None:
        labels = np.array(
            [["State", "Sports"][i % 2] for i in range(n)], dtype="<U13"
        )
    if row_ids is None:
        row_ids = np.arange(n, dtype=np.int64)
    np.savez_compressed(path, X=X, labels=labels, row_ids=row_ids)
    return path


def _point_config_at_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config, "TRAIN_PATH", tmp_path / "Gujarati_SHC_LDC_LPC_Train.csv"
    )
    monkeypatch.setattr(
        config, "VALID_PATH", tmp_path / "Gujarati_SHC_LDC_LPC_Valid.csv"
    )
    monkeypatch.setattr(
        config, "TEST_PATH", tmp_path / "Gujarati_SHC_LDC_LPC_Test.csv"
    )


# ---------------------------------------------------------------------------
# Document-vector loading
# ---------------------------------------------------------------------------

def test_docvec_path_unknown_model():
    with pytest.raises(ValueError):
        docvec_path("nope", "train")


def test_load_docvec_split_roundtrip(tmp_path, monkeypatch):
    _point_config_at_tmp(tmp_path, monkeypatch)
    path = docvec_path("cbow", "train", processed_dir=tmp_path)
    _write_docvec(path, n=6)

    fm = load_docvec_split(
        "cbow", "train", processed_dir=tmp_path,
        class_names=["State", "Sports"],
    )
    assert fm.X.shape == (6, 100)
    assert fm.X.dtype == np.float32
    assert fm.num_features == 100
    assert fm.is_sparse is False
    assert fm.class_names == ("State", "Sports")
    assert np.array_equal(fm.row_ids, np.arange(6))
    assert list(fm.y[:2]) == ["State", "Sports"]


def test_load_docvec_split_missing_file(tmp_path, monkeypatch):
    _point_config_at_tmp(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        load_docvec_split("cbow", "train", processed_dir=tmp_path)


def test_load_docvec_split_rejects_non_finite(tmp_path, monkeypatch):
    _point_config_at_tmp(tmp_path, monkeypatch)
    path = docvec_path("skipgram", "train", processed_dir=tmp_path)
    X = np.random.rand(5, 4).astype(np.float32)
    X[0, 0] = np.nan
    _write_docvec(path, n=5, dim=4, X=X)
    with pytest.raises(ValueError, match="non-finite"):
        load_docvec_split("skipgram", "train", processed_dir=tmp_path)


def test_load_docvec_split_rejects_bad_row_ids(tmp_path, monkeypatch):
    _point_config_at_tmp(tmp_path, monkeypatch)
    path = docvec_path("cbow", "train", processed_dir=tmp_path)
    _write_docvec(path, n=4, row_ids=np.array([0, 1, 3, 4]))
    with pytest.raises(ValueError, match="row_ids"):
        load_docvec_split("cbow", "train", processed_dir=tmp_path)


def test_labels_match_docvec(tmp_path, monkeypatch):
    _point_config_at_tmp(tmp_path, monkeypatch)
    path = docvec_path("cbow", "train", processed_dir=tmp_path)
    _write_docvec(path, n=4, labels=np.array(["State", "Sports", "State", "Sports"]))
    fm = load_docvec_split("cbow", "train", processed_dir=tmp_path)
    assert labels_match_docvec(["State", "Sports", "State", "Sports"], fm)
    assert not labels_match_docvec(["Sports", "State", "State", "Sports"], fm)


# ---------------------------------------------------------------------------
# TF-IDF leakage prevention
# ---------------------------------------------------------------------------

def test_tfidf_fitted_on_train_only():
    cleaned = {
        "train": ["gujarati cricket match today", "state election result"],
        "validation": ["politics election special"],
        "test": ["stock market crash"],
    }
    vectorizer, matrices = build_tfidf_features(
        cleaned, TfidfConfig(ngram_range=(1, 1), min_df=1, max_features=1000)
    )
    vocab = vectorizer.vocabulary_
    # Tokens occurring only in validation/test must never enter the vocabulary.
    assert "politics" not in vocab
    assert "special" not in vocab
    assert "stock" not in vocab
    assert "cricket" in vocab
    assert set(matrices) == set(SPLITS)
    assert matrices["train"].shape[0] == 2
    assert matrices["validation"].shape[0] == 1
    assert sparse.issparse(matrices["train"])


def test_tfidf_transform_unknown_token_is_zero():
    cleaned = {
        "train": ["gujarati news"],
        "validation": ["completelyunseentoken"],
        "test": ["gujarati news"],
    }
    vectorizer, matrices = build_tfidf_features(
        cleaned, TfidfConfig(ngram_range=(1, 1), min_df=1)
    )
    assert matrices["validation"].nnz == 0


# ---------------------------------------------------------------------------
# SVD reduction
# ---------------------------------------------------------------------------

def test_svd_reduce_shape_and_cap():
    X = sparse.random(20, 50, density=0.1, random_state=0).tocsr()
    svd, dense = svd_reduce(X, n_components=8, seed=42)
    assert dense.shape == (20, 8)
    assert dense.dtype == np.float32
    # Requesting more components than possible is capped, not an error.
    svd2, dense2 = svd_reduce(X, n_components=999, seed=42)
    assert dense2.shape[0] == 20
    assert dense2.shape[1] <= 50
