"""Tests for src/prediction.py (Phase 5 inference interface)."""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.classification import run_experiment, save_classifier_artifact
from src.prediction import NewsClassifier, predict_headline
from src.text_features import build_tfidf_bundle

CLASSES = ("Sports", "Business")


def _make_tfidf_bundle():
    cleaned = {
        "train": [
            "રમત ક્રિકેટ મેચ",
            "વ્યાપાર બજાર શેર",
            "રમત ફૂટબોલ મેચ",
            "વ્યાપાર નફો બજાર",
        ],
        "validation": [
            "રમત ક્રિકેટ",
            "વ્યાપાર શેર",
        ],
        "test": ["રમત ફૂટબોલ", "વ્યાપાર બજાર"],
    }
    labels = {
        "train": np.array(["Sports", "Business", "Sports", "Business"]),
        "validation": np.array(["Sports", "Business"]),
        "test": np.array(["Sports", "Business"]),
    }
    entry = {"key": "tfidf_word", "analyzer": "word", "label": "tfidf_word"}
    return build_tfidf_bundle(entry, cleaned, labels, class_names=CLASSES)


def _save_tfidf_model(tmp_path, classifier="logistic_regression"):
    bundle = _make_tfidf_bundle()
    trained = run_experiment(1, bundle, classifier, CLASSES)
    path = tmp_path / f"{classifier}.joblib"
    save_classifier_artifact(
        path,
        trained.model,
        feature_type=bundle.feature_type,
        feature_family=bundle.feature_family,
        class_names=CLASSES,
        vectorizer=trained.vectorizer,
        svd=trained.svd,
        model_name=None,
    )
    return path


def test_tfidf_prediction_returns_known_category(tmp_path):
    path = _save_tfidf_model(tmp_path)
    clf = NewsClassifier.load(path)
    label = clf.predict("રમત ક્રિકેટ મેચ")
    assert label in CLASSES

    assert predict_headline("વ્યાપાર બજાર", model_path=path) in CLASSES


def test_predict_with_scores_covers_all_classes(tmp_path):
    path = _save_tfidf_model(tmp_path, classifier="logistic_regression")
    clf = NewsClassifier.load(path)
    label, scores = clf.predict_with_scores("રમત ક્રિકેટ")
    assert label in CLASSES
    assert set(scores) == set(CLASSES)


def test_linear_svm_scores_via_decision_function(tmp_path):
    path = _save_tfidf_model(tmp_path, classifier="linear_svm")
    clf = NewsClassifier.load(path)
    _, scores = clf.predict_with_scores("વ્યાપાર શેર બજાર")
    assert set(scores) == set(CLASSES)


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_invalid_input_raises(tmp_path, bad):
    clf = NewsClassifier.load(_save_tfidf_model(tmp_path))
    with pytest.raises(ValueError):
        clf.predict(bad)


def test_docvec_inference_path(tmp_path, monkeypatch):
    from gensim.models import Word2Vec
    from sklearn.linear_model import LogisticRegression

    sentences = [
        ["રમત", "ક્રિકેટ", "મેચ"],
        ["વ્યાપાર", "બજાર", "શેર"],
    ]
    w2v = Word2Vec(sentences, vector_size=8, min_count=1, epochs=5, sg=0, seed=42)
    w2v_path = tmp_path / "word2vec_cbow.model"
    w2v.save(str(w2v_path))
    monkeypatch.setattr(config, "W2V_MODEL_CBOW_PATH", w2v_path)

    rng = np.random.RandomState(0)
    X = rng.rand(20, 8).astype(np.float32)
    y = np.array(["Sports", "Business"] * 10)
    clf = LogisticRegression(max_iter=200).fit(X, y)

    path = tmp_path / "docvec_model.joblib"
    save_classifier_artifact(
        path,
        clf,
        feature_type="docvec:cbow",
        feature_family="docvec",
        class_names=["Sports", "Business"],
        vectorizer=None,
        svd=None,
        model_name="cbow",
    )
    nc = NewsClassifier.load(path)
    assert nc.predict("રમત ક્રિકેટ મેચ") in {"Sports", "Business"}
    # An all out-of-vocabulary headline still yields a (documented) zero-vector
    # prediction rather than an error.
    assert nc.predict("zzz qqq www") in {"Sports", "Business"}
