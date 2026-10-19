"""Tests for src/experiments.py (Phase 7 targeted experiments)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from src.evaluation import ExperimentResult
from src.experiments import (
    HISTGB_BASELINE,
    ExperimentSpec,
    build_tfidf_word_bundle,
    default_cbow_configs,
    default_skipgram_configs,
    default_tfidf_word_configs,
    load_original_results,
    probability_quality,
    run_experiments,
    run_spec,
    save_experiment_outputs,
    select_replacement,
    unified_comparison_dataframe,
)
from src.text_features import FeatureBundle

CLASSES = ("Sports", "Business", "State")


def _signal(rng, n, d, n_classes):
    y = np.array([CLASSES[i % n_classes] for i in range(n)])
    X = rng.rand(n, d).astype(np.float32)
    X[np.arange(n), np.array([i % n_classes for i in range(n)])] += 4.0
    return X, y


def _dense_bundle(feature_type="docvec:skipgram", model_name="skipgram", n=90, d=8, seed=0):
    rng = np.random.RandomState(seed)
    Xtr, ytr = _signal(rng, n, d, len(CLASSES))
    Xva, yva = _signal(rng, n // 3, d, len(CLASSES))
    Xte, yte = _signal(rng, n // 3, d, len(CLASSES))
    return FeatureBundle(
        key=model_name,
        label=feature_type,
        feature_type=feature_type,
        feature_family="docvec",
        is_sparse=False,
        num_features=d,
        class_names=CLASSES,
        X={"train": Xtr, "validation": Xva, "test": Xte},
        y={"train": ytr, "validation": yva, "test": yte},
        row_ids={
            "train": np.arange(len(ytr)),
            "validation": np.arange(len(yva)),
            "test": np.arange(len(yte)),
        },
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Config specs
# ---------------------------------------------------------------------------

def test_default_configs_include_control_and_unique_keys():
    for configs in (
        default_skipgram_configs(),
        default_cbow_configs(),
        default_tfidf_word_configs(),
    ):
        assert len(configs) >= 2
        keys = [c.key for c in configs]
        assert len(keys) == len(set(keys))


def test_skipgram_configs_have_baseline_control():
    configs = default_skipgram_configs()
    control = [c for c in configs if c.key.endswith("control")]
    assert len(control) == 1
    assert control[0].params["max_iter"] == 200
    assert control[0].params["learning_rate"] == 0.1


def test_histgb_baseline_matches_selected_model_params():
    assert HISTGB_BASELINE["max_iter"] == 200
    assert HISTGB_BASELINE["learning_rate"] == 0.1
    assert HISTGB_BASELINE["max_leaf_nodes"] == 31
    assert HISTGB_BASELINE["l2_regularization"] == 0.0


def test_tfidf_configs_cover_unigram_and_bigram_estimators():
    configs = default_tfidf_word_configs()
    ngrams = {c.ngram_range for c in configs}
    assert (1, 1) in ngrams and (1, 2) in ngrams
    classifiers = {c.classifier for c in configs}
    assert {"logistic_regression", "linear_svm"}.issubset(classifiers)


# ---------------------------------------------------------------------------
# TF-IDF bundle
# ---------------------------------------------------------------------------

def test_build_tfidf_word_bundle_excludes_validation_only_token():
    cleaned = {
        "train": ["રમત ક્રિકેટ મેચ", "વ્યાપાર બજાર શેર", "રમત ફૂટબોલ"],
        "validation": ["અનન્યશબ્દ રમત"],
        "test": ["રમત"],
    }
    labels = {
        "train": np.array(["Sports", "Business", "Sports"]),
        "validation": np.array(["Sports"]),
        "test": np.array(["Sports"]),
    }
    bundle = build_tfidf_word_bundle(
        ngram_range=(1, 1),
        cleaned=cleaned,
        labels=labels,
        class_names=CLASSES,
        key="tfidf_word_uni",
        label="word unigram",
    )
    assert bundle.is_sparse
    assert bundle.feature_type == "tfidf_word_uni"
    vocab = bundle.vectorizer.vocabulary_
    assert "અનન્યશબ્દ" not in vocab  # validation-only token must be excluded
    assert "રમત" in vocab


# ---------------------------------------------------------------------------
# Running a spec
# ---------------------------------------------------------------------------

def test_run_spec_trains_and_records_n_iter(tmp_path):
    bundle = _dense_bundle()
    spec = ExperimentSpec(
        key="sg_histgb_control",
        feature_key="skipgram",
        feature_label="Skip-gram doc vectors (100d)",
        kind="docvec",
        classifier="gradient_boosting",
        params={"max_iter": 15, "learning_rate": 0.1},
    )
    trained, extra = run_spec(
        101, spec, {"skipgram": bundle}, CLASSES, seed=42, artifact_dir=tmp_path
    )
    assert trained.result.metrics["macro_f1"] > 0.5
    assert extra["key"] == "sg_histgb_control"
    assert extra["n_iter"] is not None
    assert (tmp_path / "exp101_sg_histgb_control.joblib").exists()
    assert trained.result.artifact_path.endswith("sg_histgb_control.joblib")


def test_run_experiments_collects_results_and_failures(tmp_path):
    bundle = _dense_bundle()
    specs = [
        ExperimentSpec("good", "skipgram", "lbl", "docvec", "logistic_regression", {}),
        ExperimentSpec("bad", "missing", "lbl", "docvec", "logistic_regression", {}),
    ]
    outcome = run_experiments(
        specs, {"skipgram": bundle}, CLASSES, seed=42, artifact_dir=tmp_path
    )
    assert len(outcome["trained"]) == 1
    assert outcome["trained"][0].result.metrics["macro_f1"] > 0.5
    assert len(outcome["failures"]) == 1
    assert outcome["failures"][0]["key"] == "bad"


# ---------------------------------------------------------------------------
# Comparison + selection
# ---------------------------------------------------------------------------

def _result(index, feature_type, classifier, macro_f1, accuracy, source=None):
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": macro_f1,
        "per_class": {c: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1} for c in CLASSES},
        "support": 3,
    }
    return ExperimentResult(
        index=index,
        feature_type=feature_type,
        feature_label=feature_type,
        classifier=classifier,
        classifier_params="seed=42",
        train_rows=10,
        val_rows=3,
        num_features=100,
        is_sparse=False,
        train_seconds=1.0,
        eval_seconds=0.1,
        metrics=metrics,
    )


def test_unified_comparison_dataframe_tags_source_and_ranks():
    original = [_result(8, "docvec:skipgram", "gradient_boosting", 0.888, 0.894)]
    new = [_result(101, "docvec:skipgram", "gradient_boosting", 0.900, 0.905)]
    df = unified_comparison_dataframe(original, new)
    assert set(df["source"]) == {"phase5_original", "phase7_experiment"}
    assert df.iloc[0]["index"] == 101  # higher macro-F1 first
    assert "rank" in df.columns


def test_select_replacement_threshold():
    assert select_replacement(0.888, 0.893) is True   # +0.005 exactly
    assert select_replacement(0.888, 0.900) is True
    assert select_replacement(0.888, 0.8929) is False
    assert select_replacement(0.888, 0.880) is False


def test_load_original_results_roundtrip(tmp_path):
    import json
    from src.evaluation import result_to_json

    original = [_result(1, "docvec:cbow", "linear_svm", 0.85, 0.86)]
    path = tmp_path / "experiments.json"
    path.write_text(
        json.dumps([result_to_json(r) for r in original]), encoding="utf-8"
    )
    loaded = load_original_results(path)
    assert len(loaded) == 1
    assert loaded[0].macro_f1 == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Probability quality
# ---------------------------------------------------------------------------

def test_probability_quality_perfect_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["Sports", "Business", "State", "Sports"])

    class Perfect:
        classes_ = np.array(list(CLASSES))

        def predict_proba(self, X):
            out = np.full((len(X), len(CLASSES)), 1e-9)
            for i in range(len(X)):
                out[i, list(CLASSES).index(y[i])] = 1.0
            out = out / out.sum(axis=1, keepdims=True)
            return out

    q = probability_quality(Perfect(), X, y, CLASSES)
    assert q["available"] is True
    assert q["ece"] < 1e-6
    assert q["log_loss"] < 1e-6
    assert q["brier"] < 1e-6


def test_probability_quality_unavailable_without_predict_proba():
    class Margins:
        classes_ = np.array(list(CLASSES))

        def decision_function(self, X):
            return np.zeros((len(X), len(CLASSES)))

    q = probability_quality(Margins(), np.zeros((2, 1)), ["Sports", "Business"], CLASSES)
    assert q["available"] is False


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def test_save_experiment_outputs_writes_files(tmp_path):
    df = unified_comparison_dataframe(
        [_result(8, "docvec:skipgram", "gradient_boosting", 0.888, 0.894)],
        [_result(101, "docvec:skipgram", "gradient_boosting", 0.900, 0.905)],
    )
    paths = save_experiment_outputs(
        tmp_path,
        df,
        failures=[{"key": "bad", "error": "boom"}],
        title="Phase 7 experiments",
    )
    for key in ("csv", "json", "markdown", "plot", "failures"):
        assert key in paths
        assert paths[key].exists()
    table = paths["markdown"].read_text(encoding="utf-8")
    assert "Gradient" in table or "gradient_boosting" in table
