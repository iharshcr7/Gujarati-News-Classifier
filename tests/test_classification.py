"""Tests for src/classification.py (Phase 5 training / experiment pipeline)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.classification import (
    CLASSIFIER_NAMES,
    build_classifier,
    load_classifier_artifact,
    run_all_experiments,
    run_experiment,
    save_classifier_artifact,
    select_best,
    supports_sparse,
)
from src.evaluation import ExperimentResult
from src.text_features import FeatureBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASSES = ("C0", "C1", "C2")


def _signal_matrix(rng, n, d, n_classes):
    y = np.array([CLASSES[i % n_classes] for i in range(n)])
    X = rng.rand(n, d).astype(np.float32)
    X[np.arange(n), np.array([i % n_classes for i in range(n)])] += 4.0
    return X, y


def _make_bundle(*, n=60, d=8, is_sparse=False, seed=0) -> FeatureBundle:
    rng = np.random.RandomState(seed)
    Xtr, ytr = _signal_matrix(rng, n, d, len(CLASSES))
    Xva, yva = _signal_matrix(rng, n // 3, d, len(CLASSES))
    Xte, yte = _signal_matrix(rng, n // 3, d, len(CLASSES))
    if is_sparse:
        Xtr = sparse.csr_matrix(Xtr)
        Xva = sparse.csr_matrix(Xva)
        Xte = sparse.csr_matrix(Xte)
    feature_type = "tfidf_word" if is_sparse else "docvec:cbow"
    return FeatureBundle(
        key="tfidf_word" if is_sparse else "cbow",
        label=feature_type,
        feature_type=feature_type,
        feature_family="tfidf" if is_sparse else "docvec",
        is_sparse=is_sparse,
        num_features=d,
        class_names=CLASSES,
        X={"train": Xtr, "validation": Xva, "test": Xte},
        y={"train": ytr, "validation": yva, "test": yte},
        row_ids={
            "train": np.arange(len(ytr)),
            "validation": np.arange(len(yva)),
            "test": np.arange(len(yte)),
        },
        model_name="cbow",
    )


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

def test_build_classifier_types_and_support():
    assert build_classifier("logistic_regression") is not None
    assert build_classifier("linear_svm") is not None
    assert build_classifier("random_forest", n_estimators=5, n_jobs=1) is not None
    assert build_classifier("gradient_boosting", max_iter=5) is not None
    assert supports_sparse("logistic_regression")
    assert supports_sparse("linear_svm")
    assert not supports_sparse("random_forest")
    assert not supports_sparse("gradient_boosting")
    with pytest.raises(ValueError):
        build_classifier("not_a_classifier")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("classifier", CLASSIFIER_NAMES)
def test_run_experiment_dense_all_classifiers(classifier):
    bundle = _make_bundle(is_sparse=False)
    params = {"max_iter": 20} if classifier == "gradient_boosting" else None
    if classifier == "random_forest":
        params = {"n_estimators": 10, "n_jobs": 1}
    trained = run_experiment(
        1, bundle, classifier, CLASSES, classifier_params=params
    )
    assert trained.result.metrics["macro_f1"] > 0.8
    assert trained.result.train_rows == 60
    assert trained.result.val_rows == 20
    assert trained.result.reduced_dim is None


def test_run_experiment_sparse_tree_uses_svd():
    bundle = _make_bundle(d=200, is_sparse=True)
    trained = run_experiment(
        7,
        bundle,
        "random_forest",
        CLASSES,
        svd_components=6,
        classifier_params={"n_estimators": 10, "n_jobs": 1},
    )
    assert trained.result.reduced_dim == 6
    assert trained.svd is not None
    assert trained.result.warnings  # adaptation documented
    assert trained.result.metrics["macro_f1"] > 0.5


def test_run_experiment_reproducible():
    bundle = _make_bundle(is_sparse=False, seed=3)
    params = {"n_estimators": 10, "n_jobs": 1}
    first = run_experiment(1, bundle, "random_forest", CLASSES, classifier_params=params)
    second = run_experiment(1, bundle, "random_forest", CLASSES, classifier_params=params)
    p1 = first.model.predict(bundle.X["validation"])
    p2 = second.model.predict(bundle.X["validation"])
    assert np.array_equal(p1, p2)


def test_run_all_experiments_and_select_best():
    dense = _make_bundle(is_sparse=False)
    sparse_bundle = _make_bundle(d=200, is_sparse=True)
    trained = run_all_experiments(
        [dense, sparse_bundle],
        CLASSES,
        classifiers=("logistic_regression", "linear_svm"),
        svd_components=6,
    )
    assert len(trained) == 4
    assert {t.result.index for t in trained} == {1, 2, 3, 4}

    best = select_best([t.result for t in trained])
    assert best.macro_f1 == max(t.result.macro_f1 for t in trained)


def test_select_best_tie_breaks_on_accuracy():
    def result(idx, macro_f1, accuracy):
        return ExperimentResult(
            index=idx,
            feature_type="x",
            feature_label="x",
            classifier="c",
            classifier_params="",
            train_rows=1,
            val_rows=1,
            num_features=1,
            is_sparse=False,
            train_seconds=0.0,
            eval_seconds=0.0,
            metrics={
                "accuracy": accuracy,
                "macro_precision": macro_f1,
                "macro_recall": macro_f1,
                "macro_f1": macro_f1,
                "weighted_f1": macro_f1,
                "per_class": {},
                "support": 1,
            },
        )

    best = select_best([result(1, 0.5, 0.9), result(2, 0.5, 0.6)])
    assert best.index == 1


# ---------------------------------------------------------------------------
# Artifact serialization
# ---------------------------------------------------------------------------

def test_save_and_load_artifact(tmp_path):
    bundle = _make_bundle()
    trained = run_experiment(1, bundle, "logistic_regression", CLASSES)
    path = tmp_path / "model.joblib"
    save_classifier_artifact(
        path,
        trained.model,
        feature_type=trained.result.feature_type,
        feature_family=trained.feature_family,
        class_names=CLASSES,
        vectorizer=trained.vectorizer,
        svd=trained.svd,
        model_name=trained.model_name,
        metadata={"experiment_index": 1},
    )
    assert path.exists()
    payload = load_classifier_artifact(path)
    assert payload["feature_type"] == trained.result.feature_type
    assert payload["class_names"] == list(CLASSES)
    assert hasattr(payload["model"], "predict")


def test_load_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_classifier_artifact(tmp_path / "nope.joblib")


def test_disabled_feature_configuration_is_documented():
    from src.classification import feature_configuration_notes

    notes = feature_configuration_notes()
    assert "tfidf_char" in notes
    assert notes["tfidf_char"].strip()
    # Explicitly requesting the key suppresses the "disabled" note.
    assert feature_configuration_notes(feature_keys=["tfidf_char"]) == {}
