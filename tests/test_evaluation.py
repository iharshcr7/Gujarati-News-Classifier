"""Tests for src/evaluation.py (Phase 5 metrics and reporting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    ExperimentResult,
    build_validation_comparison_table,
    classification_report_text,
    compute_metrics,
    confusion_matrix_array,
    experiments_to_dataframe,
    result_to_json,
    save_confusion_matrix_plot,
    save_metric_comparison_plot,
)


CLASSES = ["State", "Sports"]


def _make_result(index, feature_type, classifier, macro_f1, accuracy):
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": macro_f1,
        "per_class": {
            "State": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
            "Sports": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
        },
        "support": 4,
    }
    return ExperimentResult(
        index=index,
        feature_type=feature_type,
        feature_label=feature_type,
        classifier=classifier,
        classifier_params="seed=42",
        train_rows=10,
        val_rows=4,
        num_features=5,
        is_sparse=False,
        train_seconds=0.1,
        eval_seconds=0.01,
        metrics=metrics,
    )


def test_compute_metrics_known_values():
    y_true = ["State", "State", "Sports", "Sports"]
    y_pred = ["State", "Sports", "Sports", "Sports"]
    m = compute_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["macro_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert m["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["support"] == 4
    assert set(m["per_class"]) == set(CLASSES)
    assert m["per_class"]["State"]["support"] == 2
    assert m["per_class"]["Sports"]["recall"] == pytest.approx(1.0)


def test_compute_metrics_length_mismatch():
    with pytest.raises(ValueError):
        compute_metrics(["A"], ["A", "B"], ["A", "B"])


def test_confusion_matrix_counts():
    cm = confusion_matrix_array(
        ["State", "State", "Sports", "Sports"],
        ["State", "Sports", "Sports", "Sports"],
        CLASSES,
    )
    assert cm.shape == (2, 2)
    assert cm.sum() == 4
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 0] == 0 and cm[1, 1] == 2


def test_classification_report_text_has_categories():
    text = classification_report_text(
        ["State", "Sports"], ["State", "State"], CLASSES
    )
    assert "State" in text
    assert "Sports" in text
    assert "precision" in text


def test_experiments_to_dataframe_columns_and_sort():
    results = [
        _make_result(1, "cbow", "logistic_regression", 0.60, 0.62),
        _make_result(2, "tfidf_word", "linear_svm", 0.75, 0.77),
    ]
    df = experiments_to_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    for col in ("index", "feature_type", "classifier", "macro_f1", "accuracy"):
        assert col in df.columns
    table = build_validation_comparison_table(results)
    # Higher macro-F1 experiment should appear first.
    assert table.index("tfidf_word") < table.index("cbow")


def test_result_to_json_keeps_per_class():
    payload = result_to_json(_make_result(1, "cbow", "logistic_regression", 0.5, 0.5))
    assert payload["metrics"]["per_class"]["State"]["f1"] == 1.0
    assert isinstance(payload["warnings"], list)


def test_plots_are_written(tmp_path):
    cm = np.array([[3, 1], [2, 4]])
    cm_path = save_confusion_matrix_plot(
        cm, CLASSES, tmp_path / "cm.png", "test cm"
    )
    assert cm_path.exists() and cm_path.stat().st_size > 0

    df = experiments_to_dataframe(
        [
            _make_result(1, "cbow", "linear_svm", 0.6, 0.6),
            _make_result(2, "tfidf_word", "logistic_regression", 0.7, 0.7),
        ]
    )
    plot_path = save_metric_comparison_plot(df, tmp_path / "f1.png")
    assert plot_path.exists() and plot_path.stat().st_size > 0
