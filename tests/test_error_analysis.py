"""Tests for src/error_analysis.py (Phase 7 validation error analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation import compute_metrics
from src.error_analysis import (
    ScoreInfo,
    build_error_analysis_report,
    build_baseline_reproduction_report,
    build_error_cause_investigation,
    class_vocabulary_overlap,
    collect_predictions,
    compare_to_recorded,
    confusion_pairs,
    enrich_token_features,
    error_breakdown,
    get_prediction_scores,
    misclassified_examples,
    save_error_analysis,
)

CLASSES = ("State", "Sports", "Crime")

# Genuine Gujarati headlines used to prove Unicode survives every report/CSV.
TEXTS = [
    "રાજ્ય સરકારે નવી યોજના જાહેર કરી",       # State
    "ભારતે ક્રિકેટ મેચ જીતી લીધી",           # Sports
    "પોલીસે ચોરીના કેસમાં આરોપીની ધરપકડ કરી",  # Crime
    "રાજ્યમાં વરસાદને કારણે પૂર આવ્યું",       # State
    "ટીમ ઇન્ડિયાએ શ્રેણી જીતી",                # Sports
]


def _toy_data():
    rng = np.random.RandomState(0)
    X = rng.rand(len(TEXTS), 4).astype(np.float32)
    y = np.array(["State", "Sports", "Crime", "State", "Sports"])
    return X, y


def _trained_proba_model():
    X, y = _toy_data()
    model = LogisticRegression(max_iter=500).fit(X, y)
    return model, X, y


def _trained_margin_model():
    X, y = _toy_data()
    model = LinearSVC(max_iter=5000).fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Score semantics
# ---------------------------------------------------------------------------

def test_get_prediction_scores_probability_kind():
    model, X, _ = _trained_proba_model()
    info = get_prediction_scores(model, X, CLASSES)
    assert isinstance(info, ScoreInfo)
    assert info.kind == "probability"
    assert info.source == "predict_proba"
    assert info.values is not None and info.values.shape == (len(X), len(CLASSES))


def test_get_prediction_scores_decision_kind():
    model, X, _ = _trained_margin_model()
    info = get_prediction_scores(model, X, CLASSES)
    assert info.kind == "decision"
    assert info.source == "decision_function"
    assert info.values is not None and info.values.shape == (len(X), len(CLASSES))


def test_get_prediction_scores_unavailable_for_bare_object():
    class Bare:
        def predict(self, X):  # pragma: no cover - not called here
            return np.array(["State"])

    info = get_prediction_scores(Bare(), np.zeros((2, 3)), CLASSES)
    assert info.kind == "unavailable"
    assert info.source == "none"
    assert info.values is None


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def test_collect_predictions_columns_and_alignment():
    model, X, y = _trained_proba_model()
    df = collect_predictions(
        model, X, y, CLASSES, row_ids=np.arange(len(y)), texts=TEXTS
    )
    expected = {
        "row_id",
        "text",
        "true_category",
        "predicted_category",
        "correct",
        "score_type",
        "predicted_score",
        "runner_up_category",
        "runner_up_score",
        "margin",
    }
    assert expected.issubset(df.columns)
    assert len(df) == len(y)
    assert list(df["true_category"]) == list(y)
    assert list(df["row_id"]) == list(range(len(y)))
    # Gujarati text preserved exactly.
    assert list(df["text"]) == TEXTS
    # Probability semantics recorded truthfully.
    assert set(df["score_type"]) == {"probability"}
    assert df["correct"].dtype == bool
    # margin = predicted_score - runner_up_score, never negative.
    assert (df["margin"] >= -1e-9).all()


def test_collect_predictions_decision_scores_recorded():
    model, X, y = _trained_margin_model()
    df = collect_predictions(model, X, y, CLASSES)
    assert set(df["score_type"]) == {"decision"}
    assert df["text"].eq("").all()  # no texts supplied


def test_collect_predictions_unavailable_scores_are_blank():
    class Deterministic:
        classes_ = np.array(list(CLASSES))

        def predict(self, X):
            return np.array([CLASSES[0]] * len(X))

    y = np.array(["State", "Sports"])
    df = collect_predictions(Deterministic(), np.zeros((2, 3)), y, CLASSES)
    assert set(df["score_type"]) == {"unavailable"}
    assert df["predicted_score"].isna().all()
    assert df["margin"].isna().all()


# ---------------------------------------------------------------------------
# Confusion pairs
# ---------------------------------------------------------------------------

def test_confusion_pairs_counts_and_shares():
    y_true = ["State", "State", "State", "Sports", "Sports", "Crime"]
    y_pred = ["State", "Sports", "Sports", "Sports", "Crime", "Crime"]
    pairs = confusion_pairs(y_true, y_pred, CLASSES)
    # 3 errors: State->Sports x2, Sports->Crime x1.
    assert pairs["count"].sum() == 3
    top = pairs.iloc[0]
    assert top["true_category"] == "State"
    assert top["predicted_category"] == "Sports"
    assert top["count"] == 2
    assert top["share_of_true_class"] == pytest.approx(2 / 3)
    assert top["share_of_predicted_class"] == pytest.approx(2 / 3)
    assert top["share_of_all_errors"] == pytest.approx(2 / 3)
    second = pairs.iloc[1]
    assert (second["true_category"], second["predicted_category"]) == ("Sports", "Crime")
    assert second["share_of_true_class"] == pytest.approx(1 / 2)
    assert second["share_of_predicted_class"] == pytest.approx(1 / 2)
    assert second["share_of_all_errors"] == pytest.approx(1 / 3)


def test_confusion_pairs_excludes_correct_predictions():
    y = ["State", "Sports", "Crime"]
    pairs = confusion_pairs(y, y, CLASSES)
    assert pairs.empty
    assert set(pairs.columns) == {
        "rank",
        "true_category",
        "predicted_category",
        "count",
        "share_of_true_class",
        "share_of_predicted_class",
        "share_of_all_errors",
    }


def test_confusion_pairs_rank_is_sequential():
    y_true = ["State", "State", "Sports", "Crime", "Crime", "Crime"]
    y_pred = ["Sports", "Sports", "State", "State", "State", "Sports"]
    pairs = confusion_pairs(y_true, y_pred, CLASSES)
    assert list(pairs["rank"]) == list(range(1, len(pairs) + 1))
    assert pairs["count"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Report + CSV outputs
# ---------------------------------------------------------------------------

def _build_report_fixture():
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array(["State", "Sports", "Crime", "State"])
    y_pred = np.array(["Sports", "Crime", "Crime", "State"])
    metrics = compute_metrics(y, y_pred, CLASSES)
    df = collect_predictions(
        _StubModel(y_pred), X, y, CLASSES, row_ids=np.arange(len(y)), texts=TEXTS[:4]
    )
    pairs = confusion_pairs(y, y_pred, CLASSES)
    report = build_error_analysis_report(
        metrics=metrics,
        pairs_df=pairs,
        preds_df=df,
        class_names=CLASSES,
        split="validation",
        model_description="toy model",
        dataset_counts={"validation": len(y)},
    )
    return report, df, pairs


class _StubModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds)
        self.classes_ = np.array(list(CLASSES))

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        out = np.zeros((len(X), len(CLASSES)))
        for i, label in enumerate(self._preds):
            out[i, list(CLASSES).index(label)] = 0.7
        out += 0.1
        return out


def test_report_contains_error_rate_pairs_and_gujarati():
    report, _, _ = _build_report_fixture()
    assert "validation" in report
    assert "Error rate" in report or "error rate" in report
    assert "State" in report and "Sports" in report
    # The single State->Sports confusion must appear.
    assert "State" in report
    assert "Sports" in report
    # Gujarati headline text is carried into the report, not just row indices.
    assert TEXTS[0] in report


def test_misclassified_examples_selects_only_errors():
    _, df, _ = _build_report_fixture()
    mis = misclassified_examples(df)
    assert len(mis) == 2
    assert (~mis["correct"]).all()


# ---------------------------------------------------------------------------
# Step B investigation helpers
# ---------------------------------------------------------------------------

def _toy_predictions():
    df = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "text": [
                "રાજ્ય સરકારે નવી યોજના જાહેર કરી",
                "cricket match India",
                "ટૂંકી",
                "",
                "બોલિવૂડ ફિલ્મ રેકોર્ડ કમાણી",
            ],
            "true_category": ["State", "Sports", "Crime", "State", "Sports"],
            "predicted_category": ["Sports", "Sports", "State", "State", "Sports"],
            "correct": [False, True, False, True, True],
            "score_type": ["probability"] * 5,
            "predicted_score": [0.6, 0.9, 0.55, 0.8, 0.95],
            "runner_up_category": ["State", "Crime", "Crime", "Crime", "State"],
            "runner_up_score": [0.3, 0.05, 0.4, 0.1, 0.03],
            "margin": [0.3, 0.85, 0.15, 0.7, 0.92],
        }
    )
    return df


def test_enrich_token_features_adds_length_and_ascii_flags():
    enriched = enrich_token_features(_toy_predictions())
    assert "num_tokens" in enriched.columns
    assert "has_ascii" in enriched.columns
    assert enriched.loc[0, "num_tokens"] == 6
    assert bool(enriched.loc[1, "has_ascii"]) is True
    assert bool(enriched.loc[0, "has_ascii"]) is False
    assert enriched.loc[3, "num_tokens"] == 0


def test_error_breakdown_buckets_by_length_and_mixed_script():
    enriched = enrich_token_features(_toy_predictions())
    breakdown = error_breakdown(enriched)
    assert {"bucket", "support", "errors", "error_rate"}.issubset(breakdown.columns)
    length_rows = breakdown[breakdown["bucket"] != "mixed_script_ascii"]
    assert length_rows["support"].sum() == len(enriched)
    # Mixed-script row is correct (row 1), so its bucket error count is 0.
    mixed = breakdown[breakdown["bucket"] == "mixed_script_ascii"]
    assert int(mixed["errors"].iloc[0]) == 0


def test_class_vocabulary_overlap_reports_shared_tokens():
    texts_by_class = {
        "State": ["રાજ્ય સરકાર યોજના", "રાજ્ય પૂર"],
        "Sports": ["રાજ્ય ક્રિકેટ", "ક્રિકેટ મેચ"],
        "Crime": ["પોલીસ ધરપકડ"],
    }
    overlap = class_vocabulary_overlap(texts_by_class, min_len=2)
    assert "jaccard" in overlap.columns
    mask = (
        ((overlap["class_a"] == "State") & (overlap["class_b"] == "Sports"))
        | ((overlap["class_a"] == "Sports") & (overlap["class_b"] == "State"))
    )
    state_sports = overlap[mask].iloc[0]
    assert "રાજ્ય" in state_sports["shared_tokens"]
    # Every overlap value is in [0, 1] and sorted descending.
    assert overlap["jaccard"].between(0.0, 1.0).all()
    assert overlap["jaccard"].is_monotonic_decreasing


def test_build_error_cause_investigation_contains_gujarati_and_sections():
    enriched = enrich_token_features(_toy_predictions())
    breakdown = error_breakdown(enriched)
    pairs = confusion_pairs(
        enriched["true_category"], enriched["predicted_category"], CLASSES
    )
    overlap = class_vocabulary_overlap({"State": ["રાજ્ય"], "Sports": ["રાજ્ય"]})
    report = build_error_cause_investigation(
        enriched, breakdown, pairs, overlap, class_names=CLASSES, split="validation"
    )
    assert "length" in report.lower()
    assert "રાજ્ય સરકારે નવી યોજના જાહેર કરી" in report
    assert "Terminology" in report or "vocabulary" in report.lower()


def test_compare_to_recorded_detects_match_and_mismatch():
    recorded = {
        "accuracy": 0.8947, "macro_precision": 0.8952, "macro_recall": 0.8826,
        "macro_f1": 0.8883, "weighted_f1": 0.8942,
    }
    reproduced = dict(recorded)
    matches, md = compare_to_recorded(reproduced, recorded)
    assert matches is True
    assert "Within tolerance" in md

    off = dict(recorded)
    off["macro_f1"] = 0.5
    matches2, _ = compare_to_recorded(off, recorded)
    assert matches2 is False


def test_baseline_reproduction_report_mentions_verdict():
    recorded = {
        "accuracy": 0.8947, "macro_precision": 0.8952, "macro_recall": 0.8826,
        "macro_f1": 0.8883, "weighted_f1": 0.8942,
    }
    report = build_baseline_reproduction_report(
        reproduced=dict(recorded),
        recorded=recorded,
        model_description="skipgram / gradient_boosting",
        artifact_path="models/classifiers/selected_model.joblib",
    )
    assert "reproduc" in report.lower()
    assert "match" in report.lower()
    assert "0.8883" in report


def test_save_error_analysis_writes_utf8_files(tmp_path):
    report, df, pairs = _build_report_fixture()
    paths = save_error_analysis(
        tmp_path, report=report, predictions=df, pairs=pairs
    )
    for key in (
        "report",
        "predictions",
        "misclassified",
        "confusion_pairs",
    ):
        assert key in paths
        assert paths[key].exists() and paths[key].stat().st_size > 0

    text = paths["report"].read_text(encoding="utf-8")
    assert TEXTS[0] in text

    mis = pd.read_csv(paths["misclassified"], encoding="utf-8")
    assert len(mis) == 2
    assert TEXTS[0] in set(mis["text"])

    full = pd.read_csv(paths["predictions"], encoding="utf-8")
    assert len(full) == 4
