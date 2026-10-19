"""Supervised classifiers and the Phase 5 experiment pipeline.

Trains and evaluates four classifiers (Logistic Regression, Linear SVM,
Random Forest, Gradient Boosting) across multiple feature configurations,
ranks them by validation macro-F1, selects one configuration using validation
results only, then evaluates it once on the untouched test split.

Compatibility notes
-------------------
- Document vectors are dense; TF-IDF matrices are sparse.
- Logistic Regression and ``LinearSVC`` consume the sparse TF-IDF directly.
- Random Forest and Gradient Boosting cannot consume a large sparse matrix
  without excessive memory, so they are trained on an SVD-reduced dense view
  of the TF-IDF matrix (``config.CLASSIFIER_SVD_COMPONENTS``); this adaptation
  is recorded as a per-experiment warning.
- Gradient Boosting uses ``HistGradientBoostingClassifier`` (the histogram
  implementation), which is the practical option at ~79k rows.
"""

from __future__ import annotations

import json
import time
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import config
from src.evaluation import (
    ExperimentResult,
    classification_report_text,
    compute_metrics,
    confusion_matrix_array,
    experiments_to_dataframe,
    build_validation_comparison_table,
    result_to_json,
    save_confusion_matrix_plot,
    save_metric_comparison_plot,
)
from src.text_features import (
    EXPECTED_ROW_COUNTS,
    SPLITS,
    FeatureBundle,
    build_docvec_bundle,
    build_tfidf_bundle,
    labels_match_docvec,
    load_cleaned_texts,
    svd_reduce,
)

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIFIER_NAMES: Tuple[str, ...] = (
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "gradient_boosting",
)

# Estimators that can consume a sparse matrix efficiently.
SPARSE_CAPABLE: frozenset[str] = frozenset({"logistic_regression", "linear_svm"})


def supports_sparse(classifier_name: str) -> bool:
    """True when the classifier consumes sparse features directly."""
    return classifier_name in SPARSE_CAPABLE


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

def build_classifier(
    name: str,
    *,
    seed: int = config.RANDOM_SEED,
    **overrides,
):
    """Build a classifier by name with documented defaults and a fixed seed."""
    if name not in CLASSIFIER_NAMES:
        raise ValueError(
            f"Unknown classifier {name!r}; expected one of {CLASSIFIER_NAMES}"
        )
    params = dict(config.CLASSIFIER_DEFAULTS.get(name, {}))
    params.update(overrides)

    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "linear_svm":
        return LinearSVC(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    # gradient_boosting
    return HistGradientBoostingClassifier(random_state=seed, **params)


def _params_summary(
    name: str,
    overrides: Optional[Mapping[str, object]] = None,
    seed: int = config.RANDOM_SEED,
) -> str:
    params = dict(config.CLASSIFIER_DEFAULTS.get(name, {}))
    if overrides:
        params.update(overrides)
    params["seed"] = seed
    if not params:
        return "defaults"
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


# ---------------------------------------------------------------------------
# Trained experiment container
# ---------------------------------------------------------------------------

@dataclass
class TrainedExperiment:
    """A fitted classifier plus the feature bundle it was trained on."""

    result: ExperimentResult
    bundle: FeatureBundle
    model: object
    vectorizer: object
    svd: object
    feature_family: str
    model_name: Optional[str]


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    index: int,
    bundle: FeatureBundle,
    classifier_name: str,
    class_names: Sequence[str],
    *,
    seed: int = config.RANDOM_SEED,
    svd_components: Optional[int] = None,
    classifier_params: Optional[Mapping[str, object]] = None,
    use_svd_for_trees: bool = True,
) -> TrainedExperiment:
    """Train one classifier on ``bundle`` and evaluate it on validation."""
    X_train = bundle.X["train"]
    y_train = bundle.y["train"]
    X_val = bundle.X["validation"]
    y_val = bundle.y["validation"]

    experiment_warnings: List[str] = []
    svd = None
    reduced_dim: Optional[int] = None

    if bundle.is_sparse and not supports_sparse(classifier_name):
        if not use_svd_for_trees:
            raise ValueError(
                f"{classifier_name} cannot consume sparse {bundle.feature_type} "
                "features without excessive memory"
            )
        components = svd_components or config.CLASSIFIER_SVD_COMPONENTS
        svd, X_train = svd_reduce(X_train, components, seed=seed)
        X_val = svd.transform(X_val)
        reduced_dim = int(X_train.shape[1])
        experiment_warnings.append(
            f"{classifier_name} does not support sparse TF-IDF directly; "
            f"trained on a {reduced_dim}-component TruncatedSVD view fitted on "
            "training features only."
        )

    classifier = build_classifier(
        classifier_name, seed=seed, **(classifier_params or {})
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        start = time.perf_counter()
        classifier.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        eval_start = time.perf_counter()
        y_pred = classifier.predict(X_val)
        eval_seconds = time.perf_counter() - eval_start

    for warning in caught:
        message = f"{warning.category.__name__}: {warning.message}"
        if warning.category is ConvergenceWarning:
            experiment_warnings.append(message)

    metrics = compute_metrics(y_val, y_pred, class_names)

    result = ExperimentResult(
        index=index,
        feature_type=bundle.feature_type,
        feature_label=bundle.label,
        classifier=classifier_name,
        classifier_params=_params_summary(
            classifier_name, classifier_params, seed
        ),
        train_rows=int(len(y_train)),
        val_rows=int(len(y_val)),
        num_features=int(bundle.num_features),
        is_sparse=bool(bundle.is_sparse),
        train_seconds=float(train_seconds),
        eval_seconds=float(eval_seconds),
        metrics=metrics,
        reduced_dim=reduced_dim,
        warnings=tuple(experiment_warnings),
    )
    return TrainedExperiment(
        result=result,
        bundle=bundle,
        model=classifier,
        vectorizer=bundle.vectorizer,
        svd=svd,
        feature_family=bundle.feature_family,
        model_name=bundle.model_name,
    )


# ---------------------------------------------------------------------------
# All experiments
# ---------------------------------------------------------------------------

def run_all_experiments(
    bundles: Sequence[FeatureBundle],
    class_names: Sequence[str],
    *,
    seed: int = config.RANDOM_SEED,
    classifiers: Optional[Sequence[str]] = None,
    svd_components: Optional[int] = None,
) -> List[TrainedExperiment]:
    """Run every (feature configuration, classifier) experiment on validation."""
    selected_classifiers = list(classifiers or CLASSIFIER_NAMES)
    trained: List[TrainedExperiment] = []
    index = 0
    for bundle in bundles:
        for classifier_name in selected_classifiers:
            index += 1
            trained.append(
                run_experiment(
                    index,
                    bundle,
                    classifier_name,
                    class_names,
                    seed=seed,
                    svd_components=svd_components,
                )
            )
    return trained


# ---------------------------------------------------------------------------
# Selection and test evaluation
# ---------------------------------------------------------------------------

def select_best(results: Sequence[ExperimentResult]) -> ExperimentResult:
    """Return the best result by validation macro-F1 (tie-break accuracy)."""
    if not results:
        raise ValueError("No experiment results to select from")
    return max(results, key=lambda r: (r.macro_f1, r.accuracy))


def evaluate_selected_on_test(
    trained: TrainedExperiment,
    class_names: Sequence[str],
) -> Tuple[Dict[str, object], str, np.ndarray, np.ndarray]:
    """Evaluate the frozen selected experiment on the untouched test split."""
    X_test = trained.bundle.X["test"]
    y_test = trained.bundle.y["test"]
    if trained.svd is not None:
        X_test = trained.svd.transform(X_test)
    y_pred = trained.model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, class_names)
    report = classification_report_text(y_test, y_pred, class_names)
    cm = confusion_matrix_array(y_test, y_pred, class_names)
    return metrics, report, cm, np.asarray(y_pred)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_classifier_artifact(
    path: Path,
    model: object,
    *,
    feature_type: str,
    feature_family: str,
    class_names: Sequence[str],
    vectorizer: object = None,
    svd: object = None,
    model_name: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Persist a classifier together with its vectorizer/SVD and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "vectorizer": vectorizer,
        "svd": svd,
        "feature_type": feature_type,
        "feature_family": feature_family,
        "model_name": model_name,
        "class_names": [str(c) for c in class_names],
        "metadata": dict(metadata or {}),
    }
    joblib.dump(payload, path)
    return path


def load_classifier_artifact(path: Path) -> Dict[str, object]:
    """Load a classifier payload saved by :func:`save_classifier_artifact`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier artifact not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Feature bundle construction
# ---------------------------------------------------------------------------

def feature_configuration_notes(
    feature_keys: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Reasons for configured-but-disabled feature configurations."""
    notes: Dict[str, str] = {}
    for entry in config.FEATURE_CONFIGURATIONS:
        if feature_keys is not None and entry["key"] in feature_keys:
            continue
        if not entry.get("enabled", True):
            notes[entry["key"]] = str(
                entry.get("disabled_reason", "disabled")
            )
    return notes


def build_feature_bundles(
    *,
    class_names: Optional[Sequence[str]] = None,
    feature_keys: Optional[Sequence[str]] = None,
    cleaned: Optional[Mapping[str, Tuple[List[str], np.ndarray]]] = None,
) -> List[FeatureBundle]:
    """Build every configured feature bundle, verifying split alignment.

    Document-vector bundles are checked positionally against the cleaned
    split labels so a drifted artifact cannot silently enter a comparison.
    """
    class_names = tuple(class_names or config.CATEGORY_LABELS)
    if feature_keys is None:
        entries = [
            entry
            for entry in config.FEATURE_CONFIGURATIONS
            if entry.get("enabled", True)
        ]
    else:
        entries = [
            entry
            for entry in config.FEATURE_CONFIGURATIONS
            if entry["key"] in feature_keys
        ]
    if cleaned is None:
        cleaned = load_cleaned_texts(expected_counts=EXPECTED_ROW_COUNTS)

    texts = {split: cleaned[split][0] for split in SPLITS}
    labels = {split: cleaned[split][1] for split in SPLITS}

    bundles: List[FeatureBundle] = []
    for entry in entries:
        if entry["kind"] == "docvec":
            bundle = build_docvec_bundle(entry, class_names=class_names)
            for split in SPLITS:
                expected = labels[split]
                feature = _feature_matrix_for_label_check(bundle, split, class_names)
                if not labels_match_docvec(expected, feature):
                    raise ValueError(
                        f"{split}: cleaned labels do not align positionally "
                        f"with the {bundle.feature_type} document vectors"
                    )
        else:
            bundle = build_tfidf_bundle(
                entry, texts, labels, class_names=class_names
            )
        bundles.append(bundle)
    return bundles


def _feature_matrix_for_label_check(bundle, split, class_names):
    from src.text_features import FeatureMatrix

    return FeatureMatrix(
        X=bundle.X[split],
        y=bundle.y[split],
        row_ids=bundle.row_ids[split],
        feature_type=bundle.feature_type,
        num_features=bundle.num_features,
        is_sparse=bundle.is_sparse,
        class_names=tuple(class_names),
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _confusion_matrix_text(cm: np.ndarray, class_names: Sequence[str]) -> str:
    width = max(len(str(c)) for c in class_names) + 1
    lines = ["", "  Confusion matrix (rows = true, columns = predicted)"]
    header = " " * (width + 1) + " ".join(
        f"{str(c):>{max(6, len(str(c)))}}" for c in class_names
    )
    lines.append(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{int(v):>{max(6, len(str(c)))}}" for c, v in zip(class_names, cm[i]))
        lines.append(f"  {str(name):>{width}} {row}")
    return "\n".join(lines)


def _per_class_text(metrics: Mapping[str, object], class_names: Sequence[str]) -> str:
    lines = [
        f"  {'category':<16}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}"
    ]
    per_class = metrics["per_class"]  # type: ignore[index]
    for name in class_names:
        pc = per_class[name]  # type: ignore[index]
        lines.append(
            f"  {name:<16}{pc['precision']:>12.4f}{pc['recall']:>10.4f}"
            f"{pc['f1']:>10.4f}{pc['support']:>10}"
        )
    return "\n".join(lines)


def build_classification_report(
    trained: Sequence[TrainedExperiment],
    best: TrainedExperiment,
    class_names: Sequence[str],
    *,
    feature_notes: Optional[Mapping[str, str]] = None,
) -> str:
    """Human-readable validation comparison + selection report."""
    results = [t.result for t in trained]
    df = experiments_to_dataframe(results)
    sep = "-" * 72
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GUJARATI NEWS CLASSIFICATION REPORT (PHASE 5, VALIDATION)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. SCOPE AND DATA-LEAKAGE PREVENTION")
    lines.append(sep)
    lines.append(f"  Text column:     {config.TEXT_COLUMN!r}")
    lines.append(f"  Label column:    {config.LABEL_COLUMN!r}")
    lines.append(f"  Categories:      {', '.join(class_names)}")
    lines.append("  Train rows:      %d" % EXPECTED_ROW_COUNTS["train"])
    lines.append("  Validation rows: %d (used for model selection)" % EXPECTED_ROW_COUNTS["validation"])
    lines.append("  Test rows:       %d (untouched during selection)" % EXPECTED_ROW_COUNTS["test"])
    lines.append("  TF-IDF:          fitted on training text only")
    lines.append("  Word2Vec:        loaded read-only, never retrained")
    lines.append("  Original CSVs:   never modified")
    lines.append("  Primary metric:  macro-F1 (secondary: accuracy)")
    lines.append("")
    lines.append("2. FEATURE CONFIGURATIONS")
    lines.append(sep)
    seen = set()
    for t in trained:
        b = t.bundle
        if b.key in seen:
            continue
        seen.add(b.key)
        if b.feature_family == "docvec":
            lines.append(
                f"  {b.key:12s}: {b.label}  dims={b.num_features}"
            )
        else:
            vocab = len(b.vectorizer.vocabulary_)
            lines.append(
                f"  {b.key:12s}: {b.label}  analyzer={b.vectorizer.analyzer}  "
                f"ngram={getattr(b.vectorizer, 'ngram_range', None)}  "
                f"vocab={vocab}"
            )
    if feature_notes:
        for key, note in feature_notes.items():
            lines.append(f"  NOTE {key}: {note}")
    lines.append("")
    lines.append("3. EXPERIMENT INVENTORY")
    lines.append(sep)
    lines.append(
        df[
            [
                "index",
                "feature_type",
                "classifier",
                "num_features",
                "reduced_dim",
                "train_rows",
                "val_rows",
                "train_seconds",
                "eval_seconds",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("4. VALIDATION COMPARISON (sorted by macro-F1)")
    lines.append(sep)
    lines.append(build_validation_comparison_table(results))
    lines.append("")
    lines.append("5. WARNINGS AND LIMITATIONS PER EXPERIMENT")
    lines.append(sep)
    any_warnings = False
    for t in trained:
        if t.result.warnings:
            any_warnings = True
            lines.append(f"  exp {t.result.index:>3} ({t.result.feature_type} / {t.result.classifier}):")
            for w in t.result.warnings:
                lines.append(f"      - {w}")
    if not any_warnings:
        lines.append("  none")
    lines.append("")
    lines.append("6. SELECTED CONFIGURATION (by validation macro-F1)")
    lines.append(sep)
    lines.append(f"  Experiment index: {best.result.index}")
    lines.append(f"  Feature:          {best.result.feature_label} ({best.result.feature_type})")
    lines.append(f"  Classifier:       {best.result.classifier}")
    lines.append(f"  Parameters:       {best.result.classifier_params}")
    lines.append(f"  Validation macro-F1:    {best.result.macro_f1:.4f}")
    lines.append(f"  Validation accuracy:    {best.result.accuracy:.4f}")
    lines.append(f"  Validation weighted F1: {best.result.weighted_f1:.4f}")
    lines.append("")
    lines.append("  Per-class validation metrics for the selected configuration:")
    lines.append(_per_class_text(best.result.metrics, class_names))
    lines.append("")
    lines.append(
        "  Selection note: the winner is the best among the evaluated "
        "configurations on this dataset and protocol, not a universal claim."
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF VALIDATION REPORT")
    lines.append("=" * 72)
    return "\n".join(lines)


def build_test_report(
    best: TrainedExperiment,
    test_metrics: Mapping[str, object],
    test_report_text: str,
    cm: np.ndarray,
    class_names: Sequence[str],
) -> str:
    """Human-readable test-split evaluation report for the selected model."""
    sep = "-" * 72
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GUJARATI NEWS CLASSIFICATION - TEST EVALUATION (PHASE 5)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Frozen selected configuration (chosen on validation only)")
    lines.append(sep)
    lines.append(f"  Experiment index: {best.result.index}")
    lines.append(f"  Feature:          {best.result.feature_label} ({best.result.feature_type})")
    lines.append(f"  Classifier:       {best.result.classifier}")
    lines.append(f"  Parameters:       {best.result.classifier_params}")
    lines.append("")
    lines.append("Test metrics")
    lines.append(sep)
    lines.append(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
    lines.append(f"  Macro precision:   {test_metrics['macro_precision']:.4f}")
    lines.append(f"  Macro recall:      {test_metrics['macro_recall']:.4f}")
    lines.append(f"  Macro F1:          {test_metrics['macro_f1']:.4f}")
    lines.append(f"  Weighted F1:       {test_metrics['weighted_f1']:.4f}")
    lines.append(f"  Test rows:         {test_metrics['support']}")
    lines.append("")
    lines.append("Per-class test metrics")
    lines.append(sep)
    lines.append(_per_class_text(test_metrics, class_names))
    lines.append("")
    lines.append("Sklearn classification report (test)")
    lines.append(sep)
    lines.append(test_report_text)
    lines.append(_confusion_matrix_text(cm, class_names))
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF TEST REPORT")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_classification(
    *,
    seed: int = config.RANDOM_SEED,
    feature_keys: Optional[Sequence[str]] = None,
    classifiers: Optional[Sequence[str]] = None,
    svd_components: Optional[int] = None,
) -> Dict[str, object]:
    """Run the full Phase 5 training + evaluation pipeline.

    Steps: build leakage-safe features, run every experiment on validation,
    persist artifacts and tracking files, select by validation macro-F1, then
    evaluate the frozen selection on the test split and write reports.
    """
    config.validate_paths()
    class_names = tuple(config.CATEGORY_LABELS)

    cleaned = load_cleaned_texts(expected_counts=EXPECTED_ROW_COUNTS)
    bundles = build_feature_bundles(
        class_names=class_names, feature_keys=feature_keys, cleaned=cleaned
    )

    trained = run_all_experiments(
        bundles,
        class_names,
        seed=seed,
        classifiers=classifiers,
        svd_components=svd_components,
    )

    # Persist each trained classifier (with its vectorizer/SVD).
    config.CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    for t in trained:
        filename = (
            f"exp{t.result.index:03d}_"
            f"{t.result.feature_type.replace(':', '_')}_"
            f"{t.result.classifier}.joblib"
        )
        path = config.CLASSIFIERS_DIR / filename
        save_classifier_artifact(
            path,
            t.model,
            feature_type=t.result.feature_type,
            feature_family=t.feature_family,
            class_names=class_names,
            vectorizer=t.vectorizer,
            svd=t.svd,
            model_name=t.model_name,
            metadata={
                "experiment_index": t.result.index,
                "feature_label": t.result.feature_label,
                "classifier": t.result.classifier,
                "classifier_params": t.result.classifier_params,
                "num_features": t.result.num_features,
                "reduced_dim": t.result.reduced_dim,
                "validation_metrics": t.result.metrics,
                "warnings": list(t.result.warnings),
                "seed": seed,
            },
        )
        t.result.artifact_path = str(path)

    # Experiment-tracking outputs.
    results = [t.result for t in trained]
    df = experiments_to_dataframe(results)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.REPORTS_DIR / config.CLASSIFICATION_RESULTS_CSV
    json_path = config.REPORTS_DIR / config.CLASSIFICATION_RESULTS_JSON
    df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps([result_to_json(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    table = build_validation_comparison_table(results)
    table_path = config.REPORTS_DIR / config.CLASSIFICATION_VALIDATION_TABLE_FILENAME
    table_path.write_text(table + "\n", encoding="utf-8")
    plot_path = save_metric_comparison_plot(
        df, config.REPORTS_DIR / config.CLASSIFICATION_COMPARISON_PLOT_FILENAME
    )

    # Model selection on validation only.
    best_result = select_best(results)
    best_trained = next(t for t in trained if t.result.index == best_result.index)

    # Confusion matrices for the leading validation models.
    leading = sorted(
        results, key=lambda r: (r.macro_f1, r.accuracy), reverse=True
    )[:3]
    leading_cm_paths: List[str] = []
    for r in leading:
        t = next(x for x in trained if x.result.index == r.index)
        X_val = t.bundle.X["validation"]
        if t.svd is not None:
            X_val = t.svd.transform(X_val)
        y_pred_val = t.model.predict(X_val)
        cm_val = confusion_matrix_array(
            t.bundle.y["validation"], y_pred_val, class_names
        )
        cm_path = save_confusion_matrix_plot(
            cm_val,
            class_names,
            config.REPORTS_DIR
            / f"classification_validation_confusion_exp{r.index:03d}.png",
            title=(
                f"Validation confusion matrix - exp {r.index} - "
                f"{r.feature_type} / {r.classifier} (macro-F1={r.macro_f1:.4f})"
            ),
        )
        leading_cm_paths.append(str(cm_path))

    # Persist the selected model under a stable name for inference.
    selected_path = config.CLASSIFIERS_DIR / "selected_model.joblib"
    save_classifier_artifact(
        selected_path,
        best_trained.model,
        feature_type=best_result.feature_type,
        feature_family=best_trained.feature_family,
        class_names=class_names,
        vectorizer=best_trained.vectorizer,
        svd=best_trained.svd,
        model_name=best_trained.model_name,
        metadata={
            "experiment_index": best_result.index,
            "feature_label": best_result.feature_label,
            "classifier": best_result.classifier,
            "classifier_params": best_result.classifier_params,
            "num_features": best_result.num_features,
            "reduced_dim": best_result.reduced_dim,
            "validation_metrics": best_result.metrics,
            "seed": seed,
        },
    )

    report = build_classification_report(
        trained,
        best_trained,
        class_names,
        feature_notes=feature_configuration_notes(feature_keys),
    )
    report_path = config.REPORTS_DIR / config.CLASSIFICATION_REPORT_FILENAME
    report_path.write_text(report, encoding="utf-8")

    # Frozen test evaluation.
    test_metrics, test_report_text, cm, _ = evaluate_selected_on_test(
        best_trained, class_names
    )
    cm_plot = save_confusion_matrix_plot(
        cm,
        class_names,
        config.REPORTS_DIR / config.CLASSIFICATION_CONFUSION_PLOT_FILENAME,
        title=(
            f"Test confusion matrix - {best_result.feature_type} / "
            f"{best_result.classifier}"
        ),
    )
    test_report = build_test_report(
        best_trained, test_metrics, test_report_text, cm, class_names
    )
    test_report_path = (
        config.REPORTS_DIR / config.CLASSIFICATION_TEST_REPORT_FILENAME
    )
    test_report_path.write_text(test_report, encoding="utf-8")

    return {
        "trained": trained,
        "results": results,
        "dataframe": df,
        "best": best_trained,
        "test_metrics": test_metrics,
        "test_report": test_report,
        "confusion_matrix": cm,
        "paths": {
            "csv": csv_path,
            "json": json_path,
            "validation_table": table_path,
            "comparison_plot": plot_path,
            "validation_report": report_path,
            "test_report": test_report_path,
            "confusion_plot": cm_plot,
            "validation_confusion_plots": leading_cm_paths,
            "selected_model": selected_path,
        },
    }
