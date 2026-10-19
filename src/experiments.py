"""Targeted Phase 7 experiments for the Gujarati news classifier.

Reuses the Phase 5 machinery end to end rather than duplicating it:

- Feature bundles come from :mod:`src.text_features` (read-only document vectors
  and TF-IDF fitted on training text only).
- Estimators are trained/evaluated by :func:`src.classification.run_experiment`,
  so every candidate gets the same metrics, warnings, and timing.
- Artifacts are persisted with :func:`src.classification.save_classifier_artifact`
  into a dedicated ``models/experiments/`` directory; the Phase 5 selected model
  and reports are never overwritten.

The module also provides probability-quality/calibration analysis and the
unified validation comparison used to decide (with a pre-declared +0.005
macro-F1 threshold) whether to retain the existing selected model.

Nothing here touches the test split.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from src.classification import (
    load_classifier_artifact,
    run_experiment,
    save_classifier_artifact,
)
from src.evaluation import (
    ExperimentResult,
    experiments_to_dataframe,
    result_to_json,
)
from src.text_features import (
    EXPECTED_ROW_COUNTS,
    SPLITS,
    FeatureBundle,
    TfidfConfig,
    build_docvec_bundle,
    build_tfidf_features,
    load_cleaned_texts,
)

# ---------------------------------------------------------------------------
# Experiment specification
# ---------------------------------------------------------------------------

HISTGB_BASELINE: Mapping[str, object] = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.0,
}

REPLACEMENT_THRESHOLD = float(getattr(config, "REPLACEMENT_THRESHOLD", 0.005))


@dataclass
class ExperimentSpec:
    """One (feature configuration, estimator, hyperparameters) experiment."""

    key: str
    feature_key: str
    feature_label: str
    kind: str                              # "docvec" or "tfidf"
    classifier: str
    params: Mapping[str, object] = field(default_factory=dict)
    ngram_range: Tuple[int, int] = (1, 1)
    notes: str = ""


def _histgb(**overrides) -> Dict[str, object]:
    params = dict(HISTGB_BASELINE)
    params.update(overrides)
    return params


def default_skipgram_configs() -> Tuple[ExperimentSpec, ...]:
    """A small, justified search around the selected Skip-gram baseline."""
    label = "Skip-gram doc vectors (100d)"

    def spec(key, params, notes=""):
        return ExperimentSpec(
            key=key, feature_key="skipgram", feature_label=label,
            kind="docvec", classifier="gradient_boosting", params=params,
            notes=notes,
        )

    return (
        spec("sg_histgb_control", _histgb(),
             "Phase 5 baseline control (max_iter=200, lr=0.1, leaves=31, l2=0)."),
        spec("sg_histgb_iter100", _histgb(max_iter=100),
             "Fewer boosting iterations."),
        spec("sg_histgb_iter300", _histgb(max_iter=300),
             "More boosting iterations."),
        spec("sg_histgb_lr005", _histgb(learning_rate=0.05),
             "Slower learning rate."),
        spec("sg_histgb_leaves15", _histgb(max_leaf_nodes=15),
             "Simpler trees (less capacity)."),
        spec("sg_histgb_leaves63", _histgb(max_leaf_nodes=63),
             "More complex trees."),
        spec("sg_histgb_l2_1", _histgb(l2_regularization=1.0),
             "Explicit L2 regularisation."),
        spec("sg_histgb_lr005_leaves63", _histgb(learning_rate=0.05, max_leaf_nodes=63),
             "Slower rate with more complex trees."),
    )


def default_cbow_configs() -> Tuple[ExperimentSpec, ...]:
    """A small comparable experiment on the existing CBOW document vectors."""
    label = "CBOW doc vectors (100d)"

    def spec(key, params, notes=""):
        return ExperimentSpec(
            key=key, feature_key="cbow", feature_label=label,
            kind="docvec", classifier="gradient_boosting", params=params,
            notes=notes,
        )

    return (
        spec("cbow_histgb_control", _histgb(),
             "CBOW baseline control for a like-for-like representation check."),
        spec("cbow_histgb_lr005", _histgb(learning_rate=0.05),
             "Slower learning rate."),
        spec("cbow_histgb_iter300_leaves63_l2_1",
             _histgb(max_iter=300, max_leaf_nodes=63, l2_regularization=1.0),
             "Larger capacity with regularisation."),
    )


def default_tfidf_word_configs() -> Tuple[ExperimentSpec, ...]:
    """Word unigram vs unigram+bigram TF-IDF with LogReg and Linear SVM."""
    logreg = {"max_iter": 1000, "solver": "lbfgs"}
    svm = {"C": 1.0}
    return (
        ExperimentSpec("tfidf_uni_logreg", "tfidf_word_uni",
                       "TF-IDF word unigram (1,1)", "tfidf",
                       "logistic_regression", logreg, ngram_range=(1, 1),
                       notes="Unigram word TF-IDF + logistic regression."),
        ExperimentSpec("tfidf_uni_svm", "tfidf_word_uni",
                       "TF-IDF word unigram (1,1)", "tfidf",
                       "linear_svm", svm, ngram_range=(1, 1),
                       notes="Unigram word TF-IDF + linear SVM."),
        ExperimentSpec("tfidf_12_logreg", "tfidf_word_12",
                       "TF-IDF word unigram+bigram (1,2)", "tfidf",
                       "logistic_regression", logreg, ngram_range=(1, 2),
                       notes="Word TF-IDF + logistic regression."),
        ExperimentSpec("tfidf_12_svm", "tfidf_word_12",
                       "TF-IDF word unigram+bigram (1,2)", "tfidf",
                       "linear_svm", svm, ngram_range=(1, 2),
                       notes="Word TF-IDF + linear SVM."),
    )


# ---------------------------------------------------------------------------
# Feature bundles
# ---------------------------------------------------------------------------

def _docvec_entry(feature_key: str) -> Mapping[str, object]:
    for entry in config.FEATURE_CONFIGURATIONS:
        if entry["key"] == feature_key and entry.get("kind") == "docvec":
            return entry
    raise KeyError(f"No docvec feature configuration named {feature_key!r}")


def build_tfidf_word_bundle(
    *,
    ngram_range: Tuple[int, int],
    cleaned: Mapping[str, Sequence[str]],
    labels: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    key: str,
    label: str,
    fit_split: str = "train",
) -> FeatureBundle:
    """Fit a word TF-IDF (given n-gram range) on train text only."""
    params = dict(config.TFIDF_WORD_PARAMS)
    params["ngram_range"] = tuple(ngram_range)
    tfidf_config = TfidfConfig.from_params(params)
    vectorizer, matrices = build_tfidf_features(
        cleaned, tfidf_config, fit_split=fit_split
    )
    bundle = FeatureBundle(
        key=key,
        label=label,
        feature_type=key,
        feature_family="tfidf",
        is_sparse=True,
        num_features=int(matrices[fit_split].shape[1]),
        class_names=tuple(class_names),
        vectorizer=vectorizer,
    )
    for split in SPLITS:
        bundle.X[split] = matrices[split]
        bundle.y[split] = np.asarray(labels[split]).astype(str)
        bundle.row_ids[split] = np.arange(len(bundle.y[split]), dtype=np.int64)
    return bundle


# ---------------------------------------------------------------------------
# Running experiments
# ---------------------------------------------------------------------------

def run_spec(
    index: int,
    spec: ExperimentSpec,
    bundles: Mapping[str, FeatureBundle],
    class_names: Sequence[str],
    *,
    seed: int = config.RANDOM_SEED,
    artifact_dir: Optional[Path] = None,
) -> Tuple[object, Dict[str, object]]:
    """Train one experiment, persist its artifact, and return (trained, extra)."""
    if spec.feature_key not in bundles:
        raise KeyError(f"Feature bundle {spec.feature_key!r} is not available")
    bundle = bundles[spec.feature_key]

    trained = run_experiment(
        index,
        bundle,
        spec.classifier,
        class_names,
        seed=seed,
        classifier_params=dict(spec.params),
    )

    n_iter_raw = getattr(trained.model, "n_iter_", None)
    n_iter: Optional[int] = None
    if n_iter_raw is not None:
        arr = np.asarray(n_iter_raw)
        n_iter = int(arr.max()) if arr.size else None
    extra: Dict[str, object] = {
        "key": spec.key,
        "feature_key": spec.feature_key,
        "notes": spec.notes,
        "n_iter": n_iter,
        "seed": seed,
    }

    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        path = artifact_dir / f"exp{index:03d}_{spec.key}.joblib"
        save_classifier_artifact(
            path,
            trained.model,
            feature_type=trained.result.feature_type,
            feature_family=trained.feature_family,
            class_names=class_names,
            vectorizer=trained.vectorizer,
            svd=trained.svd,
            model_name=trained.model_name,
            metadata={
                "phase": 7,
                "key": spec.key,
                "feature_key": spec.feature_key,
                "feature_label": spec.feature_label,
                "classifier": spec.classifier,
                "classifier_params": dict(spec.params),
                "params_summary": trained.result.classifier_params,
                "notes": spec.notes,
                "n_iter": extra["n_iter"],
                "validation_metrics": trained.result.metrics,
                "warnings": list(trained.result.warnings),
                "seed": seed,
            },
        )
        trained.result.artifact_path = str(path)
        extra["artifact_path"] = str(path)

    return trained, extra


def run_experiments(
    specs: Sequence[ExperimentSpec],
    bundles: Mapping[str, FeatureBundle],
    class_names: Sequence[str],
    *,
    seed: int = config.RANDOM_SEED,
    artifact_dir: Optional[Path] = None,
    start_index: int = 100,
    on_result: Optional[Callable[[object, Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """Run every spec on the validation split, recording failures explicitly."""
    trained_list: List[object] = []
    extras: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for offset, spec in enumerate(specs):
        index = start_index + offset
        try:
            trained, extra = run_spec(
                index, spec, bundles, class_names, seed=seed,
                artifact_dir=artifact_dir,
            )
        except Exception as exc:  # noqa: BLE001 - failures are recorded, not hidden
            failures.append(
                {
                    "index": index,
                    "key": spec.key,
                    "feature_key": spec.feature_key,
                    "classifier": spec.classifier,
                    "params": dict(spec.params),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            if on_result is not None:
                on_result(None, {"key": spec.key, "index": index, "failure": True})
            continue

        trained_list.append(trained)
        extras.append(extra)
        if on_result is not None:
            on_result(trained, extra)

    return {"trained": trained_list, "extras": extras, "failures": failures}


# ---------------------------------------------------------------------------
# Load original Phase 5 results & unify comparison
# ---------------------------------------------------------------------------

def load_original_results(json_path: Path) -> List[ExperimentResult]:
    """Load the recorded Phase 5 validation experiments as results."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Original experiment results not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results: List[ExperimentResult] = []
    for entry in payload:
        data = dict(entry)
        data["warnings"] = tuple(data.get("warnings", []) or [])
        results.append(ExperimentResult(**data))
    return results


def unified_comparison_dataframe(
    original_results: Sequence[ExperimentResult],
    new_results: Sequence[ExperimentResult],
) -> pd.DataFrame:
    """One ranked table tagging Phase 5 baselines vs Phase 7 experiments."""
    frames = []
    if original_results:
        orig = experiments_to_dataframe(list(original_results))
        orig["source"] = "phase5_original"
        frames.append(orig)
    if new_results:
        new = experiments_to_dataframe(list(new_results))
        new["source"] = "phase7_experiment"
        frames.append(new)

    if not frames:
        return pd.DataFrame(
            columns=["rank", "index", "feature_type", "classifier",
                     "macro_f1", "accuracy", "source"]
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    return df


def select_replacement(
    baseline_macro_f1: float,
    candidate_macro_f1: float,
    *,
    threshold: float = REPLACEMENT_THRESHOLD,
) -> bool:
    """True when a candidate clears the minimum meaningful improvement."""
    return float(candidate_macro_f1) - float(baseline_macro_f1) >= threshold - 1e-9


# ---------------------------------------------------------------------------
# Probability quality & calibration
# ---------------------------------------------------------------------------

def _ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    if total == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (
            confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(ece)


def probability_quality(
    model,
    X,
    y_true: Sequence[str],
    class_names: Sequence[str],
    *,
    n_bins: int = 10,
) -> Dict[str, object]:
    """Log-loss, multiclass Brier score, and ECE for ``predict_proba`` output.

    Returns ``{"available": False}`` when the estimator exposes no
    ``predict_proba`` (no probabilities to evaluate or fabricate).
    """
    if not hasattr(model, "predict_proba"):
        return {"available": False, "reason": "estimator has no predict_proba"}

    from sklearn.metrics import log_loss

    classes = [str(c) for c in getattr(model, "classes_", class_names)]
    proba = np.asarray(model.predict_proba(X), dtype=float)
    y_true_arr = np.asarray(list(y_true), dtype=str)
    class_index = {c: i for i, c in enumerate(classes)}
    missing = sorted(set(y_true_arr.tolist()) - set(classes))
    if missing:
        raise ValueError(f"y_true contains labels absent from model.classes_: {missing}")

    y_idx = np.array([class_index[label] for label in y_true_arr], dtype=int)
    n = len(y_idx)
    onehot = np.zeros_like(proba)
    onehot[np.arange(n), y_idx] = 1.0

    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    # sklearn's log_loss expects columns ordered lexicographically by label.
    order = np.argsort(classes)
    sorted_classes = [classes[i] for i in order]
    ll = float(log_loss(y_true_arr, proba[:, order], labels=sorted_classes))
    confidences = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_idx)
    return {
        "available": True,
        "n_samples": int(n),
        "n_bins": int(n_bins),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "log_loss": ll,
        "brier": brier,
        "ece": _ece(confidences, correct, n_bins=n_bins),
    }


def calibrated_variant(
    model,
    X_train,
    y_train: Sequence[str],
    X_val,
    y_val: Sequence[str],
    class_names: Sequence[str],
    *,
    method: str = "sigmoid",
    cv: int = 3,
) -> Dict[str, object]:
    """Fit a cross-validated probabilistic calibration on training data only.

    The calibrated estimator is fitted on ``X_train``/``y_train`` (never the
    test split) and evaluated on the validation split. Returns the calibration
    quality plus the fit duration; failures are reported, not hidden.
    """
    from sklearn.calibration import CalibratedClassifierCV

    started = time.perf_counter()
    try:
        calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
        calibrated.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "method": method,
            "cv": cv,
            "error": f"{type(exc).__name__}: {exc}",
            "fit_seconds": time.perf_counter() - started,
        }
    fit_seconds = time.perf_counter() - started
    quality = probability_quality(calibrated, X_val, y_val, class_names)
    quality.update({"method": method, "cv": cv, "fit_seconds": fit_seconds})
    return quality


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _comparison_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| Rank | Exp | Source | Feature | Classifier | Macro-F1 | Accuracy | "
        "Weighted-F1 | Macro-P | Macro-R | Train (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {int(row['index'])} | {row['source']} "
            f"| {row['feature_label']} | {row['classifier']} "
            f"| {float(row['macro_f1']):.4f} | {float(row['accuracy']):.4f} "
            f"| {float(row['weighted_f1']):.4f} | {float(row['macro_precision']):.4f} "
            f"| {float(row['macro_recall']):.4f} | {float(row['train_seconds']):.2f} |"
        )
    return "\n".join(lines)


def _save_comparison_plot(df: pd.DataFrame, path: Path, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values("macro_f1", ascending=True).reset_index(drop=True)
    labels = [
        f"{int(r['index'])}: {r['feature_label']} / {r['classifier']}"
        for _, r in plot_df.iterrows()
    ]
    colors = [
        "#0f766e" if r["source"] == "phase7_experiment" else "#9aa7b8"
        for _, r in plot_df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.42 * len(plot_df))))
    ax.barh(labels, plot_df["macro_f1"], color=colors)
    ax.set_xlabel("Validation macro-F1")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    for y, value in enumerate(plot_df["macro_f1"]):
        ax.text(value + 0.002, y, f"{value:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_experiment_outputs(
    out_dir: Path,
    df: pd.DataFrame,
    *,
    failures: Sequence[Mapping[str, object]] = (),
    results: Optional[Sequence[ExperimentResult]] = None,
    title: str = "Phase 7 validation comparison",
) -> Dict[str, Path]:
    """Write the comparison CSV/JSON/markdown/plot and the failures log."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "validation_comparison.csv"
    json_path = out_dir / "validation_experiments_detail.json"
    md_path = out_dir / "validation_comparison.md"
    plot_path = out_dir / "validation_comparison_macro_f1.png"
    failures_path = out_dir / "experiment_failures.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    payload = [result_to_json(r) for r in (results or [])]
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(
        f"# {title}\n\n" + _comparison_markdown(df) + "\n", encoding="utf-8"
    )
    _save_comparison_plot(df, plot_path, title)
    failures_path.write_text(
        json.dumps(list(failures), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": md_path,
        "plot": plot_path,
        "failures": failures_path,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_phase7_experiments(
    *,
    artifact_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    seed: int = config.RANDOM_SEED,
    run_calibration: bool = True,
    progress: bool = True,
) -> Dict[str, object]:
    """Run the Step D/E experiments and write the Step F comparison.

    Uses validation data only, preserves the Phase 5 selected model, and saves
    every candidate under ``models/experiments/``.
    """
    config.validate_paths()
    artifact_dir = Path(artifact_dir or (config.MODELS_DIR / "experiments"))
    out_dir = Path(out_dir or (config.REPORTS_DIR / "experiments"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = tuple(config.CATEGORY_LABELS)

    # ---- features (read-only / train-fit only) --------------------------
    cleaned = load_cleaned_texts(expected_counts=EXPECTED_ROW_COUNTS)
    texts = {split: cleaned[split][0] for split in SPLITS}
    labels = {split: cleaned[split][1] for split in SPLITS}

    bundles: Dict[str, FeatureBundle] = {}
    for key in ("skipgram", "cbow"):
        bundles[key] = build_docvec_bundle(
            _docvec_entry(key), class_names=class_names
        )
    bundles["tfidf_word_uni"] = build_tfidf_word_bundle(
        ngram_range=(1, 1), cleaned=texts, labels=labels,
        class_names=class_names, key="tfidf_word_uni",
        label="TF-IDF word unigram (1,1)",
    )
    bundles["tfidf_word_12"] = build_tfidf_word_bundle(
        ngram_range=(1, 2), cleaned=texts, labels=labels,
        class_names=class_names, key="tfidf_word_12",
        label="TF-IDF word unigram+bigram (1,2)",
    )

    # ---- specs ----------------------------------------------------------
    specs: List[ExperimentSpec] = list(default_skipgram_configs())
    specs += list(default_cbow_configs())
    specs += list(default_tfidf_word_configs())

    # ---- run, writing incremental progress ------------------------------
    progress_path = out_dir / "experiment_progress.json"
    collected: List[Dict[str, object]] = []

    def _on_result(trained, extra):
        if trained is not None:
            collected.append(
                {
                    "index": trained.result.index,
                    "key": extra.get("key"),
                    "macro_f1": trained.result.macro_f1,
                    "accuracy": trained.result.accuracy,
                }
            )
        progress_path.write_text(
            json.dumps({"completed": collected}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if progress:
            status = "failed" if extra.get("failure") else "done"
            print(f"  [{status}] exp{extra.get('index')} {extra.get('key')}")

    outcome = run_experiments(
        specs, bundles, class_names, seed=seed, artifact_dir=artifact_dir,
        start_index=100, on_result=_on_result,
    )
    new_results = [t.result for t in outcome["trained"]]
    extras = outcome["extras"]
    failures = outcome["failures"]

    # ---- unified comparison --------------------------------------------
    original_results = load_original_results(
        config.REPORTS_DIR / config.CLASSIFICATION_RESULTS_JSON
    )
    df = unified_comparison_dataframe(original_results, new_results)
    paths = save_experiment_outputs(
        out_dir, df, failures=failures, results=new_results,
        title="Phase 7 validation comparison (Word2Vec doc vectors + TF-IDF)",
    )

    # ---- baseline & decision -------------------------------------------
    baseline = max(original_results, key=lambda r: (r.macro_f1, r.accuracy))
    candidates = sorted(new_results, key=lambda r: (r.macro_f1, r.accuracy), reverse=True)
    best_candidate = candidates[0] if candidates else None

    decision: Dict[str, object] = {
        "baseline": {
            "index": baseline.index,
            "feature_type": baseline.feature_type,
            "classifier": baseline.classifier,
            "macro_f1": baseline.macro_f1,
            "accuracy": baseline.accuracy,
        },
        "threshold": REPLACEMENT_THRESHOLD,
    }
    if best_candidate is not None:
        decision["best_candidate"] = {
            "index": best_candidate.index,
            "feature_type": best_candidate.feature_type,
            "classifier": best_candidate.classifier,
            "macro_f1": best_candidate.macro_f1,
            "accuracy": best_candidate.accuracy,
            "improvement": best_candidate.macro_f1 - baseline.macro_f1,
        }
        decision["replace"] = select_replacement(
            baseline.macro_f1, best_candidate.macro_f1
        )
    else:
        decision["best_candidate"] = None
        decision["replace"] = False

    # ---- probability quality & optional calibration --------------------
    analysis = _baseline_probability_analysis(
        class_names, run_calibration=run_calibration, seed=seed
    )
    probability_path = out_dir / "validation_probability_quality.json"
    probability_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["probability"] = probability_path

    return {
        "specs": specs,
        "bundles": bundles,
        "new_results": new_results,
        "new_extras": extras,
        "failures": failures,
        "original_results": original_results,
        "comparison": df,
        "paths": paths,
        "decision": decision,
        "probability": analysis,
        "artifact_dir": artifact_dir,
    }


def _baseline_probability_analysis(
    class_names: Sequence[str],
    *,
    run_calibration: bool,
    seed: int,
) -> Dict[str, object]:
    """Characterise the selected model's probability output on validation."""
    try:
        artifact = load_classifier_artifact(
            config.CLASSIFIERS_DIR / "selected_model.joblib"
        )
    except FileNotFoundError:
        return {"available": False, "reason": "selected model artifact missing"}

    model = artifact["model"]
    model_name = artifact.get("model_name")
    if artifact.get("feature_family") != "docvec" or not model_name:
        return {"available": False, "reason": "selected model is not doc-vector based"}

    from src.text_features import load_docvec_split

    val = load_docvec_split(str(model_name), "validation", class_names=class_names)
    uncalibrated = probability_quality(model, val.X, val.y, class_names)

    result: Dict[str, object] = {
        "available": True,
        "model_name": str(model_name),
        "score_source": (
            "predict_proba" if hasattr(model, "predict_proba") else "unavailable"
        ),
        "uncalibrated": uncalibrated,
    }

    if run_calibration and hasattr(model, "predict_proba"):
        train = load_docvec_split(str(model_name), "train", class_names=class_names)
        result["calibrated_sigmoid_cv3"] = calibrated_variant(
            model, train.X, train.y, val.X, val.y, class_names,
            method="sigmoid", cv=3,
        )
    return result
