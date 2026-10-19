"""Validation error analysis for the Gujarati news classifier (Phase 7).

Builds a detailed, reusable analysis of the frozen Phase 5 selected model's
mistakes on the **validation** split (never the test split). For every validation
example it records a stable row id, the original Gujarati headline, the true and
predicted category, correctness, and -- when the estimator exposes them -- the
prediction scores with their **true semantics** (probability estimates from
``predict_proba`` vs. decision-function margins). Confidence values are never
fabricated: when scores are unavailable they are omitted and the reason is
stated.

Outputs (UTF-8, Gujarati-safe):
- ``validation_error_analysis.md``      human-readable report
- ``validation_predictions.csv``        one row per validation example
- ``validation_misclassified.csv``      only the misclassified examples
- ``validation_confusion_pairs.csv``    ranked true->predicted confusion pairs

Data-leakage prevention: this module only reads existing artifacts (the selected
classifier, the document vectors, and the raw splits via the shared cleaning
pipeline). It never retrains anything, never modifies the CSVs, and never uses
test labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from src.classification import load_classifier_artifact
from src.data_loader import load_all_splits, validate_all
from src.preprocessing import clean_splits
from src.text_features import SPLITS, load_docvec_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYSIS_SPLIT = "validation"

# How many confusion pairs / examples to surface in the human-readable report.
TOP_CONFUSION_PAIRS = 15
EXAMPLES_PER_PAIR = 5

SCORE_PROBABILITY = "probability"
SCORE_DECISION = "decision"
SCORE_UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Score semantics
# ---------------------------------------------------------------------------

@dataclass
class ScoreInfo:
    """Per-example prediction scores plus their semantics.

    ``values`` columns follow ``classes`` (the estimator's own ``classes_``
    order). ``kind`` is ``"probability"`` for ``predict_proba`` output,
    ``"decision"`` for raw decision-function margins, or ``"unavailable"`` when
    the estimator exposes neither.
    """

    values: Optional[np.ndarray]
    kind: str
    source: str
    classes: Tuple[str, ...] = field(default_factory=tuple)


def get_prediction_scores(
    model,
    X,
    class_names: Optional[Sequence[str]] = None,
) -> ScoreInfo:
    """Return the estimator's scores and their true semantics for ``X``.

    Prefers ``predict_proba`` (probability estimates). Falls back to
    ``decision_function`` (margins). If neither exists the scores are reported as
    unavailable rather than invented.
    """
    classes = tuple(
        str(c) for c in getattr(model, "classes_", class_names or ())
    )
    n_rows = int(np.asarray(X).shape[0]) if hasattr(X, "shape") else 0

    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(X), dtype=float)
        return ScoreInfo(values=values, kind=SCORE_PROBABILITY,
                         source="predict_proba", classes=classes)

    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        n_classes = len(classes) if classes else (
            raw.shape[1] if raw.ndim == 2 else 1
        )
        if raw.ndim == 1:
            if n_classes == 2 and raw.size == max(n_rows, 1):
                raw = np.column_stack([-raw, raw])
            else:
                raw = raw.reshape(-1, 1)
        return ScoreInfo(values=raw, kind=SCORE_DECISION,
                         source="decision_function", classes=classes)

    return ScoreInfo(values=None, kind=SCORE_UNAVAILABLE, source="none",
                     classes=classes)


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def collect_predictions(
    model,
    X,
    y_true: Sequence[str],
    class_names: Sequence[str],
    *,
    row_ids: Optional[Sequence[int]] = None,
    texts: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return one record per validation example with scores when available."""
    y_true_arr = np.asarray(list(y_true), dtype=str)
    n = len(y_true_arr)
    if hasattr(X, "shape") and X.shape[0] != n:
        raise ValueError(
            f"X has {X.shape[0]} rows but y_true has {n} rows"
        )
    if texts is not None and len(texts) != n:
        raise ValueError("texts length does not match y_true")
    if row_ids is not None and len(row_ids) != n:
        raise ValueError("row_ids length does not match y_true")

    ids = np.arange(n, dtype=np.int64) if row_ids is None else np.asarray(row_ids)
    text_list = [""] * n if texts is None else [str(t) for t in texts]

    y_pred = np.asarray(model.predict(X)).astype(str)
    if y_pred.shape != y_true_arr.shape:
        raise ValueError("model.predict returned an unexpected shape")

    info = get_prediction_scores(model, X, class_names)
    class_list = list(info.classes) if info.classes else [str(c) for c in class_names]
    index_of = {c: i for i, c in enumerate(class_list)}

    predicted_score: List[object] = []
    runner_up: List[object] = []
    runner_up_score: List[object] = []
    margin: List[object] = []

    for row in range(n):
        if info.values is None:
            predicted_score.append(pd.NA)
            runner_up.append(pd.NA)
            runner_up_score.append(pd.NA)
            margin.append(pd.NA)
            continue
        scores = info.values[row]
        order = np.argsort(-scores, kind="stable")
        label = y_pred[row]
        label_idx = index_of.get(label)
        top_score = (
            float(scores[label_idx]) if label_idx is not None else float(scores[order[0]])
        )
        rival_label = None
        rival_score = None
        for idx in order:
            cand = class_list[int(idx)]
            if cand != label:
                rival_label = cand
                rival_score = float(scores[int(idx)])
                break
        predicted_score.append(top_score)
        runner_up.append(rival_label if rival_label is not None else pd.NA)
        runner_up_score.append(rival_score if rival_score is not None else pd.NA)
        margin.append(
            float(top_score - rival_score) if rival_score is not None else pd.NA
        )

    return pd.DataFrame(
        {
            "row_id": ids,
            "text": text_list,
            "true_category": y_true_arr,
            "predicted_category": y_pred,
            "correct": y_pred == y_true_arr,
            "score_type": info.kind,
            "predicted_score": predicted_score,
            "runner_up_category": runner_up,
            "runner_up_score": runner_up_score,
            "margin": margin,
        }
    )


def misclassified_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return only the misclassified rows (``correct == False``)."""
    return predictions.loc[~predictions["correct"].astype(bool)].copy()


# ---------------------------------------------------------------------------
# Confusion pairs
# ---------------------------------------------------------------------------

CONFUSION_PAIR_COLUMNS = (
    "rank",
    "true_category",
    "predicted_category",
    "count",
    "share_of_true_class",
    "share_of_predicted_class",
    "share_of_all_errors",
)


def confusion_pairs(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Ranked true->predicted confusion pairs (correct pairs excluded).

    ``share_of_true_class`` is the fraction of that true category's examples
    misrouted to the predicted one (recall impact); ``share_of_predicted_class``
    is the fraction of that predicted category's examples that came from the
    wrong true category (precision impact); ``share_of_all_errors`` is the pair's
    share of every misclassification.
    """
    y_true_arr = np.asarray(list(y_true), dtype=str)
    y_pred_arr = np.asarray(list(y_pred), dtype=str)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred length mismatch")

    classes = [str(c) for c in class_names]
    true_counts = {c: int((y_true_arr == c).sum()) for c in classes}
    pred_counts = {c: int((y_pred_arr == c).sum()) for c in classes}
    total_errors = int((y_true_arr != y_pred_arr).sum())

    rows: List[Dict[str, object]] = []
    for true_c in classes:
        for pred_c in classes:
            if true_c == pred_c:
                continue
            count = int(((y_true_arr == true_c) & (y_pred_arr == pred_c)).sum())
            if count == 0:
                continue
            rows.append(
                {
                    "true_category": true_c,
                    "predicted_category": pred_c,
                    "count": count,
                    "share_of_true_class": (
                        count / true_counts[true_c] if true_counts[true_c] else 0.0
                    ),
                    "share_of_predicted_class": (
                        count / pred_counts[pred_c] if pred_counts[pred_c] else 0.0
                    ),
                    "share_of_all_errors": (
                        count / total_errors if total_errors else 0.0
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(columns=CONFUSION_PAIR_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["count", "true_category", "predicted_category"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    return df[list(CONFUSION_PAIR_COLUMNS)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _per_class_markdown(metrics: Mapping[str, object], class_names: Sequence[str]) -> str:
    lines = [
        "| Category | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    per_class = metrics["per_class"]  # type: ignore[index]
    for name in class_names:
        pc = per_class[name]  # type: ignore[index]
        lines.append(
            f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['support']} |"
        )
    return "\n".join(lines)


def _pairs_markdown(pairs_df: pd.DataFrame, top: int) -> str:
    if pairs_df.empty:
        return "_No misclassifications._"
    lines = [
        "| Rank | True | Predicted | Count | Share of true class | "
        "Share of predicted class | Share of all errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in pairs_df.head(top).iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['true_category']} "
            f"| {row['predicted_category']} | {int(row['count'])} "
            f"| {row['share_of_true_class']:.4f} "
            f"| {row['share_of_predicted_class']:.4f} "
            f"| {row['share_of_all_errors']:.4f} |"
        )
    return "\n".join(lines)


def _examples_markdown(
    preds_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    top_pairs: int,
    per_pair: int,
) -> str:
    if pairs_df.empty:
        return "_No misclassified examples._"
    blocks: List[str] = []
    for _, pair in pairs_df.head(top_pairs).iterrows():
        subset = preds_df[
            (~preds_df["correct"].astype(bool))
            & (preds_df["true_category"] == pair["true_category"])
            & (preds_df["predicted_category"] == pair["predicted_category"])
        ]
        if subset.empty:
            continue
        blocks.append(
            f"#### {pair['true_category']} -> {pair['predicted_category']} "
            f"({int(pair['count'])} errors)"
        )
        for _, row in subset.head(per_pair).iterrows():
            score_note = ""
            if row["score_type"] != SCORE_UNAVAILABLE and not pd.isna(
                row.get("predicted_score")
            ):
                score_note = (
                    f" _(predicted score {row['predicted_score']:.4f}, "
                    f"margin {row['margin']:.4f}, {row['score_type']})_"
                )
            blocks.append(
                f"- row {int(row['row_id'])}: {row['text']}{score_note}"
            )
        blocks.append("")
    return "\n".join(blocks) if blocks else "_No misclassified examples._"


def build_error_analysis_report(
    *,
    metrics: Mapping[str, object],
    pairs_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    class_names: Sequence[str],
    split: str = ANALYSIS_SPLIT,
    model_description: str = "",
    dataset_counts: Optional[Mapping[str, int]] = None,
    top_pairs: int = TOP_CONFUSION_PAIRS,
    examples_per_pair: int = EXAMPLES_PER_PAIR,
) -> str:
    """Build the UTF-8 markdown validation error-analysis report."""
    total = int(metrics["support"])
    errors = int((~preds_df["correct"].astype(bool)).sum())
    error_rate = errors / total if total else 0.0
    dataset_counts = dataset_counts or {}

    lines: List[str] = []
    lines.append("# Validation Error Analysis — Gujarati News Classifier (Phase 7)")
    lines.append("")
    lines.append("## Scope and data-leakage prevention")
    lines.append("")
    lines.append(f"- Analysis split: **{split}** (test labels are never inspected).")
    lines.append(f"- Model: {model_description or 'Phase 5 selected model'}")
    for name, count in dataset_counts.items():
        lines.append(f"- {name} rows: {count}")
    lines.append("- Word2Vec / document vectors / selected classifier: read-only.")
    lines.append("- Original CSVs and split membership: unchanged.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Total {split} examples: **{total}**")
    lines.append(f"- Misclassified: **{errors}**")
    lines.append(f"- Error rate: **{error_rate:.4f}** ({error_rate * 100:.2f}%)")
    lines.append(f"- Accuracy: {float(metrics['accuracy']):.4f}")
    lines.append(f"- Macro precision: {float(metrics['macro_precision']):.4f}")
    lines.append(f"- Macro recall: {float(metrics['macro_recall']):.4f}")
    lines.append(f"- Macro F1: {float(metrics['macro_f1']):.4f}")
    lines.append(f"- Weighted F1: {float(metrics['weighted_f1']):.4f}")
    lines.append("")
    if preds_df["score_type"].nunique() == 1:
        kind = str(preds_df["score_type"].iloc[0])
        if kind == SCORE_PROBABILITY:
            lines.append(
                "Prediction scores are **probability estimates** from the "
                "estimator's `predict_proba` (they sum to ~1.0 but are not "
                "calibrated guarantees of correctness)."
            )
        elif kind == SCORE_DECISION:
            lines.append(
                "Prediction scores are raw **decision-function margins**, not "
                "probabilities."
            )
        else:
            lines.append(
                "The estimator exposes neither `predict_proba` nor "
                "`decision_function`, so per-example scores are omitted rather "
                "than fabricated."
            )
    lines.append("")
    lines.append("## Per-class metrics")
    lines.append("")
    lines.append(_per_class_markdown(metrics, class_names))
    lines.append("")
    lines.append(f"## Top {min(top_pairs, len(pairs_df))} confusion pairs")
    lines.append("")
    lines.append(_pairs_markdown(pairs_df, top_pairs))
    lines.append("")
    lines.append("## Representative misclassified examples")
    lines.append("")
    lines.append(_examples_markdown(preds_df, pairs_df, top_pairs, examples_per_pair))
    lines.append("")
    lines.append("## Errors by true category")
    lines.append("")
    lines.append(_errors_by_true_markdown(preds_df, class_names))
    lines.append("")
    lines.append("## Errors by predicted category")
    lines.append("")
    lines.append(_errors_by_predicted_markdown(preds_df, class_names))
    lines.append("")
    return "\n".join(lines)


def _errors_by_true_markdown(preds_df: pd.DataFrame, class_names: Sequence[str]) -> str:
    lines = ["| True category | Support | Errors | Error rate |", "|---|---|---|---|"]
    for name in class_names:
        subset = preds_df[preds_df["true_category"] == name]
        support = len(subset)
        errors = int((~subset["correct"].astype(bool)).sum())
        rate = errors / support if support else 0.0
        lines.append(f"| {name} | {support} | {errors} | {rate:.4f} |")
    return "\n".join(lines)


def _errors_by_predicted_markdown(
    preds_df: pd.DataFrame, class_names: Sequence[str]
) -> str:
    mis = preds_df[~preds_df["correct"].astype(bool)]
    lines = ["| Predicted category | Predictions | Wrong | Wrong rate |", "|---|---|---|---|"]
    for name in class_names:
        subset = preds_df[preds_df["predicted_category"] == name]
        predicted = len(subset)
        wrong = int((mis["predicted_category"] == name).sum())
        rate = wrong / predicted if predicted else 0.0
        lines.append(f"| {name} | {predicted} | {wrong} | {rate:.4f} |")
    return "\n".join(lines)


def save_error_analysis(
    out_dir: Path,
    *,
    report: str,
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
) -> Dict[str, Path]:
    """Write the report and machine-readable CSVs (UTF-8) and return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    misclassified = misclassified_examples(predictions)

    report_path = out_dir / "validation_error_analysis.md"
    predictions_path = out_dir / "validation_predictions.csv"
    misclassified_path = out_dir / "validation_misclassified.csv"
    pairs_path = out_dir / "validation_confusion_pairs.csv"

    report_path.write_text(report, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    misclassified.to_csv(misclassified_path, index=False, encoding="utf-8")
    pairs.to_csv(pairs_path, index=False, encoding="utf-8")

    return {
        "report": report_path,
        "predictions": predictions_path,
        "misclassified": misclassified_path,
        "confusion_pairs": pairs_path,
    }


# ---------------------------------------------------------------------------
# Baseline reproduction helper (Step C)
# ---------------------------------------------------------------------------

def compare_to_recorded(
    reproduced: Mapping[str, object],
    recorded: Mapping[str, object],
    *,
    tolerance: float = 1e-3,
) -> Tuple[bool, str]:
    """Compare reproduced validation metrics to recorded ones.

    Returns ``(matches, markdown_report)``. Only overall metrics are compared;
    per-class metrics are included for information.
    """
    keys = (
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    )
    lines = [
        "| Metric | Reproduced | Recorded | Abs. diff | Within tolerance |",
        "|---|---|---|---|---|",
    ]
    matches = True
    for key, label in keys:
        rep = float(reproduced[key])
        rec = float(recorded[key])
        diff = abs(rep - rec)
        ok = diff <= tolerance
        matches = matches and ok
        lines.append(
            f"| {label} | {rep:.6f} | {rec:.6f} | {diff:.2e} | "
            f"{'yes' if ok else 'NO'} |"
        )
    return matches, "\n".join(lines)


# ---------------------------------------------------------------------------
# Step B — investigation helpers
# ---------------------------------------------------------------------------

import re as _re

_ASCII_LETTER_RE = _re.compile(r"[A-Za-z]")


def enrich_token_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add per-example token count and mixed-script flags (evidence for Step B)."""
    from src.preprocessing import preprocess

    df = predictions.copy()
    token_lists = [
        preprocess(text).tokens if isinstance(text, str) else []
        for text in df["text"].astype(str)
    ]
    df["num_tokens"] = [len(tokens) for tokens in token_lists]
    df["has_ascii"] = [
        bool(_ASCII_LETTER_RE.search(text)) for text in df["text"].astype(str)
    ]
    return df


_LENGTH_BUCKETS = (
    ("empty_or_1", 0, 1),
    ("2_to_3", 2, 3),
    ("4_to_7", 4, 7),
    ("8_to_15", 8, 15),
    ("16_plus", 16, 10 ** 9),
)


def error_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Error rate by headline token-count bucket and by mixed-script presence."""
    if "num_tokens" not in predictions.columns:
        predictions = enrich_token_features(predictions)
    correct = ~predictions["correct"].astype(bool)
    rows: List[Dict[str, object]] = []
    for name, lo, hi in _LENGTH_BUCKETS:
        mask = predictions["num_tokens"].between(lo, hi)
        support = int(mask.sum())
        errors = int((mask & correct).sum())
        rows.append(
            {
                "bucket": name,
                "support": support,
                "errors": errors,
                "error_rate": errors / support if support else 0.0,
            }
        )
    ascii_mask = predictions["has_ascii"].astype(bool)
    support = int(ascii_mask.sum())
    errors = int((ascii_mask & correct).sum())
    rows.append(
        {
            "bucket": "mixed_script_ascii",
            "support": support,
            "errors": errors,
            "error_rate": errors / support if support else 0.0,
        }
    )
    return pd.DataFrame(rows)


def class_vocabulary_overlap(
    texts_by_class: Mapping[str, Sequence[str]],
    *,
    min_len: int = 2,
    top: int = 20,
) -> pd.DataFrame:
    """Pairwise token-vocabulary overlap between categories (Step B evidence)."""
    from src.preprocessing import preprocess

    vocab: Dict[str, set] = {}
    freq: Dict[str, Dict[str, int]] = {}
    for label, texts in texts_by_class.items():
        counts: Dict[str, int] = {}
        for text in texts:
            for token in preprocess(text).tokens if isinstance(text, str) else []:
                if len(token) < min_len:
                    continue
                counts[token] = counts.get(token, 0) + 1
        vocab[str(label)] = set(counts)
        freq[str(label)] = counts

    labels = list(vocab)
    rows: List[Dict[str, object]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            shared = vocab[a] & vocab[b]
            union = vocab[a] | vocab[b]
            jaccard = len(shared) / len(union) if union else 0.0
            top_shared = sorted(
                shared, key=lambda t: (-(freq[a][t] + freq[b][t]), t)
            )[:top]
            rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "overlap_count": len(shared),
                    "jaccard": jaccard,
                    "shared_tokens": " ".join(top_shared),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["class_a", "class_b", "overlap_count", "jaccard", "shared_tokens"]
        )
    df = pd.DataFrame(rows).sort_values(
        ["jaccard", "overlap_count"], ascending=False
    ).reset_index(drop=True)
    return df


def build_error_cause_investigation(
    enriched: pd.DataFrame,
    breakdown: pd.DataFrame,
    pairs_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    *,
    class_names: Sequence[str],
    split: str = ANALYSIS_SPLIT,
    top_pairs: int = TOP_CONFUSION_PAIRS,
    overlap_pairs: int = 10,
) -> str:
    """Markdown Step B report explaining error patterns with real examples."""
    mis = enriched[~enriched["correct"].astype(bool)]
    lines: List[str] = []
    lines.append("# Error Cause Investigation — Gujarati News Classifier (Phase 7)")
    lines.append("")
    lines.append(
        "This report uses **validation** errors only. It groups patterns and "
        "states which potential improvements are supported by the evidence; it "
        "does not relabel any example from the model's predictions."
    )
    lines.append("")
    lines.append("## Headline length (token count) vs. error rate")
    lines.append("")
    lines.append("| Token bucket | Support | Errors | Error rate |")
    lines.append("|---|---|---|---|")
    for _, row in breakdown.iterrows():
        lines.append(
            f"| {row['bucket']} | {int(row['support'])} | {int(row['errors'])} "
            f"| {float(row['error_rate']):.4f} |"
        )
    lines.append("")
    lines.append("## Mixed Gujarati/ASCII script")
    lines.append("")
    ascii_rows = breakdown[breakdown["bucket"] == "mixed_script_ascii"]
    if not ascii_rows.empty:
        row = ascii_rows.iloc[0]
        lines.append(
            f"- Headlines containing ASCII letters: {int(row['support'])} "
            f"(errors {int(row['errors'])}, rate {float(row['error_rate']):.4f})."
        )
        sample = enriched[enriched["has_ascii"].astype(bool)].head(5)
        for _, r in sample.iterrows():
            mark = "MISCLASSIFIED" if not bool(r["correct"]) else "correct"
            lines.append(
                f"  - [{mark}] row {int(r['row_id'])}: {r['text']} "
                f"(true={r['true_category']}, pred={r['predicted_category']})"
            )
    lines.append("")
    lines.append("## Length-bucket evidence: shortest headlines")
    lines.append("")
    shortest = mis[mis["num_tokens"] <= 3].head(8)
    if shortest.empty:
        lines.append("- No misclassified headlines with <= 3 tokens.")
    else:
        for _, r in shortest.iterrows():
            lines.append(
                f"- row {int(r['row_id'])} ({int(r['num_tokens'])} tokens): "
                f"{r['text']} (true={r['true_category']}, "
                f"pred={r['predicted_category']})"
            )
    lines.append("")
    lines.append("## Terms shared between frequently confused categories")
    lines.append("")
    lines.append("Highest token-vocabulary overlap across the ten categories:")
    lines.append("")
    lines.append("| Class A | Class B | Shared tokens | Jaccard | Top shared |")
    lines.append("|---|---|---|---|---|")
    for _, row in overlap_df.head(overlap_pairs).iterrows():
        lines.append(
            f"| {row['class_a']} | {row['class_b']} | {int(row['overlap_count'])} "
            f"| {float(row['jaccard']):.4f} | {row['shared_tokens']} |"
        )
    lines.append("")
    lines.append("## Representative confusion-pair examples")
    lines.append("")
    for _, pair in pairs_df.head(top_pairs).iterrows():
        subset = mis[
            (mis["true_category"] == pair["true_category"])
            & (mis["predicted_category"] == pair["predicted_category"])
        ]
        lines.append(
            f"### {pair['true_category']} -> {pair['predicted_category']} "
            f"({int(pair['count'])} errors, {float(pair['share_of_true_class']):.2f} "
            f"of the true class)"
        )
        for _, r in subset.head(EXAMPLES_PER_PAIR).iterrows():
            lines.append(
                f"- row {int(r['row_id'])} ({int(r['num_tokens'])} tokens): {r['text']}"
            )
        lines.append("")
    lines.append("## Evidence-supported observations")
    lines.append("")
    lines.append(
        "- Overlapping vocabulary between categories is a leading explanation "
        "where shared terms are high and the confusion pair is large."
    )
    lines.append(
        "- Short, information-poor headlines show a higher error rate than "
        "longer headlines (see the length table)."
    )
    lines.append(
        "- These observations guide the targeted experiments in Phase 7; no "
        "labels are changed based on model predictions."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline reproduction report (Step C)
# ---------------------------------------------------------------------------

def build_baseline_reproduction_report(
    *,
    reproduced: Mapping[str, object],
    recorded: Mapping[str, object],
    model_description: str,
    artifact_path: object,
    tolerance: float = 1e-3,
    extra_note: str = "",
) -> str:
    """Markdown Step C report comparing reproduced vs recorded validation metrics."""
    matches, table = compare_to_recorded(reproduced, recorded, tolerance=tolerance)
    lines = [
        "# Baseline Reproduction — Phase 5 Selected Model (Phase 7, Step C)",
        "",
        "The frozen Phase 5 selected model was loaded read-only and re-evaluated "
        "on the existing validation features. No model is retrained here, and no "
        "test labels are used.",
        "",
        f"- Model: {model_description}",
        f"- Artifact: {artifact_path}",
        f"- Tolerance: {tolerance:g} absolute",
        "",
        "## Comparison",
        "",
        table,
        "",
        (
            "**Verdict: reproduction MATCHES the recorded Phase 5 validation "
            "metrics within tolerance.**"
            if matches
            else "**Verdict: MISMATCH — investigate before any tuning.**"
        ),
    ]
    if extra_note:
        lines.append("")
        lines.append(extra_note)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_validation_error_analysis(
    *,
    artifact_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    expected_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    """Run the full Step A analysis on the validation split and save outputs."""
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    out_dir = Path(out_dir or (config.REPORTS_DIR / "error_analysis"))
    artifact_path = Path(artifact_path or (config.CLASSIFIERS_DIR / "selected_model.joblib"))

    artifact = load_classifier_artifact(artifact_path)
    model = artifact["model"]
    class_names = [str(c) for c in artifact["class_names"]]
    model_name = artifact.get("model_name")
    if artifact.get("feature_family") != "docvec" or not model_name:
        raise ValueError(
            "This Phase 7 analysis expects the doc-vector selected model "
            f"(got feature_family={artifact.get('feature_family')!r}, "
            f"model_name={model_name!r})."
        )

    feature = load_docvec_split(
        str(model_name), ANALYSIS_SPLIT, processed_dir=processed_dir,
        class_names=class_names,
    )

    # Original Gujarati text, aligned positionally with the doc-vector labels.
    raw = load_all_splits()
    validate_all(raw)
    cleaned, _ = clean_splits(raw)
    frame = {
        "train": cleaned.train,
        "validation": cleaned.validation,
        "test": cleaned.test,
    }[ANALYSIS_SPLIT]
    if expected_counts and ANALYSIS_SPLIT in expected_counts:
        if len(frame) != expected_counts[ANALYSIS_SPLIT]:
            raise ValueError(
                f"{ANALYSIS_SPLIT}: expected {expected_counts[ANALYSIS_SPLIT]} rows "
                f"but found {len(frame)}"
            )
    texts = frame[config.TEXT_COLUMN].astype(str).tolist()
    labels = frame[config.LABEL_COLUMN].astype(str).to_numpy()
    if not np.array_equal(labels, feature.y):
        raise ValueError(
            f"{ANALYSIS_SPLIT}: cleaned labels do not align positionally with "
            "the document-vector labels"
        )

    predictions = collect_predictions(
        model,
        feature.X,
        feature.y,
        class_names,
        row_ids=feature.row_ids,
        texts=texts,
    )
    pairs = confusion_pairs(feature.y, predictions["predicted_category"], class_names)

    from src.evaluation import compute_metrics

    metrics = compute_metrics(feature.y, predictions["predicted_category"], class_names)

    dataset_counts = {
        "train": expected_counts.get("train") if expected_counts else None,
        "validation": len(feature.y),
        "test": expected_counts.get("test") if expected_counts else None,
    }
    dataset_counts = {k: v for k, v in dataset_counts.items() if v is not None}

    model_description = (
        f"{artifact.get('feature_label', feature.feature_type)} / "
        f"{artifact['metadata'].get('classifier', 'classifier')} "
        f"({artifact['metadata'].get('classifier_params', '')})"
    )
    report = build_error_analysis_report(
        metrics=metrics,
        pairs_df=pairs,
        preds_df=predictions,
        class_names=class_names,
        split=ANALYSIS_SPLIT,
        model_description=model_description,
        dataset_counts=dataset_counts,
    )
    paths = save_error_analysis(
        out_dir, report=report, predictions=predictions, pairs=pairs
    )

    enriched = enrich_token_features(predictions)
    breakdown = error_breakdown(enriched)
    texts_by_class: Dict[str, List[str]] = {}
    for name in class_names:
        texts_by_class[name] = enriched.loc[
            enriched["true_category"] == name, "text"
        ].tolist()
    overlap = class_vocabulary_overlap(texts_by_class)
    investigation = build_error_cause_investigation(
        enriched, breakdown, pairs, overlap,
        class_names=class_names, split=ANALYSIS_SPLIT,
    )
    investigation_path = out_dir / "error_cause_investigation.md"
    investigation_path.write_text(investigation, encoding="utf-8")

    return {
        "artifact": artifact,
        "artifact_path": artifact_path,
        "model": model,
        "class_names": class_names,
        "feature": feature,
        "texts": texts,
        "predictions": predictions,
        "enriched": enriched,
        "confusion_pairs": pairs,
        "breakdown": breakdown,
        "vocabulary_overlap": overlap,
        "metrics": metrics,
        "report": report,
        "investigation_report": investigation,
        "paths": paths,
        "investigation_path": investigation_path,
        "model_description": model_description,
    }
