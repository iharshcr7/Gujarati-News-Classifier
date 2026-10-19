"""Evaluation metrics, reports, and plots for Gujarati news classification.

Macro-F1 is the primary model-selection metric (the category distribution is
imbalanced); accuracy is secondary. All functions take the actual category
names so classification reports and confusion matrices are readable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config

# Use a non-interactive backend so plots can be saved on headless machines.
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Compute accuracy, macro/weighted metrics, and per-class scores.

    Returns a dict with ``accuracy``, ``macro_precision``, ``macro_recall``,
    ``macro_f1``, ``weighted_f1``, ``per_class`` (name -> precision/recall/f1/
    support), and ``support`` (number of evaluated rows).
    """
    y_true_arr = np.asarray(list(y_true), dtype=str)
    y_pred_arr = np.asarray(list(y_pred), dtype=str)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred length mismatch: "
            f"{y_true_arr.shape} vs {y_pred_arr.shape}"
        )
    classes = (
        [str(c) for c in class_names]
        if class_names is not None
        else sorted(set(y_true_arr.tolist()))
    )

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=classes, average="macro", zero_division=0
    )
    _, _, f_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=classes, average="weighted", zero_division=0
    )
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=classes, average=None, zero_division=0
    )

    per_class = {
        name: {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f_cls[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(classes)
    }

    return {
        "accuracy": accuracy,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_f1": float(f_weighted),
        "per_class": per_class,
        "support": int(len(y_true_arr)),
    }


def classification_report_text(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
    *,
    digits: int = 4,
) -> str:
    """Human-readable sklearn classification report using category names."""
    return classification_report(
        np.asarray(list(y_true), dtype=str),
        np.asarray(list(y_pred), dtype=str),
        labels=[str(c) for c in class_names],
        target_names=[str(c) for c in class_names],
        digits=digits,
        zero_division=0,
    )


def confusion_matrix_array(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
) -> np.ndarray:
    """Confusion matrix with rows/columns in ``class_names`` order."""
    return confusion_matrix(
        np.asarray(list(y_true), dtype=str),
        np.asarray(list(y_pred), dtype=str),
        labels=[str(c) for c in class_names],
    )


# ---------------------------------------------------------------------------
# Experiment result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """One classifier-on-feature-configuration validation experiment."""

    index: int
    feature_type: str
    feature_label: str
    classifier: str
    classifier_params: str
    train_rows: int
    val_rows: int
    num_features: int
    is_sparse: bool
    train_seconds: float
    eval_seconds: float
    metrics: Dict[str, object]
    reduced_dim: Optional[int] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    artifact_path: Optional[str] = None

    # Convenience accessors -------------------------------------------------
    @property
    def accuracy(self) -> float:
        return float(self.metrics["accuracy"])

    @property
    def macro_f1(self) -> float:
        return float(self.metrics["macro_f1"])

    @property
    def macro_precision(self) -> float:
        return float(self.metrics["macro_precision"])

    @property
    def macro_recall(self) -> float:
        return float(self.metrics["macro_recall"])

    @property
    def weighted_f1(self) -> float:
        return float(self.metrics["weighted_f1"])

    def to_row(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "feature_type": self.feature_type,
            "feature_label": self.feature_label,
            "classifier": self.classifier,
            "classifier_params": self.classifier_params,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "num_features": self.num_features,
            "is_sparse": self.is_sparse,
            "reduced_dim": self.reduced_dim if self.reduced_dim is not None else "",
            "train_seconds": round(self.train_seconds, 3),
            "eval_seconds": round(self.eval_seconds, 3),
            "accuracy": round(self.accuracy, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "macro_f1": round(self.macro_f1, 6),
            "weighted_f1": round(self.weighted_f1, 6),
            "warnings": " | ".join(self.warnings),
            "artifact_path": self.artifact_path or "",
        }


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def experiments_to_dataframe(results: Sequence[ExperimentResult]) -> pd.DataFrame:
    """Flatten experiment results into a DataFrame (one row per experiment)."""
    columns = [
        "index",
        "feature_type",
        "feature_label",
        "classifier",
        "classifier_params",
        "train_rows",
        "val_rows",
        "num_features",
        "is_sparse",
        "reduced_dim",
        "train_seconds",
        "eval_seconds",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "warnings",
        "artifact_path",
    ]
    rows = [r.to_row() for r in results]
    df = pd.DataFrame(rows, columns=columns)
    return df


def build_validation_comparison_table(results: Sequence[ExperimentResult]) -> str:
    """Markdown table of validation experiments sorted by macro-F1 (desc)."""
    ordered = sorted(
        results, key=lambda r: (r.macro_f1, r.accuracy), reverse=True
    )
    header = (
        "| Rank | Experiment | Feature | Classifier | Macro-F1 | Accuracy | "
        "Weighted-F1 | Macro-P | Macro-R |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for rank, r in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | {r.index} | {r.feature_label} | {r.classifier} | "
            f"{r.macro_f1:.4f} | {r.accuracy:.4f} | {r.weighted_f1:.4f} | "
            f"{r.macro_precision:.4f} | {r.macro_recall:.4f} |"
        )
    return "\n".join(lines)


def result_to_json(result: ExperimentResult) -> Dict[str, object]:
    """JSON-serializable representation of a result (keeps per-class metrics)."""
    payload = asdict(result)
    payload["warnings"] = list(result.warnings)
    return payload


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    title: str,
) -> Path:
    """Save a labelled confusion-matrix heatmap and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_metric_comparison_plot(df: pd.DataFrame, path: Path) -> Path:
    """Save a bar chart of validation macro-F1 per experiment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["experiment"] = (
        plot_df["feature_type"].astype(str)
        + " / "
        + plot_df["classifier"].astype(str)
    )
    plot_df = plot_df.sort_values("macro_f1", ascending=False)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(plot_df))))
    sns.barplot(
        data=plot_df, y="experiment", x="macro_f1", hue="feature_type",
        dodge=False, ax=ax,
    )
    ax.set_xlabel("Validation macro-F1")
    ax.set_ylabel("Experiment")
    ax.set_title("Validation macro-F1 by experiment")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
