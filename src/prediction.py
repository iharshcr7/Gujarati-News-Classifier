"""Reusable Gujarati news headline inference (Phase 5).

Loads a classifier artifact saved by :mod:`src.classification` and predicts the
category of a Gujarati headline using exactly the preprocessing and feature
transformation the model was trained with. Nothing is retrained here, and the
Word2Vec model (for document-vector models) is loaded read-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import config
from src.classification import load_classifier_artifact
from src.preprocessing import preprocess


DEFAULT_MODEL_PATH = config.CLASSIFIERS_DIR / "selected_model.joblib"


class NewsClassifier:
    """Predicts the category of a Gujarati news headline.

    The artifact stores the estimator plus, for TF-IDF models, the exact fitted
    vectorizer and (for tree ensembles) the SVD reducer, so inference uses the
    same transformation as training.
    """

    def __init__(self, artifact: Dict[str, object]):
        self.model = artifact["model"]
        self.vectorizer = artifact.get("vectorizer")
        self.svd = artifact.get("svd")
        self.feature_type = str(artifact.get("feature_type", ""))
        self.feature_family = str(artifact.get("feature_family", ""))
        self.model_name = artifact.get("model_name")
        self.class_names = [str(c) for c in artifact.get("class_names", [])]
        self.metadata = dict(artifact.get("metadata", {}) or {})
        self._w2v = None

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NewsClassifier":
        """Load a classifier artifact (defaults to the selected model)."""
        return cls(load_classifier_artifact(Path(path or DEFAULT_MODEL_PATH)))

    # -- feature transformation --------------------------------------------
    def _ensure_w2v(self):
        if self._w2v is None:
            from src.document_vectors import load_embedding_model

            if self.model_name == "cbow":
                model_path = config.W2V_MODEL_CBOW_PATH
            elif self.model_name == "skipgram":
                model_path = config.W2V_MODEL_SKIPGRAM_PATH
            else:
                raise ValueError(
                    f"Unknown document-vector model_name {self.model_name!r}"
                )
            self._w2v = load_embedding_model(model_path)
        return self._w2v

    def _vectorize(self, headline: str):
        if not isinstance(headline, str) or not headline.strip():
            raise ValueError(
                "Input headline must be a non-empty Gujarati string."
            )
        result = preprocess(headline)
        if not result.tokens:
            raise ValueError(
                "Input headline has no usable tokens after preprocessing."
            )

        if self.feature_family == "docvec":
            from src.document_vectors import mean_pool_tokens

            w2v = self._ensure_w2v()
            vector, _ = mean_pool_tokens(
                result.tokens, w2v.wv.key_to_index, w2v.wv.vectors
            )
            X = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        elif self.feature_family == "tfidf":
            if self.vectorizer is None:
                raise ValueError("TF-IDF classifier artifact has no vectorizer")
            X = self.vectorizer.transform([result.cleaned])
        else:
            raise ValueError(
                f"Unknown feature family {self.feature_family!r} in artifact"
            )

        if self.svd is not None:
            X = self.svd.transform(X)
        return X

    # -- prediction ---------------------------------------------------------
    def predict(self, headline: str) -> str:
        """Return the predicted category name for ``headline``."""
        X = self._vectorize(headline)
        return str(self.model.predict(X)[0])

    def predict_with_scores(
        self, headline: str
    ) -> Tuple[str, Dict[str, float]]:
        """Return ``(label, scores)`` where scores cover every class.

        Uses ``predict_proba`` when available, otherwise ``decision_function``.
        """
        X = self._vectorize(headline)
        classes = [str(c) for c in getattr(self.model, "classes_", self.class_names)]
        label = str(self.model.predict(X)[0])

        if hasattr(self.model, "predict_proba"):
            values = np.asarray(self.model.predict_proba(X))[0]
        elif hasattr(self.model, "decision_function"):
            raw = np.asarray(self.model.decision_function(X)).reshape(-1)
            if len(classes) == 2 and raw.size == 1:
                # Binary ``LinearSVC`` returns one signed score per sample.
                values = np.array([-float(raw[0]), float(raw[0])])
            else:
                values = raw
        else:  # pragma: no cover - all supported estimators expose one
            values = np.zeros(len(classes), dtype=float)

        return label, {name: float(v) for name, v in zip(classes, values)}


def predict_headline(
    headline: str, *, model_path: Optional[Path] = None
) -> str:
    """Convenience wrapper: load the selected model and predict one headline."""
    return NewsClassifier.load(model_path).predict(headline)
