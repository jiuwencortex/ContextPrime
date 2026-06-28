# context_selectors/classifier_selector.py
# Query-time selection using a trained ComponentInclusionClassifier.
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..shared.classifier_model import ComponentInclusionClassifier

logger = logging.getLogger(__name__)


class ClassifierSelector:
    """Context selector driven by a trained ComponentInclusionClassifier.

    Loads classifier.pkl and predicts per-component inclusion probabilities
    from a query embedding.  No fallback logic — caller decides what to do
    when the classifier is not available or not confident enough.

    Usage::

        selector = ClassifierSelector.load(oracle_dir)
        result = selector.select(query_embedding)
        # result keys: skills, memory, tools, probabilities, confidence, source
    """

    def __init__(self, classifier: ComponentInclusionClassifier) -> None:
        self._classifier = classifier

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, oracle_dir: Path) -> "ClassifierSelector":
        """Load classifier.pkl from oracle_dir.

        Raises FileNotFoundError if classifier.pkl does not exist.
        """
        classifier_path = oracle_dir / "classifier.pkl"
        classifier = ComponentInclusionClassifier.load(classifier_path)
        logger.info(
            "Loaded classifier from %s (%d components)",
            classifier_path,
            classifier.n_components,
        )
        return cls(classifier)

    # ── inference ─────────────────────────────────────────────────────────────

    def select(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """Predict component inclusion from a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            Pre-computed embedding vector, shape (d_embed,).
        threshold : float
            Inclusion threshold for sigmoid output (default 0.5).

        Returns
        -------
        dict with keys:
            skills        list[str]        — skill names above threshold
            memory        list[str]        — memory section names above threshold
            tools         list[str]        — tool names above threshold
            probabilities dict[str, float] — per-component sigmoid scores
            confidence    float            — mean certainty (mean max(p, 1-p))
            source        str              — always "classifier"
        """
        result = self._classifier.predict(query_embedding, threshold=threshold)
        proba = self._classifier.predict_proba(query_embedding)
        confidence = _mean_confidence(proba)

        return {
            "skills": result["skills"],
            "memory": result["memory"],
            "tools": result["tools"],
            "probabilities": result["probabilities"],
            "confidence": round(confidence, 4),
            "source": "classifier",
        }

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def n_components(self) -> int:
        return self._classifier.n_components

    @property
    def component_names(self) -> list[str]:
        return self._classifier.component_names


def _mean_confidence(proba: np.ndarray) -> float:
    """Return mean max(p, 1-p) — average certainty per component decision."""
    if len(proba) == 0:
        return 0.0
    return float(np.mean(np.maximum(proba, 1 - proba)))
