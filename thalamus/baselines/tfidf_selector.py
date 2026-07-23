"""Baseline: TF-IDF cosine similarity retrieval — top-k components per query.

Each component is represented by the concatenation of its example_input texts from
the scoring matrix.  At query time the query is vectorized with the same TF-IDF
model and cosine similarity is computed against every component document.  The top-k
most similar components are returned, where k is derived from the oracle budget counts.

This is the "current practice" baseline — equivalent to ``SKILL_MODE_RECOMMENDATION``
in agent-core's ``RecommendSkillTool``.  A structured selector should beat it on
tasks where component interactions matter (multi-file features, architecture tasks).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..shared.context_orderer import bookend_order
from .component_catalog import ComponentCatalog, ComponentEntry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FEATURES = 2000


class TFIDFSelector:
    """TF-IDF cosine similarity retrieval baseline.

    Usage::

        selector = TFIDFSelector.load(oracle_dir)
        result = selector.select("Set up CI/CD pipeline", budget="medium")
        # result = {"skills": [...top-k by TF-IDF similarity...], ...}

    The vectorizer is fit at load time on all component texts.  Subsequent
    ``select()`` calls are O(n_components) cosine similarity lookups — fast enough
    for interactive use but not optimized for high-throughput serving.
    """

    def __init__(
        self,
        catalog: ComponentCatalog,
        vectorizer: TfidfVectorizer,
        doc_matrix: np.ndarray,  # shape (n_components, n_features)
        entries_order: list[ComponentEntry],
    ) -> None:
        self._catalog = catalog
        self._vectorizer = vectorizer
        self._doc_matrix = doc_matrix
        self._entries_order = entries_order

    @classmethod
    def load(
        cls,
        oracle_dir: str | Path,
        max_features: int = _DEFAULT_MAX_FEATURES,
    ) -> "TFIDFSelector":
        """Load catalog and fit TF-IDF vectorizer on component texts.

        Parameters
        ----------
        oracle_dir:
            Directory containing scoring matrices and optionally context_configs.json.
        max_features:
            TF-IDF vocabulary size (default 2000, matching QueryClusterer).
        """
        oracle_dir = Path(oracle_dir)
        catalog = ComponentCatalog.load(oracle_dir)
        entries = catalog.entries()
        if not entries:
            raise FileNotFoundError(
                f"No scoring matrices found in {oracle_dir}. "
                "Run thalamus-score first."
            )
        # Build one document per component: join all example texts
        docs = [" ".join(e.texts) if e.texts else e.name for e in entries]
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        doc_matrix = vectorizer.fit_transform(docs).toarray()
        logger.info(
            "TFIDFSelector: fit on %d components, vocab=%d",
            len(entries), len(vectorizer.vocabulary_),
        )
        return cls(catalog, vectorizer, doc_matrix, entries)

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Return top-k components by TF-IDF cosine similarity.

        Parameters
        ----------
        query:
            Raw user query text.
        budget:
            Budget tier for k selection.  ``None`` defaults to ``"medium"``.
        ordering:
            ``"relevance"`` — most-similar first (default for retrieval).
            ``"bookend"``   — bookend reordering on the relevance-sorted list.
            ``"none"``      — preserve corpus load order.
        """
        if not self.is_ready:
            return None

        k = min(self._catalog.count_for_budget(budget), len(self._entries_order))

        # Vectorize query
        query_vec = self._vectorizer.transform([query]).toarray()  # (1, n_features)
        sims = cosine_similarity(query_vec, self._doc_matrix)[0]   # (n_components,)

        if ordering == "none":
            # Preserve corpus order: take the k components with highest sim but
            # keep them in their original index order.
            top_indices = set(np.argsort(sims)[-k:])
            chosen = [e.name for i, e in enumerate(self._entries_order) if i in top_indices]
        else:
            # Sort by similarity descending
            top_indices = np.argsort(sims)[::-1][:k]
            chosen = [self._entries_order[i].name for i in top_indices]
            if ordering == "bookend":
                chosen = bookend_order(chosen)

        result = self._catalog.as_result(chosen, source="tfidf")
        result["n_components"] = k
        return result

    @property
    def active_path(self) -> str:
        return "tfidf"

    @property
    def is_ready(self) -> bool:
        return len(self._entries_order) > 0
