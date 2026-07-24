"""Baseline: dense embedding retrieval — best commonly-used retrieval baseline.

Uses a sentence-transformer model to embed both the component corpus and the
query, then ranks components by cosine similarity of dense vectors.  This is
the strongest single-component retrieval baseline: it handles paraphrase and
semantic variation that TF-IDF and BM25 miss.

Requires: ``pip install thalamus[sentence]`` (sentence-transformers >= 2.2).

A structured selector (Thalamus Path A or B) should beat dense retrieval on
tasks where component *interactions* matter — dense retrieval still treats each
component independently.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from thalamus.shared.context_orderer import bookend_order
from .component_catalog import ComponentCatalog, ComponentEntry

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class DenseSelector:
    """Sentence-transformer dense retrieval baseline — top-k by cosine similarity.

    Usage::

        # Requires sentence-transformers installed
        selector = DenseSelector.load(oracle_dir)
        result = selector.select("Set up CI/CD pipeline", budget="medium")
    """

    def __init__(
        self,
        catalog: ComponentCatalog,
        entries_order: list[ComponentEntry],
        doc_embeddings: np.ndarray,   # shape (n_components, d_embed)
        model_name: str,
        _model,                        # SentenceTransformer instance (lazy)
    ) -> None:
        self._catalog = catalog
        self._entries_order = entries_order
        self._doc_embeddings = doc_embeddings
        self._model_name = model_name
        self._model = _model

    @classmethod
    def load(
        cls,
        oracle_dir: str | Path,
        model_name: str = _DEFAULT_MODEL,
    ) -> "DenseSelector":
        """Load catalog, encode all component texts, and prepare for retrieval.

        Parameters
        ----------
        oracle_dir:
            Directory containing scoring matrices and optionally context_configs.json.
        model_name:
            Sentence-transformer model name (default: ``all-MiniLM-L6-v2``).

        Raises
        ------
        ImportError
            If ``sentence-transformers`` is not installed.
        FileNotFoundError
            If no scoring matrices are found in *oracle_dir*.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "DenseSelector requires sentence-transformers. "
                "Install with: pip install thalamus[sentence]"
            ) from exc

        oracle_dir = Path(oracle_dir)
        catalog = ComponentCatalog.load(oracle_dir)
        entries = catalog.entries()
        if not entries:
            raise FileNotFoundError(
                f"No scoring matrices found in {oracle_dir}. "
                "Run thalamus-score first."
            )

        model = SentenceTransformer(model_name)
        docs = [" ".join(e.texts) if e.texts else e.name for e in entries]
        doc_embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_embeddings = doc_embeddings / np.maximum(norms, 1e-10)

        logger.info(
            "DenseSelector: encoded %d components with %s (d=%d)",
            len(entries), model_name, doc_embeddings.shape[1],
        )
        return cls(catalog, entries, doc_embeddings, model_name, model)

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Return top-k components by dense cosine similarity.

        Parameters
        ----------
        query:
            Raw user query text.
        budget:
            Budget tier for k selection.  ``None`` defaults to ``"medium"``.
        ordering:
            ``"relevance"`` — most-similar first.
            ``"bookend"``   — bookend reordering.
            ``"none"``      — preserve corpus load order among top-k.
        """
        if not self.is_ready:
            return None

        k = min(self._catalog.count_for_budget(budget), len(self._entries_order))

        query_emb = self._model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        norm = np.linalg.norm(query_emb)
        query_emb = query_emb / max(norm, 1e-10)
        sims = self._doc_embeddings @ query_emb  # (n_components,)

        if ordering == "none":
            top_set = set(np.argsort(sims)[::-1][:k].tolist())
            chosen = [e.name for i, e in enumerate(self._entries_order) if i in top_set]
        else:
            top_indices = np.argsort(sims)[::-1][:k]
            chosen = [self._entries_order[i].name for i in top_indices]
            if ordering == "bookend":
                chosen = bookend_order(chosen)

        result = self._catalog.as_result(chosen, source="dense")
        result["n_components"] = k
        return result

    @property
    def active_path(self) -> str:
        return "dense"

    @property
    def is_ready(self) -> bool:
        return len(self._entries_order) > 0
