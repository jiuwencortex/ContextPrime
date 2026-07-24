"""Baseline: BM25 retrieval — stronger lexical baseline than TF-IDF.

BM25 (Best Match 25 / Okapi BM25) is the de-facto standard lexical retrieval
function.  It outperforms TF-IDF on most IR benchmarks due to length normalization
and a saturating term-frequency function.

Implemented from scratch: no external dependencies beyond the standard library.

Score formula (Robertson et al., 1994):
    score(D, Q) = Σ_{t ∈ Q} IDF(t) · tf(t,D) · (k1+1) / (tf(t,D) + k1·(1-b+b·|D|/avgdl))
    IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)

where N = corpus size, n(t) = document frequency of term t, k1 = 1.5, b = 0.75.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path

from thalamus._shared.context_orderer import bookend_order
from .component_catalog import ComponentCatalog, ComponentEntry

logger = logging.getLogger(__name__)

# Tunable BM25 parameters (standard defaults)
_K1 = 1.5
_B = 0.75

_TOKENIZE_RE = re.compile(r"[^\w]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKENIZE_RE.split(text.lower()) if t]


class BM25Selector:
    """BM25 retrieval baseline — top-k components by BM25 score.

    Usage::

        selector = BM25Selector.load(oracle_dir)
        result = selector.select("Set up CI/CD pipeline", budget="medium")
        # result = {"skills": [...top-k by BM25...], ...}
    """

    def __init__(
        self,
        catalog: ComponentCatalog,
        entries_order: list[ComponentEntry],
        term_freqs: list[Counter],   # per-document term frequency counts
        doc_freqs: Counter,          # term → number of documents containing it
        avg_dl: float,
    ) -> None:
        self._catalog = catalog
        self._entries_order = entries_order
        self._tf = term_freqs
        self._df = doc_freqs
        self._avg_dl = avg_dl
        self._n = len(entries_order)

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "BM25Selector":
        """Load catalog and build BM25 index from component example texts."""
        oracle_dir = Path(oracle_dir)
        catalog = ComponentCatalog.load(oracle_dir)
        entries = catalog.entries()
        if not entries:
            raise FileNotFoundError(
                f"No scoring matrices found in {oracle_dir}. "
                "Run thalamus-score first."
            )

        docs_tokens: list[list[str]] = [
            _tokenize(" ".join(e.texts) if e.texts else e.name)
            for e in entries
        ]
        term_freqs = [Counter(tokens) for tokens in docs_tokens]

        doc_freqs: Counter = Counter()
        for tf in term_freqs:
            doc_freqs.update(tf.keys())

        total_dl = sum(len(tokens) for tokens in docs_tokens)
        avg_dl = total_dl / len(docs_tokens) if docs_tokens else 1.0

        logger.info(
            "BM25Selector: indexed %d components, vocab=%d, avg_dl=%.1f",
            len(entries), len(doc_freqs), avg_dl,
        )
        return cls(catalog, entries, term_freqs, doc_freqs, avg_dl)

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Return top-k components by BM25 score.

        Parameters
        ----------
        query:
            Raw user query text.
        budget:
            Budget tier for k selection.  ``None`` defaults to ``"medium"``.
        ordering:
            ``"relevance"`` — highest-score first.
            ``"bookend"``   — bookend reordering on the relevance-sorted list.
            ``"none"``      — preserve corpus load order among top-k.
        """
        if not self.is_ready:
            return None

        k = min(self._catalog.count_for_budget(budget), len(self._entries_order))
        scores = self._score(query)

        if ordering == "none":
            top_set = set(sorted(range(self._n), key=lambda i: scores[i], reverse=True)[:k])
            chosen = [e.name for i, e in enumerate(self._entries_order) if i in top_set]
        else:
            top_indices = sorted(range(self._n), key=lambda i: scores[i], reverse=True)[:k]
            chosen = [self._entries_order[i].name for i in top_indices]
            if ordering == "bookend":
                chosen = bookend_order(chosen)

        result = self._catalog.as_result(chosen, source="bm25")
        result["n_components"] = k
        return result

    @property
    def active_path(self) -> str:
        return "bm25"

    @property
    def is_ready(self) -> bool:
        return len(self._entries_order) > 0

    # ── BM25 scoring ──────────────────────────────────────────────────────────

    def _score(self, query: str) -> list[float]:
        """Return BM25 scores for all documents."""
        q_terms = _tokenize(query)
        scores = [0.0] * self._n
        for term in q_terms:
            idf = self._idf(term)
            for i, tf_counter in enumerate(self._tf):
                tf_val = tf_counter.get(term, 0)
                if tf_val == 0:
                    continue
                dl = sum(tf_counter.values())
                denom = tf_val + _K1 * (1 - _B + _B * dl / self._avg_dl)
                scores[i] += idf * (tf_val * (_K1 + 1)) / denom
        return scores

    def _idf(self, term: str) -> float:
        n_t = self._df.get(term, 0)
        return math.log((self._n - n_t + 0.5) / (n_t + 0.5) + 1)
