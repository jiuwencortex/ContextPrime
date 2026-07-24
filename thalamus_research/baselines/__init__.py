"""Baseline context selectors for research evaluation (Phase R1).

All baselines implement :class:`SelectorProtocol` and can be used
interchangeably with :class:`~thalamus.selection.ContextSelector`
in evaluation harnesses or A/B tests.

Baselines
---------
- :class:`AllSelector`    — return every component; quality upper bound
- :class:`RandomSelector` — random k components; null hypothesis
- :class:`TFIDFSelector`  — TF-IDF cosine similarity top-k; current industry practice
- :class:`BM25Selector`   — BM25 top-k; strongest lexical baseline
- :class:`DenseSelector`  — sentence-transformer cosine top-k; strongest retrieval baseline
                            (requires ``pip install thalamus[sentence]``)

Usage::

    from thalamus_research.baselines import TFIDFSelector, BM25Selector

    tfidf = TFIDFSelector.load("/oracle")
    bm25  = BM25Selector.load("/oracle")

    result = tfidf.select("Set up CI/CD", budget="medium")
    # {"skills": [...], "memory": [...], "tools": [...], "source": "tfidf"}

Evaluation harness::

    from thalamus.selection import ContextSelector
    from thalamus_research.baselines import TFIDFSelector, BM25Selector, SelectorProtocol

    selectors: dict[str, SelectorProtocol] = {
        "thalamus": ContextSelector.load(oracle_dir),
        "tfidf":    TFIDFSelector.load(oracle_dir),
        "bm25":     BM25Selector.load(oracle_dir),
    }
    for name, sel in selectors.items():
        result = sel.select(query)
        print(name, "→", result)
"""

from .protocol import SelectorProtocol
from .component_catalog import ComponentCatalog, ComponentEntry
from .all_selector import AllSelector
from .random_selector import RandomSelector
from .tfidf_selector import TFIDFSelector
from .bm25_selector import BM25Selector

# DenseSelector imported lazily to avoid hard dependency on sentence-transformers.
# Use: from thalamus_research.baselines.dense_selector import DenseSelector

__all__ = [
    "SelectorProtocol",
    "ComponentCatalog",
    "ComponentEntry",
    "AllSelector",
    "RandomSelector",
    "TFIDFSelector",
    "BM25Selector",
]
