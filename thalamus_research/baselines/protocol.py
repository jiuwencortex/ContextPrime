"""Selector protocol: common interface for ContextSelector and all baselines.

Any class that implements ``select()``, ``active_path``, and ``is_ready`` satisfies
this protocol and can be used interchangeably with ``ContextSelector`` in jiuwenswarm
or an evaluation harness.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SelectorProtocol(Protocol):
    """Common interface shared by ContextSelector and all baseline selectors.

    Usage::

        from thalamus_research.baselines import SelectorProtocol
        from thalamus.selection import ContextSelector
        from thalamus_research.baselines import TFIDFSelector

        def evaluate(selector: SelectorProtocol, query: str) -> dict | None:
            return selector.select(query)

        # Both satisfy the protocol:
        evaluate(ContextSelector.load(oracle_dir), query)
        evaluate(TFIDFSelector.load(oracle_dir), query)
    """

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Select context components for a query.

        Parameters
        ----------
        query:
            Raw user query text.
        budget:
            ``"small"``, ``"medium"``, or ``"large"``.  ``None`` lets the
            selector infer the appropriate tier.
        ordering:
            ``"relevance"``, ``"bookend"``, or ``"none"``.  Selectors that
            have no relevance scores may ignore this parameter.

        Returns
        -------
        dict or None
            ``{"skills": [...], "memory": [...], "tools": [...], "source": str}``
            or ``None`` if the selector is not ready.
        """
        ...

    @property
    def active_path(self) -> str:
        """Short identifier for this selector, e.g. ``"tfidf"``, ``"bm25"``."""
        ...

    @property
    def is_ready(self) -> bool:
        """``True`` if the selector can produce results."""
        ...
