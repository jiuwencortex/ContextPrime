"""Unified context selector: Path B (classifier) → Path A (cluster) → None.

This facade is the single entry point for jiuwenswarm and other callers.
It handles path selection, fallback, and optional imports transparently.

Path priority
-------------
1. **Path B — classifier** (``classifier_current.pkl`` present):
   Embeds the query via the same TF-IDF/sentence-transformer model used at
   build time, then runs the trained logistic regression.  Most accurate once
   enough operational turn logs have been collected (~100-500 turns).

2. **Path A — cluster lookup** (``context_configs.json`` + ``context_configs.pkl``
   present): embeds query → nearest K-means cluster → pre-computed optimal
   component set.  Available immediately after ``thalamus-oracle`` runs.
   Supports budget tiers (small / medium / large) and bookend ordering.

3. **None**: oracle dir is missing or empty.  Caller should fall back to its
   own default behavior (e.g. load all skills).

Usage::

    from thalamus.selection import ContextSelector

    selector = ContextSelector.load("/path/to/oracle")
    result = selector.select("Set up CI/CD with Docker")
    # result = {"skills": [...], "memory": [...], "tools": [...], ...}
    # result is None if no oracle is available

    print(selector.active_path)   # "classifier" | "cluster" | "none"
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextSelector:
    """Unified selector with automatic Path B → Path A → None fallback.

    Instantiate via :meth:`load` rather than directly.
    """

    def __init__(
        self,
        cluster_selector,   # ClusterSelector | None
        classifier_selector,  # ClassifierSelector | None
    ) -> None:
        self._cluster_selector = cluster_selector
        self._classifier_selector = classifier_selector

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "ContextSelector":
        """Load whichever paths are available in *oracle_dir*.

        Neither path is required.  Missing files are silently skipped so that
        a cold-start deployment (no classifier yet) works out of the box.

        Parameters
        ----------
        oracle_dir:
            Directory produced by ``thalamus-score`` + ``thalamus-oracle``.

        Returns
        -------
        ContextSelector
            Always returns an instance; check :attr:`active_path` to see what
            is actually available.
        """
        oracle_dir = Path(oracle_dir)

        cluster_selector = cls._try_load_cluster(oracle_dir)
        classifier_selector = cls._try_load_classifier(oracle_dir)

        active = (
            "classifier" if classifier_selector is not None
            else "cluster" if cluster_selector is not None
            else "none"
        )
        logger.info("ContextSelector loaded from %s (active_path=%s)", oracle_dir, active)

        return cls(cluster_selector, classifier_selector)

    # ── public API ────────────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Select the optimal context components for *query*.

        Parameters
        ----------
        query:
            Raw user query text.
        budget:
            ``"small"``, ``"medium"``, or ``"large"``.  ``None`` (default)
            delegates to :class:`~by_clusters.budget_estimator.BudgetEstimator`
            which infers the tier from query length and complexity markers.
            Ignored when Path B (classifier) is active.
        ordering:
            Component ordering strategy for Path A.  One of:

            - ``"bookend"`` *(default)* — most-relevant components at the
              beginning and end of each list; mitigates lost-in-the-middle
              attention decay.
            - ``"relevance"`` — most-relevant first (pre-sorted at build time).
            - ``"none"`` — preserve insertion order.

            Path B results are returned in probability-descending order and
            are not reordered.

        Returns
        -------
        dict or None
            On success: ``{"skills": [...], "memory": [...], "tools": [...], ...}``

            Returns ``None`` when no oracle is available or both paths fail.
            Callers should fall back to their default context-loading behavior.
        """
        # ── Path B: classifier (preferred) ────────────────────────────────────
        if self._classifier_selector is not None and self._cluster_selector is not None:
            try:
                clusterer = self._cluster_selector._get_clusterer()
                embedding = clusterer.transform(query)
                result = self._classifier_selector.select(embedding)
                logger.debug(
                    "Path B selected: skills=%s memory=%s tools=%s confidence=%.3f",
                    result.get("skills"), result.get("memory"),
                    result.get("tools"), result.get("confidence", 0),
                )
                return result
            except Exception:
                logger.warning(
                    "Path B (classifier) failed; falling back to Path A",
                    exc_info=True,
                )

        # ── Path A: cluster lookup ─────────────────────────────────────────────
        if self._cluster_selector is not None:
            try:
                if budget is None:
                    result = self._cluster_selector.select_auto(query, ordering=ordering)
                else:
                    result = self._cluster_selector.select(query, budget=budget,
                                                           ordering=ordering)
                logger.debug(
                    "Path A selected: skills=%s memory=%s tools=%s",
                    result.get("skills"), result.get("memory"), result.get("tools"),
                )
                return result
            except Exception:
                logger.warning("Path A (cluster) failed", exc_info=True)

        return None

    @property
    def active_path(self) -> str:
        """Which selection path is active: ``"classifier"``, ``"cluster"``, or ``"none"``."""
        if self._classifier_selector is not None:
            return "classifier"
        if self._cluster_selector is not None:
            return "cluster"
        return "none"

    @property
    def is_ready(self) -> bool:
        """Return True if at least one selection path is available."""
        return self.active_path != "none"

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _try_load_cluster(oracle_dir: Path):
        """Attempt to load ClusterSelector; return None on any failure."""
        from .by_clusters.cluster_selector import ClusterSelector
        try:
            return ClusterSelector.load(oracle_dir)
        except FileNotFoundError:
            logger.debug("Path A not available: context_configs.json absent in %s", oracle_dir)
            return None
        except Exception:
            logger.warning("Path A failed to load from %s", oracle_dir, exc_info=True)
            return None

    @staticmethod
    def _try_load_classifier(oracle_dir: Path):
        """Attempt to load ClassifierSelector; return None on any failure."""
        from .by_classifier.classifier_selector import ClassifierSelector
        try:
            return ClassifierSelector.load(oracle_dir)
        except FileNotFoundError:
            logger.debug("Path B not available: no classifier pkl in %s", oracle_dir)
            return None
        except Exception:
            logger.warning("Path B failed to load from %s", oracle_dir, exc_info=True)
            return None
