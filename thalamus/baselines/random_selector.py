"""Baseline: return k randomly sampled components.

Null hypothesis for all quality experiments.  If a structured selector does not
beat random sampling on quality-per-token, the selection strategy adds no value.

k is derived from the oracle's budget tier counts (loaded from context_configs.json)
so the comparison with structured selectors is on equal-token-footprint terms.
"""
from __future__ import annotations

import random
from pathlib import Path

from .component_catalog import ComponentCatalog


class RandomSelector:
    """Return k randomly sampled components where k matches the oracle budget tier.

    Usage::

        selector = RandomSelector.load(oracle_dir, seed=42)
        result = selector.select("Fix login bug", budget="medium")
        # result = {"skills": [...k random skills/memory/tools...], ...}
    """

    def __init__(self, catalog: ComponentCatalog, seed: int | None = None) -> None:
        self._catalog = catalog
        self._rng = random.Random(seed)

    @classmethod
    def load(
        cls,
        oracle_dir: str | Path,
        seed: int | None = None,
    ) -> "RandomSelector":
        """Load the component catalog and initialize the RNG.

        Parameters
        ----------
        oracle_dir:
            Directory containing scoring matrices and optionally context_configs.json.
        seed:
            Fixed seed for reproducible experiments.  ``None`` uses system entropy.
        """
        return cls(ComponentCatalog.load(Path(oracle_dir)), seed=seed)

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Return k randomly sampled components.

        Query and ordering are ignored.  Budget controls k via the oracle's
        precomputed budget counts.
        """
        if not self.is_ready:
            return None
        entries = self._catalog.entries()
        k = min(self._catalog.count_for_budget(budget), len(entries))
        chosen = self._rng.sample(entries, k)
        result = self._catalog.as_result([e.name for e in chosen], source="random")
        result["n_components"] = k
        return result

    @property
    def active_path(self) -> str:
        return "random"

    @property
    def is_ready(self) -> bool:
        return len(self._catalog) > 0
