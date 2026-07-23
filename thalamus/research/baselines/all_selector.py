"""Baseline: return every component unconditionally.

Upper bound on recall, lower bound on efficiency.  Useful as the quality
ceiling in ablation tables — no selection strategy should do better than this
on quality (only on token efficiency).
"""
from __future__ import annotations

from pathlib import Path

from .component_catalog import ComponentCatalog


class AllSelector:
    """Return the full component pool for every query, ignoring budget.

    Usage::

        selector = AllSelector.load(oracle_dir)
        result = selector.select("Fix the login bug")
        # result = {"skills": [...all skills...], "memory": [...], "tools": [...]}
    """

    def __init__(self, catalog: ComponentCatalog) -> None:
        self._catalog = catalog

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "AllSelector":
        """Load the component catalog from *oracle_dir*."""
        return cls(ComponentCatalog.load(Path(oracle_dir)))

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Return all components.  Query, budget, and ordering are ignored."""
        if not self.is_ready:
            return None
        skills, memory, tools = [], [], []
        for e in self._catalog.entries():
            if e.component_type == "skill":
                skills.append(e.name)
            elif e.component_type == "memory_section":
                memory.append(e.name)
            else:
                tools.append(e.name)
        return {"skills": skills, "memory": memory, "tools": tools, "source": "all"}

    @property
    def active_path(self) -> str:
        return "all"

    @property
    def is_ready(self) -> bool:
        return len(self._catalog) > 0
