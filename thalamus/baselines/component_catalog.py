"""Component catalog: loads the full component pool from scoring matrices.

Baseline selectors need to know:
  - all available component names
  - their type (skill / memory_section / tool)
  - example query texts (for retrieval-based baselines)
  - how many components to return per budget tier (read from context_configs.json)

This module is a read-only view of the oracle directory.  It never writes anything.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Glob pattern → component_type (matches oracle_builder/evolutionary step 1)
_PATTERN_MAP: list[tuple[str, str]] = [
    ("scoring_matrix_skill_*.json", "skill"),
    ("scoring_matrix_mem_*.json", "memory_section"),
    ("scoring_matrix_tool_*.json", "tool"),
]

# Budget tier name → key in context_configs.json cluster entries
_BUDGET_KEY: dict[str, str] = {
    "small": "budget_small",
    "medium": "budget_medium",
    "large": "budget_large",
}


@dataclass
class ComponentEntry:
    """A single component from the scoring matrices."""

    name: str
    component_type: str          # "skill" | "memory_section" | "tool"
    texts: list[str] = field(default_factory=list)  # example_input texts for retrieval
    mean_score: float = 0.0      # synthetic (or blended) mean F1


class ComponentCatalog:
    """Full component pool loaded from oracle_dir scoring matrices.

    Usage::

        catalog = ComponentCatalog.load("/path/to/oracle")
        for entry in catalog.entries():
            print(entry.name, entry.component_type, len(entry.texts))

        # How many components to return for "medium" budget:
        k = catalog.count_for_budget("medium")

        # Convert a flat name list back to {skills, memory, tools}:
        result = catalog.as_result(["skill_a", "mem_section", "tool_x"])
    """

    def __init__(
        self,
        entries: list[ComponentEntry],
        budget_counts: dict[str, int],
    ) -> None:
        self._entries = entries
        self._budget_counts = budget_counts
        # Name → type lookup for as_result()
        self._type_map: dict[str, str] = {e.name: e.component_type for e in entries}

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "ComponentCatalog":
        """Load all scoring matrices and budget counts from *oracle_dir*."""
        oracle_dir = Path(oracle_dir)
        entries = cls._load_entries(oracle_dir)
        budget_counts = cls._load_budget_counts(oracle_dir, len(entries))
        logger.info(
            "ComponentCatalog: %d components, budget_counts=%s",
            len(entries), budget_counts,
        )
        return cls(entries, budget_counts)

    # ── public API ────────────────────────────────────────────────────────────

    def entries(self) -> list[ComponentEntry]:
        """All component entries in load order (skills, memory, tools)."""
        return self._entries

    def count_for_budget(self, budget: str | None) -> int:
        """Number of components to return for a given budget tier.

        Falls back to half the catalog when budget is unknown.
        """
        tier = budget if budget in self._budget_counts else "medium"
        return self._budget_counts.get(tier, max(1, len(self._entries) // 2))

    def as_result(self, names: list[str], source: str = "") -> dict:
        """Convert a flat ordered name list to ``{skills, memory, tools, source}``."""
        skills: list[str] = []
        memory: list[str] = []
        tools: list[str] = []
        for n in names:
            t = self._type_map.get(n, "skill")
            if t == "skill":
                skills.append(n)
            elif t == "memory_section":
                memory.append(n)
            else:
                tools.append(n)
        return {"skills": skills, "memory": memory, "tools": tools, "source": source}

    def __len__(self) -> int:
        return len(self._entries)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_entries(oracle_dir: Path) -> list[ComponentEntry]:
        entries: list[ComponentEntry] = []
        n_failed = 0
        for glob_pat, ctype in _PATTERN_MAP:
            for path in sorted(oracle_dir.glob(glob_pat)):
                entry = ComponentCatalog._load_one(path, ctype)
                if entry is None:
                    n_failed += 1
                else:
                    entries.append(entry)
        if n_failed:
            logger.warning("%d scoring matrix file(s) skipped", n_failed)
        return entries

    @staticmethod
    def _load_one(path: Path, ctype: str) -> ComponentEntry | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            return None

        rows = data.get("baseline_cross_eval", [])
        if not rows:
            logger.warning("Skipping %s: no baseline_cross_eval rows", path.name)
            return None

        name = data.get("component_name") or data.get("skill_name") or path.stem
        texts = [r["example_input"] for r in rows if r.get("example_input")]

        f1_scores = [r["scores"].get("f1", 0.0) for r in rows if "scores" in r]
        mean_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        real = data.get("real_data")
        if real and "updated_mean_score" in real:
            mean_score = real["updated_mean_score"]

        return ComponentEntry(
            name=name,
            component_type=ctype,
            texts=texts,
            mean_score=round(mean_score, 4),
        )

    @staticmethod
    def _load_budget_counts(oracle_dir: Path, n_total: int) -> dict[str, int]:
        """Derive per-budget component counts from context_configs.json.

        Falls back to proportional heuristics when the file is absent.
        """
        config_path = oracle_dir / "context_configs.json"
        if not config_path.exists():
            return _fallback_counts(n_total)

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            clusters: list[dict] = data.get("clusters", [])
            sums: dict[str, list[int]] = {"small": [], "medium": [], "large": []}
            for cluster in clusters:
                for tier, key in _BUDGET_KEY.items():
                    cfg = cluster.get(key, {})
                    total = (
                        len(cfg.get("skills", []))
                        + len(cfg.get("memory", []))
                        + len(cfg.get("tools", []))
                    )
                    if total > 0:
                        sums[tier].append(total)
            counts: dict[str, int] = {}
            for tier, vals in sums.items():
                counts[tier] = round(sum(vals) / len(vals)) if vals else _fallback_counts(n_total)[tier]
            return counts
        except Exception as exc:
            logger.warning("Could not read context_configs.json for budget counts: %s", exc)
            return _fallback_counts(n_total)


def _fallback_counts(n: int) -> dict[str, int]:
    return {
        "small": max(1, n // 4),
        "medium": max(1, n // 2),
        "large": n,
    }
