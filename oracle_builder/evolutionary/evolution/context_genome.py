# oracle_builder/evolutionary/evolution/context_genome.py
# Core data structures: ComponentInfo, ContextGenome, fitness function.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..component_info import ComponentInfo
from .cosine import _cosine


@dataclass
class ContextGenome:
    """A candidate context configuration — a bitmask over all available components."""
    bits: np.ndarray        # shape (n_components,), dtype bool
    fitness: float = 0.0
    context_tokens: int = 0

    def included_names(self, components: list[ComponentInfo]) -> list[str]:
        return [c.name for c, b in zip(components, self.bits) if b]

    def included_by_type(self, components: list[ComponentInfo], ctype: str) -> list[str]:
        return [c.name for c, b in zip(components, self.bits) if b and c.component_type == ctype]

    def to_config(self, components: list[ComponentInfo]) -> dict:
        """Serialize into the format written to context_configs.json."""
        return {"skills":  self.included_by_type(components, "skill"),
                "memory":  self.included_by_type(components, "memory_section"),
                "tools":   self.included_by_type(components, "tool"),
                "fitness": round(self.fitness, 4),
                "context_tokens": self.context_tokens}

    def to_config_sorted(
        self,
        components: list[ComponentInfo],
        query_embedding: np.ndarray,
    ) -> dict:
        """Like to_config() but sorts each type-list by individual relevance (descending).

        Relevance for component i = mean_score_i × cosine(query_embedding, query_centroid_i).
        Storing components in relevance order lets ClusterSelector apply bookend reordering
        at query time without needing per-component scores at runtime.
        """
        def _relevance(c: ComponentInfo) -> float:
            sim = _cosine(query_embedding, c.query_centroid)
            return c.mean_score * max(sim, 0.0)

        included = [
            (c, _relevance(c))
            for c, b in zip(components, self.bits)
            if b
        ]
        included.sort(key=lambda x: x[1], reverse=True)

        skills = [c.name for c, _ in included if c.component_type == "skill"]
        memory = [c.name for c, _ in included if c.component_type == "memory_section"]
        tools  = [c.name for c, _ in included if c.component_type == "tool"]

        return {
            "skills":         skills,
            "memory":         memory,
            "tools":          tools,
            "fitness":        round(self.fitness, 4),
            "context_tokens": self.context_tokens,
        }
