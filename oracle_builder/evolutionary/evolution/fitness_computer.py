# oracle_builder/evolutionary/evolution/fitness_computer.py
# Core data structures: ComponentInfo, ContextGenome, fitness function.
from __future__ import annotations

import numpy as np

from .cosine import _cosine
from .context_genome import ContextGenome
from ..component_info import ComponentInfo


def compute_fitness(genome: ContextGenome, components: list[ComponentInfo], query_embedding: np.ndarray,
                    lambda_: float = 0.1, max_tokens: int = 8000,) -> tuple[float, int]:
    """Compute (fitness, context_tokens) for a genome without any LLM calls.

    fitness = Σ (mean_score_i × bit_i × similarity_i) − λ × (total_tokens / max_tokens)

    - mean_score_i   : average F1 across all examples in this component's matrix file
    - similarity_i   : cosine similarity of the cluster centroid to the component's
                       example-query centroid (pre-computed TF-IDF vectors)
    - λ              : penalty weight controlling quality-vs-size tradeoff
    """
    total_score = 0.0
    total_tokens = 0

    for component, included in zip(components, genome.bits):
        if not included:
            continue
        sim = _cosine(query_embedding, component.query_centroid)
        total_score += component.mean_score * max(sim, 0.0)
        total_tokens += component.body_tokens

    size_penalty = lambda_ * (total_tokens / max(max_tokens, 1))
    return total_score - size_penalty, total_tokens
