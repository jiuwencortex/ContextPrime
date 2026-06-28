# oracle_builder/evolutionary/component_info.py
# Core data structures: ComponentInfo, ContextGenome, fitness function.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComponentInfo:
    """Everything the evolutionary search needs to know about one component."""
    name: str
    component_type: str         # "skill" | "memory_section" | "tool"
    mean_score: float           # mean F1 across all example rows in its matrix file
    query_centroid: np.ndarray  # TF-IDF centroid of the component's example_input texts
    body_tokens: int            # estimated body size in tokens (chars / 4)
    is_group: bool = False      # True for tool groups (must be selected together)
    group_members: list[str] = field(default_factory=list)
