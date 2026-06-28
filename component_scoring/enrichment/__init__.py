"""Enrichment: blend real interaction logs into component scoring matrix scores.

  - ScoreEnricher : read interaction_logs turns, update scoring_matrix_*.json

TurnLogger and OutcomeScorer now live in jiuwenswarm.tools.shared
(shared between this package and oracle_builder.policy).

Usage:
    from jiuwenswarm.tools.component_scoring.enrichment import ScoreEnricher
    from jiuwenswarm.tools.shared import TurnLogger, OutcomeScorer
"""

from jiuwenswarm.thalamus.shared import OutcomeScorer, TurnLogger, compute_outcome_quality
from .score_enricher import ScoreEnricher

__all__ = [
    "TurnLogger",
    "OutcomeScorer",
    "compute_outcome_quality",
    "ScoreEnricher",
]
