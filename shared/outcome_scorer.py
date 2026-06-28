# online/outcome_scorer.py
# Compute a scalar outcome quality [0, 1] from a logged turn's signals.
from __future__ import annotations


def compute_outcome_quality(turn: dict) -> float:
    """Return a quality score in [0, 1] for a turn's outcome.

    Signal priority:
    1. Explicit user rating (strongest) — "positive" → 1.0, "negative" → 0.0
    2. Implicit signals (additive on a 0.5 baseline):
       +0.2  task_completed
       -0.3  follow_up_correction
       +max(0, 0.1 - 0.02 × conversation_length)   (shorter = better)

    Returns a value clamped to [0, 1].
    """
    outcome = turn.get("outcome", {})
    rating = outcome.get("explicit_rating")

    if rating == "positive":
        return 1.0
    if rating == "negative":
        return 0.0

    # Implicit signals
    signals = outcome.get("implicit_signals", {})
    score = 0.5

    if signals.get("task_completed", False):
        score += 0.2
    if signals.get("follow_up_correction", False):
        score -= 0.3

    length = signals.get("conversation_length", 1)
    score += max(0.0, 0.1 - 0.02 * length)

    return max(0.0, min(1.0, score))


class OutcomeScorer:
    """Thin wrapper around compute_outcome_quality for batch use."""

    def score(self, turn: dict) -> float:
        return compute_outcome_quality(turn)

    def score_batch(self, turns: list[dict]) -> list[float]:
        return [compute_outcome_quality(t) for t in turns]
