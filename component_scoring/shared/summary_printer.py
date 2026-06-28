# recommendation_matrix/shared/summary_printer.py
# Print the final human-readable summary.
from __future__ import annotations

from .fingerprint import ComponentRecord


class SummaryPrinter:
    """Print how many components were rebuilt, skipped, and total LLM calls."""

    def __init__(self) -> None:
        pass

    def print(self, changed: list[ComponentRecord], skipped: list[str], llm_calls: int) -> None:
        """Print the rebuild summary."""
        print(f"\nDone. Rebuilt: {len(changed)}, skipped: {len(skipped)}")
        if changed:
            print(f"  Rebuilt: {', '.join(c.name for c in changed)}")
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}")
        print(f"LLM calls: {llm_calls}")
