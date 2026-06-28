# recommendation_matrix/shared/changed_items_determiner.py
# Compare current components against persisted state to decide what needs rebuilding.
from __future__ import annotations

import logging
from pathlib import Path

from .fingerprint import ComponentRecord, component_fingerprint, global_fingerprint
from .state import load_state

logger = logging.getLogger(__name__)


class ChangedItemsDeterminer:
    """Decide which components need rebuilding by comparing fingerprints against persisted state."""

    def __init__(self, matrix_dir: Path, state_file: str = "matrix_state_skills.json"):
        self._matrix_dir = matrix_dir
        self._state_file = state_file

    def determine(
        self, components: list[ComponentRecord], force: bool
    ) -> tuple[list[ComponentRecord], list[str]]:
        """Return (changed_components, skipped_names) based on fingerprint comparison.

        If nothing changed, prints the skip message and returns ([], all_names).
        """
        state = load_state(self._matrix_dir, self._state_file)
        g_fp = global_fingerprint(components)

        if not force and state and state.fingerprint == g_fp:
            print("Fingerprint unchanged — nothing to rebuild.")
            print(f"Skipped: {', '.join(c.name for c in components)}")
            return [], [c.name for c in components]

        if state and not force:
            changed = [
                c for c in components
                if c.name not in state.components
                or state.components[c.name].fingerprint != component_fingerprint(c)
            ]
        else:
            changed = list(components)

        skipped = [c.name for c in components if c not in changed]
        logger.info("Building %d component(s), skipping %d", len(changed), len(skipped))
        return changed, skipped


# Backward-compatible alias
ChangedSkillsDeterminer = ChangedItemsDeterminer
