# recommendation_matrix/shared/state_saver.py
# Persist the updated matrix state to disk.
from __future__ import annotations

from pathlib import Path

from .fingerprint import ComponentRecord, global_fingerprint
from .state import ComponentState, MatrixState, load_state, now_iso, save_state


class StateSaver:
    """Carry over unchanged component states and write the updated state file."""

    def __init__(
        self,
        matrix_dir: Path,
        state_file: str = "matrix_state_skills.json",
        component_type: str = "skill",
    ):
        self._matrix_dir = matrix_dir
        self._state_file = state_file
        self._component_type = component_type

    def save(
        self,
        all_components: list[ComponentRecord],
        skipped_names: list[str],
        new_states: dict[str, ComponentState],
    ) -> None:
        """Carry over unchanged component states and write the updated state file."""
        state = load_state(self._matrix_dir, self._state_file)
        g_fp = global_fingerprint(all_components)

        carried: dict[str, ComponentState] = {}
        if state:
            carried = {
                k: v for k, v in state.components.items()
                if k in skipped_names and k not in new_states
            }

        save_state(
            self._matrix_dir,
            MatrixState(
                fingerprint=g_fp,
                built_at=now_iso(),
                component_count=len(all_components),
                component_type=self._component_type,
                components={**carried, **new_states},
            ),
            state_file=self._state_file,
        )


# Backward-compatible alias
MatrixStateSaver = StateSaver
