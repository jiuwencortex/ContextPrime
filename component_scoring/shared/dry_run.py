# recommendation_matrix/shared/dry_run.py
# Standalone dry-run logic: show what would be rebuilt without calling the LLM.
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .fingerprint import component_fingerprint, global_fingerprint
from .state import load_state


def run_dry_run(
    components: list,
    matrix_dir: Path,
    force: bool,
    state_file: str = "matrix_state_skills.json",
    label: str = "component",
) -> None:
    """Print a rebuild plan without making any LLM calls.

    components: list of ComponentRecord objects already scanned
    """
    state = load_state(matrix_dir, state_file)
    g_fp = global_fingerprint(components)
    print(f"Found {len(components)} {label}(s)")
    if state and state.fingerprint == g_fp and not force:
        print(f"Nothing to rebuild — global fingerprint unchanged.")
        return

    print(f"\n{label.capitalize():<40} {'Action'}")
    print("-" * 50)
    for c in components:
        if (
            not state
            or c.name not in state.components
            or state.components[c.name].fingerprint != component_fingerprint(c)
            or force
        ):
            print(f"  {c.name:<38} BUILD")
        else:
            print(f"  {c.name:<38} skip")
