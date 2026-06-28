# recommendation_matrix/shared/state.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ComponentState:
    fingerprint: str
    n_examples: int
    built_at: str
    mean_score: float = 0.0


# Backward-compatible alias
SkillState = ComponentState


@dataclass
class MatrixState:
    """Persistent state for a recommendation matrix (skills, memory, or tools)."""
    fingerprint: str          # global fingerprint at last build
    built_at: str
    component_count: int
    component_type: str       # "skill" | "memory_section" | "tool"
    components: dict[str, ComponentState] = field(default_factory=dict)

    # Backward-compat property: old code read state.skills
    @property
    def skills(self) -> dict[str, ComponentState]:
        return self.components

    @property
    def skill_count(self) -> int:
        return self.component_count


def load_state(matrix_dir: Path, state_file: str = "matrix_state_skills.json") -> MatrixState | None:
    path = matrix_dir / state_file
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    components = {k: ComponentState(**v) for k, v in data.get("components", {}).items()}
    # Legacy format: state files written by old skill_matrix used "skills" key
    if not components and "skills" in data:
        components = {k: ComponentState(**v) for k, v in data["skills"].items()}
    return MatrixState(
        fingerprint=data["fingerprint"],
        built_at=data["built_at"],
        component_count=data.get("component_count", data.get("skill_count", len(components))),
        component_type=data.get("component_type", "skill"),
        components=components,
    )


def save_state(matrix_dir: Path, state: MatrixState, state_file: str = "matrix_state_skills.json") -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "fingerprint": state.fingerprint,
        "built_at": state.built_at,
        "component_count": state.component_count,
        "component_type": state.component_type,
        "components": {k: asdict(v) for k, v in state.components.items()},
    }
    (matrix_dir / state_file).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
