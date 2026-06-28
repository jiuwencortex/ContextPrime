# oracle_builder/evolutionary/config_builder_step01_load_components.py
# Step 1: load all scoring matrices and extract ComponentInfo + example texts.
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .component_info import ComponentInfo

logger = logging.getLogger(__name__)

_PATTERN_MAP = [
    ("scoring_matrix_skill_*.json", "skill"),
    ("scoring_matrix_mem_*.json",   "memory_section"),
    ("scoring_matrix_tool_*.json",  "tool"),
]


class ComponentsLoader:
    """Load scoring matrices from oracle_dir and return ComponentInfo + example texts."""

    def __init__(self, oracle_dir: Path):
        self._oracle_dir = oracle_dir

    def load(self) -> tuple[list[ComponentInfo], dict[str, list[str]]]:
        """Return (components, example_texts_map) where the map is {name: [texts]}."""
        components: list[ComponentInfo] = []
        example_texts_map: dict[str, list[str]] = {}
        n_failed = 0

        for glob_pat, ctype in _PATTERN_MAP:
            for path in sorted(self._oracle_dir.glob(glob_pat)):
                result = self._load_one(path, ctype)
                if result is None:
                    n_failed += 1
                    continue
                comp, texts = result
                components.append(comp)
                example_texts_map[comp.name] = texts

        if n_failed:
            logger.warning("%d matrix file(s) failed to load and were skipped", n_failed)

        return components, example_texts_map

    @staticmethod
    def _load_one(path: Path, ctype: str) -> tuple[ComponentInfo, list[str]] | None:
        """Parse one scoring matrix JSON."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", path.name, e)
            return None

        rows = data.get("baseline_cross_eval", [])
        if not rows:
            logger.warning("Skipping %s: no baseline_cross_eval rows", path.name)
            return None

        f1_scores = [r["scores"].get("f1", 0.0) for r in rows if "scores" in r]
        synthetic_mean = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # Prefer real-world blended score when available
        real_data = data.get("real_data")
        if real_data and "updated_mean_score" in real_data:
            mean_score = real_data["updated_mean_score"]
            logger.debug("%s: using updated_mean_score %.3f (synthetic was %.3f)", path.name, mean_score, synthetic_mean)
        else:
            mean_score = synthetic_mean
        example_texts = [r["example_input"] for r in rows if r.get("example_input")]

        sample_rows = rows[: min(3, len(rows))]
        body_chars = sum(len(r.get("candidate_output", "")) for r in sample_rows)
        body_chars = body_chars // max(1, len(sample_rows))
        body_tokens = max(50, body_chars // 4)

        name = data.get("component_name") or data.get("skill_name") or path.stem

        comp = ComponentInfo(
            name=name,
            component_type=ctype,
            mean_score=round(mean_score, 4),
            query_centroid=np.zeros(0),
            body_tokens=body_tokens,
            is_group=data.get("is_group", False),
            group_members=data.get("group_members", []),
        )
        return comp, example_texts
