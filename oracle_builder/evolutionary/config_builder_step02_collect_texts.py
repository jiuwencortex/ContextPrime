# oracle_builder/evolutionary/config_builder_step02_collect_texts.py
# Step 2: collect all example texts across all components into a single flat list.
from __future__ import annotations

from .component_info import ComponentInfo


class TextsCollector:
    """Gather all example_input texts from the component map into one flat list."""

    def collect(self, components: list[ComponentInfo], example_texts_map: dict[str, list[str]]) -> list[str]:
        """Return a flat list of every example text."""
        all_texts: list[str] = []
        for comp in components:
            all_texts.extend(example_texts_map.get(comp.name, []))
        return all_texts
