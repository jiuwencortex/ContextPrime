# recommendation_matrix/skills/scanner.py
# Scan skills directory and return ComponentRecord list.
from __future__ import annotations

from pathlib import Path

import yaml

from ..shared.fingerprint import ComponentRecord


class ExistingSkillsScanner:
    """Scan a directory tree for skill subdirectories containing SKILL.md."""

    def __init__(self, skills_dir: Path):
        self._skills_dir = skills_dir

    def scan(self) -> list[ComponentRecord]:
        """Scan skills_dir for subdirectories containing SKILL.md.

        Returns one ComponentRecord per skill.
        SKILL.md is the only raw input that exists — there are no pre-existing queries
        or test cases anywhere. All test data is generated from these files.
        """
        records: list[ComponentRecord] = []
        for skill_dir in sorted(self._skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_dir.is_dir() or not skill_md.exists():
                continue
            name, description, body = self._parse_skill_md(skill_md)
            if not name:
                continue
            records.append(ComponentRecord(
                name=name,
                description=description,
                body=body,
                mtime=skill_md.stat().st_mtime,
                directory=skill_dir,
            ))
        return records

    def scan_and_filter(self, only: list[str] | None) -> list[ComponentRecord]:
        """Scan disk for skills and optionally filter by name."""
        skills = self.scan()
        if only:
            skills = [s for s in skills if s.name in only]
        return skills

    def _parse_skill_md(self, path: Path) -> tuple[str, str, str]:
        """Parse YAML frontmatter and body from SKILL.md.

        Returns (name, description, body).
        body = everything after the closing --- of the frontmatter block.
        Falls back gracefully if frontmatter is missing or malformed.
        """
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    meta = {}
                body = parts[2].lstrip()
                return (
                    meta.get("name", path.parent.name),
                    meta.get("description", ""),
                    body,
                )
        return path.parent.name, "", text
