# recommendation_matrix/memory/scanner.py
# Scan project.md and user.md and return one ComponentRecord per ## section.
from __future__ import annotations

import re
from pathlib import Path

from ..shared.fingerprint import ComponentRecord

_MIN_BODY_LEN = 50  # sections shorter than this are skipped


class MemorySectionScanner:
    """Split project.md and user.md at ## headings; return one ComponentRecord per section.

    Filtering rules:
    - Sections with fewer than 50 characters of body text are skipped (too short to be useful).
    - Sections with ### or deeper headings are skipped (only ## sections are top-level sections).
    - The first line of each file (if it is a # title) is skipped — it is the file name, not a section.
    """

    _SOURCE_FILES = ("project.md", "user.md", "AGENT.md", "SOUL.md", "IDENTITY.md", "HEARTBEAT.md")

    def __init__(self, project_dir: Path):
        self._project_dir = project_dir

    def scan(self) -> list[ComponentRecord]:
        """Scan project_dir for project.md and user.md; split each at ## headings.

        Returns one ComponentRecord per usable section.
        """
        records: list[ComponentRecord] = []
        for filename in self._SOURCE_FILES:
            path = self._project_dir / filename
            if not path.exists():
                continue
            mtime = path.stat().st_mtime
            sections = self._split_sections(path.read_text(encoding="utf-8"))
            for heading, body in sections:
                body = body.strip()
                if len(body) < _MIN_BODY_LEN:
                    continue
                description = self._first_sentence(body)
                safe_name = f"{filename}::{heading}"
                records.append(ComponentRecord(
                    name=safe_name,
                    description=description,
                    body=body,
                    mtime=mtime,
                    directory=self._project_dir,
                    source_file=filename,
                ))
        return records

    def scan_and_filter(self, only: list[str] | None) -> list[ComponentRecord]:
        """Scan and optionally filter by name."""
        sections = self.scan()
        if only:
            sections = [s for s in sections if s.name in only]
        return sections

    @staticmethod
    def _split_sections(text: str) -> list[tuple[str, str]]:
        """Split text at ## headings. Skip # title line at top and ### or deeper sections.

        Returns list of (heading_text, body_text) tuples.
        """
        lines = text.splitlines(keepends=True)
        sections: list[tuple[str, str]] = []
        current_heading: str | None = None
        current_body_lines: list[str] = []

        def flush() -> None:
            if current_heading is not None:
                sections.append((current_heading, "".join(current_body_lines)))

        for line in lines:
            # Match exactly ## (not ### or deeper, not #)
            m = re.match(r"^## (.+)$", line.rstrip("\n\r"))
            if m:
                flush()
                current_heading = m.group(1).strip()
                current_body_lines = []
            else:
                if current_heading is not None:
                    # Skip ### or deeper headings — treat their text as body content
                    current_body_lines.append(line)

        flush()
        return sections

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Return the first sentence or first line of text, truncated to 200 chars."""
        # Try splitting at sentence boundary
        for sep in (".", "!", "?", "\n"):
            idx = text.find(sep)
            if idx > 0:
                return text[: idx + 1].strip()[:200]
        return text.strip()[:200]
