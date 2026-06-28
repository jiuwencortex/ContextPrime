from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
