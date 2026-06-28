from __future__ import annotations


def length_ratio(output: str, expected: str) -> float:
    """Ratio of shorter text length to longer text length."""
    lo, le = len(output.strip()), len(expected.strip())
    if lo == 0 or le == 0:
        return 0.0
    return min(lo, le) / max(lo, le)
