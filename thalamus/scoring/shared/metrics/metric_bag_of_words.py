from __future__ import annotations

from ._tokenizer import _tokenize


def bag_of_words(output: str, expected: str) -> float:
    """Jaccard similarity over unique token sets."""
    pred = set(_tokenize(output))
    gold = set(_tokenize(expected))
    union = pred | gold
    return len(pred & gold) / len(union) if union else 0.0
