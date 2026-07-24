from __future__ import annotations

from collections import Counter
from ._tokenizer import _tokenize


def token_f1(output: str, expected: str) -> float:
    """F1 score over token overlap between output and expected."""
    pred = Counter(_tokenize(output))
    gold = Counter(_tokenize(expected))
    common = sum((pred & gold).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(gold.values())
    return 2 * precision * recall / (precision + recall)
