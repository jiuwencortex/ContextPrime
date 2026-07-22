from __future__ import annotations

from collections import Counter
from ._tokenizer import _tokenize


def _bigrams(text: str) -> list[str]:
    tokens = _tokenize(text)
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def bigram_f1(output: str, expected: str) -> float:
    """F1 score over bigram (two-word sequence) overlap."""
    pred = Counter(_bigrams(output))
    gold = Counter(_bigrams(expected))
    common = sum((pred & gold).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(gold.values())
    return 2 * precision * recall / (precision + recall)
