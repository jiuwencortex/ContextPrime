# shared/context_orderer.py
"""Bookend ordering for selected context components.

The "lost-in-the-middle" problem: LLMs pay less attention to items placed in the
centre of a long prompt than to items at the very start or end.  Bookend ordering
mitigates this by placing the most-relevant components at the edges of each list
and the least-relevant ones in the middle.

Reference: Liu et al. (2023), "Lost in the Middle: How Language Models Use Long
Contexts", https://arxiv.org/abs/2307.03172
"""
from __future__ import annotations


def bookend_order(items: list) -> list:
    """Re-order a relevance-sorted list into bookend pattern.

    Input items must be sorted from most relevant to least relevant.
    Output alternates placing items at the front and back of the result so the
    most-relevant item is first, the second-most-relevant is last, the third is
    second, the fourth is second-to-last, and so on.

    Parameters
    ----------
    items:
        List sorted from most relevant (index 0) to least relevant (index -1).

    Returns
    -------
    list
        Same items re-ordered so the most-relevant items appear at both ends
        and the least-relevant items occupy the middle positions.

    Example::

        bookend_order(["A", "B", "C", "D", "E"])
        # A = most relevant, E = least relevant
        # → front bucket: [A, C, E]   (even indices)
        # → back  bucket: [B, D]      (odd indices, reversed → [D, B])
        → ["A", "C", "E", "D", "B"]
    """
    if len(items) <= 2:
        return list(items)

    front: list = []
    back: list = []
    for i, item in enumerate(items):
        if i % 2 == 0:
            front.append(item)
        else:
            back.append(item)

    return front + list(reversed(back))
