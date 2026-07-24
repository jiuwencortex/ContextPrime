"""Phase R4 — Set-level interaction feature extractor.

Computes a fixed-length numeric feature vector for a candidate component set.
The vector captures both per-component statistics and set-level interaction
signals (pairwise co-inclusion) that are invisible to marginal-score models.

Feature layout (14 + n_cluster_buckets floats)
-----------------------------------------------
 0   mean_score              mean ComponentCatalog score for set
 1   std_score               std dev of scores
 2   min_score               minimum score in set
 3   max_score               maximum score in set
 4   n_components            total |S|
 5   n_skills                count of skill-type components
 6   n_tools                 count of tool-type components
 7   n_memory                count of memory-type components
 8   n_other                 count of other-type components
 9   mean_co_inclusion       mean pairwise co-inclusion (or 0 if no extractor)
10   min_co_inclusion        min pairwise co-inclusion (or 0)
11   max_co_inclusion        max pairwise co-inclusion (or 0)
12   cluster_id              raw cluster integer (for tree-based models)
13   cluster_id_norm         cluster_id / 100  (for linear models)

Usage::

    from thalamus_research.set_quality.interaction_features import compute_feature_vector

    fvec = compute_feature_vector(
        component_names=["skill_a", "tool_b"],
        cluster_id=3,
        catalog=catalog,
        extractor=co_inclusion_extractor,  # or None
    )
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_FEATURE_DIM = 14


def compute_feature_vector(
    component_names: list[str],
    cluster_id: int,
    catalog,            # ComponentCatalog
    extractor=None,     # CoInclusionExtractor | None
) -> list[float]:
    """Return a fixed-length feature vector for *component_names*.

    Parameters
    ----------
    component_names:
        Flat list of component identifiers in the candidate set.
    cluster_id:
        The query cluster this set is evaluated against.
    catalog:
        :class:`~thalamus_research.baselines.component_catalog.ComponentCatalog`
        for score and type lookups.
    extractor:
        Optional
        :class:`~thalamus_research.cross_path.co_inclusion_extractor.CoInclusionExtractor`.
        If *None*, co-inclusion features are set to 0.

    Returns
    -------
    list[float] of length :data:`_FEATURE_DIM` (14).
    """
    if not component_names:
        return [0.0] * _FEATURE_DIM

    # Per-component scores
    scores: list[float] = []
    n_skills = n_tools = n_memory = n_other = 0

    for name in component_names:
        try:
            score = catalog.score(name)
            ctype = catalog.component_type(name)
        except Exception:
            score = 0.0
            ctype = "other"
        scores.append(score)
        if ctype == "skill":
            n_skills += 1
        elif ctype == "tool":
            n_tools += 1
        elif ctype == "memory":
            n_memory += 1
        else:
            n_other += 1

    arr = np.array(scores, dtype=float)
    mean_score = float(arr.mean())
    std_score = float(arr.std()) if len(arr) > 1 else 0.0
    min_score = float(arr.min())
    max_score = float(arr.max())

    # Co-inclusion features
    if extractor is not None and len(component_names) > 1:
        try:
            pairs = _all_pairwise_co_inclusion(component_names, extractor)
            mean_co = float(np.mean(pairs)) if pairs else 0.0
            min_co = float(np.min(pairs)) if pairs else 0.0
            max_co = float(np.max(pairs)) if pairs else 0.0
        except Exception:
            logger.debug("Co-inclusion lookup failed", exc_info=True)
            mean_co = min_co = max_co = 0.0
    else:
        mean_co = min_co = max_co = 0.0

    fvec: list[float] = [
        mean_score,
        std_score,
        min_score,
        max_score,
        float(len(component_names)),
        float(n_skills),
        float(n_tools),
        float(n_memory),
        float(n_other),
        mean_co,
        min_co,
        max_co,
        float(cluster_id),
        float(cluster_id) / 100.0,
    ]
    assert len(fvec) == _FEATURE_DIM, f"Feature dim mismatch: {len(fvec)}"
    return fvec


def _all_pairwise_co_inclusion(
    names: list[str],
    extractor,
) -> list[float]:
    """Return flat list of pairwise co-inclusion scores for all (i<j) pairs."""
    mat = extractor.co_inclusion_matrix()
    name_to_idx: dict[str, int] = getattr(extractor, "_name_to_idx", {})
    if not name_to_idx:
        return []

    pairs: list[float] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia = name_to_idx.get(a)
            ib = name_to_idx.get(b)
            if ia is not None and ib is not None:
                pairs.append(float(mat[ia, ib]))
    return pairs
