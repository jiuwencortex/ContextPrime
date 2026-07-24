"""Phase R4 — Model-based GA fitness function.

Replaces the GA's linear ``sum(mean_score_i)`` fitness with a
:class:`SetQualityModel` prediction — a gradient-boosting regressor that
captures pairwise component interactions invisible to the linear baseline.

The fitness function is intentionally lightweight: it takes a list of
component names and a cluster ID and returns a scalar in [0, 1].

Integration with the existing GA
---------------------------------
The Thalamus GA (``thalamus.oracle.ga``) calls a fitness function at each
generation via a pluggable ``fitness_fn`` parameter.  :class:`SetQualityFitness`
implements that interface::

    fitness_fn(component_names: list[str], cluster_id: int) -> float

Usage::

    from thalamus.research.set_quality.fitness_function import SetQualityFitness

    fitness = SetQualityFitness.load(
        model_dir="/oracle/set_quality_model",
        catalog=catalog,
        extractor=co_inclusion_extractor,  # optional
    )

    # Use in GA loop
    score = fitness(["skill_a", "tool_b"], cluster_id=3)
    # or equivalently:
    score = fitness.score(["skill_a", "tool_b"], cluster_id=3)
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SetQualityFitness:
    """GA-compatible fitness function backed by :class:`SetQualityModel`.

    Parameters
    ----------
    model:
        Fitted :class:`~thalamus.research.set_quality.set_quality_model.SetQualityModel`.
    catalog:
        :class:`~thalamus.research.baselines.component_catalog.ComponentCatalog`.
    extractor:
        Optional :class:`~thalamus.research.cross_path.co_inclusion_extractor.CoInclusionExtractor`.
    fallback_weight:
        If the model raises an exception during prediction, the method falls
        back to a simple mean-score sum normalized to [0, 1].  This weight
        (default 1.0) is applied to the fallback score so it can be tuned.
    """

    def __init__(
        self,
        model,                 # SetQualityModel
        catalog,               # ComponentCatalog
        extractor=None,        # CoInclusionExtractor | None
        fallback_weight: float = 1.0,
    ) -> None:
        self._model = model
        self._catalog = catalog
        self._extractor = extractor
        self._fallback_weight = fallback_weight

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        catalog,
        extractor=None,
        fallback_weight: float = 1.0,
    ) -> "SetQualityFitness":
        """Load a saved :class:`SetQualityModel` and return a fitness instance.

        Parameters
        ----------
        model_dir:
            Directory containing ``model.pkl`` and ``meta.json``.
        catalog:
            Loaded :class:`~thalamus.research.baselines.component_catalog.ComponentCatalog`.
        extractor:
            Optional co-inclusion extractor.
        fallback_weight:
            Weight applied to the fallback (mean-score) prediction.

        Raises
        ------
        FileNotFoundError
            If ``model.pkl`` is absent in *model_dir*.
        """
        from .set_quality_model import SetQualityModel

        model = SetQualityModel.load(model_dir)
        logger.info("SetQualityFitness loaded from %s  meta=%s", model_dir, model.meta)
        return cls(model, catalog, extractor, fallback_weight)

    # ── callable interface ────────────────────────────────────────────────────

    def __call__(self, component_names: list[str], cluster_id: int) -> float:
        """Return fitness score in [0, 1] for *component_names*."""
        return self.score(component_names, cluster_id)

    def score(self, component_names: list[str], cluster_id: int) -> float:
        """Return predicted outcome quality in [0, 1].

        Falls back to mean-score heuristic if the model prediction fails.
        """
        try:
            return self._model.score_set(
                component_names, cluster_id, self._catalog, self._extractor
            )
        except Exception:
            logger.debug("SetQualityModel.score_set failed; using fallback", exc_info=True)
            return self._fallback_score(component_names)

    # ── internals ─────────────────────────────────────────────────────────────

    def _fallback_score(self, component_names: list[str]) -> float:
        if not component_names:
            return 0.0
        scores = []
        for name in component_names:
            try:
                scores.append(self._catalog.score(name))
            except Exception:
                scores.append(0.0)
        mean = sum(scores) / len(scores)
        return float(min(max(mean * self._fallback_weight, 0.0), 1.0))
