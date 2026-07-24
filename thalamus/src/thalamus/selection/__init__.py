"""Runtime context selection for agent queries.

Recommended entry point
-----------------------
Use :class:`ContextSelector` for all production callers.  It handles the
Path B → Path A → None fallback automatically::

    from thalamus.selection import ContextSelector

    selector = ContextSelector.load(oracle_dir)
    result = selector.select(user_query)   # dict | None
    # result = {"skills": [...], "memory": [...], "tools": [...], ...}

Low-level backends (use directly only when you need fine-grained control)
--------------------------------------------------------------------------
- :class:`ClusterSelector`   Phase 3 — cluster-based lookup (text in, instant)
- :class:`ClassifierSelector` Phase 4 — trained linear classifier (embedding in)

Utilities
---------
- :class:`BudgetEstimator`   Heuristic budget estimator (auto small/medium/large)
"""

from .context_selector import ContextSelector
from .by_clusters.cluster_selector import ClusterSelector
from .by_classifier.classifier_selector import ClassifierSelector
from .by_clusters.budget_estimator import BudgetEstimator

__all__ = ["ContextSelector", "ClusterSelector", "ClassifierSelector", "BudgetEstimator"]
