"""Phase R4 — Set-Level Quality Modeling.

Research goal: replace the GA's linear sum-of-marginal-scores fitness with a
gradient-boosting model (GBR) that captures non-linear pairwise component
interactions, yielding configurations that transfer better across deployment
contexts.

**Key claim (C7)**

    A set-level quality model trained on (component_set, outcome_quality)
    pairs will produce GA configurations with higher mean outcome quality
    than the linear marginal-score baseline, provided that ≥ 20 labelled
    turns are available per cluster.

**Modules**

- :class:`OutcomeDataset`        — loads turn logs → OutcomeRecord list
- :func:`compute_feature_vector` — 14-dim numeric features per candidate set
- :class:`SetQualityModel`       — GradientBoostingRegressor wrapper (train/save/load)
- :class:`SetQualityFitness`     — GA-compatible fitness callable backed by the model

CLI::

    # Train set-quality model from turn logs
    thalamus-research set-quality --oracle-dir /oracle --subcommand train

    # Evaluate on held-out logs
    thalamus-research set-quality --oracle-dir /oracle --subcommand evaluate \\
        --turn-log-dir /logs/held_out --out eval_report.json

Prerequisite: Phase R3b complete (exploration rate ε* set; turn logs contain
sufficient off-policy exploration signal).
"""
from .outcome_dataset import OutcomeDataset, OutcomeRecord
from .interaction_features import compute_feature_vector
from .set_quality_model import SetQualityModel
from .fitness_function import SetQualityFitness

__all__ = [
    "OutcomeDataset",
    "OutcomeRecord",
    "compute_feature_vector",
    "SetQualityModel",
    "SetQualityFitness",
]
