"""Phase R5 — Cross-Deployment Meta-Learning.

Research goal: reduce cold-start time for new jiuwenswarm deployments by
transferring component quality priors from existing deployments that share
overlapping components (identified by SHA-256 content fingerprints).

**Key claim (C8)**

    Warm-starting a new deployment's GA fitness scores from a cross-deployment
    knowledge base (KB) reaches the same outcome quality as a cold-start
    deployment in ≤ 50% of the turns, provided ≥ 3 source deployments and
    ≥ 30% component overlap.

**Modules**

- :func:`fingerprint_component`  — SHA-256(name + description + body_text)
- :func:`fingerprint_catalog`    — fingerprint all components in an oracle dir
- :class:`KnowledgeBase`         — persistent JSON store: fingerprint → quality stats
- :class:`TransferInitializer`   — match new deployment to KB; write transfer_priors.json

CLI::

    # Extract stats from a completed deployment into the shared KB
    thalamus-research meta-learning \\
        --oracle-dir /oracle/deployment_1 \\
        --kb-path /shared/knowledge_base.json \\
        --subcommand extract

    # Warm-start a new deployment from the KB
    thalamus-research meta-learning \\
        --oracle-dir /oracle/new_deployment \\
        --kb-path /shared/knowledge_base.json \\
        --subcommand transfer

Prerequisite: Phase R4 complete AND at least 3 distinct jiuwenswarm deployments.
"""
from .component_fingerprint import fingerprint_component, fingerprint_catalog
from .knowledge_base import KnowledgeBase
from .transfer_initializer import TransferInitializer, TransferResult

__all__ = [
    "fingerprint_component",
    "fingerprint_catalog",
    "KnowledgeBase",
    "TransferInitializer",
    "TransferResult",
]
