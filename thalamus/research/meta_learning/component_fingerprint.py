"""Phase R5 — Component fingerprinting via SHA-256.

A *component fingerprint* is a stable, deployment-independent identifier for
a component derived from its semantic content rather than its runtime ID.
Two components in different deployments that perform the same function and
share the same description will receive the same fingerprint, enabling
knowledge transfer without requiring a shared component registry.

**Fingerprint formula**

    fingerprint = SHA-256(name + "\\x00" + description + "\\x00" + body_text)

where ``body_text`` is the full tool/skill/memory body (e.g. prompt text,
function source, or empty string if unavailable).  The ``\\x00`` separator
prevents boundary ambiguity (``"a" + "bc"`` ≠ ``"ab" + "c"``).

Usage::

    from thalamus.research.meta_learning.component_fingerprint import (
        fingerprint_component,
        fingerprint_catalog,
    )

    fp = fingerprint_component(name="web_search", description="Searches the web")
    catalog_fps = fingerprint_catalog(oracle_dir="/oracle")
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SEP = "\x00"


def fingerprint_component(
    name: str,
    description: str = "",
    body_text: str = "",
) -> str:
    """Return a 64-hex-char SHA-256 fingerprint for a component.

    Parameters
    ----------
    name:
        Component identifier (e.g. ``"web_search"``).
    description:
        Short description string (from catalog or tool card).
    body_text:
        Full body text (prompt, function source, schema).  Empty string if
        unavailable — the fingerprint degrades to name + description only.

    Returns
    -------
    str — 64-character lowercase hex digest.
    """
    payload = name + _SEP + description + _SEP + body_text
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_catalog(oracle_dir: str | Path) -> dict[str, str]:
    """Return a mapping ``{component_name: fingerprint}`` for all components
    in *oracle_dir*.

    Reads component metadata from ``context_configs.json``; uses the
    component name as the sole stable field when description/body are absent.

    Parameters
    ----------
    oracle_dir:
        Directory containing ``context_configs.json``.

    Returns
    -------
    dict mapping component name → fingerprint string.
    """
    configs_path = Path(oracle_dir) / "context_configs.json"
    if not configs_path.exists():
        raise FileNotFoundError(
            f"context_configs.json not found in {oracle_dir}. "
            "Run: thalamus-oracle evolve"
        )

    with configs_path.open(encoding="utf-8") as fh:
        configs = json.load(fh)

    seen: set[str] = set()
    fps: dict[str, str] = {}

    # Collect all component names across clusters and budgets
    for cluster_config in configs.values() if isinstance(configs, dict) else configs:
        if isinstance(cluster_config, dict):
            for budget_cfg in cluster_config.values():
                if not isinstance(budget_cfg, dict):
                    continue
                for key in ("skills", "tools", "memory"):
                    for comp_name in budget_cfg.get(key, []):
                        if comp_name not in seen:
                            seen.add(comp_name)
                            fps[comp_name] = fingerprint_component(comp_name)

    logger.info("Fingerprinted %d components from %s", len(fps), oracle_dir)
    return fps
