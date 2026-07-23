"""THALAMUS research package.

Structure
---------
::

    research/
    ├── baselines/       R1 — baseline selectors (AllSelector, TFIDFSelector, BM25Selector, DenseSelector)
    ├── evaluation/      R1 — benchmark harness, result schema, overlap stats, report
    ├── cross_path/      R3a — cross-path knowledge transfer (classifier weights → GA fitness)
    ├── bandit/          R3b — contextual bandit formalization, exploration rate analysis
    ├── set_quality/     R4 — set-level quality model (XGB / joint classifier as GA fitness)
    └── meta_learning/   R5 — cross-deployment warm-start from shared knowledge base

CLI
---
``thalamus-research`` — research commands only (baseline-lookup, eval, and future R3-R5 commands).
``thalamus-select``   — production commands only (lookup, classify).
"""
