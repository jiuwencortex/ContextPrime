# context_selectors/cli_args_parser.py
from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m thalamus.context_selectors",
        description=(
            "Runtime context selection tools.\n\n"
            "  lookup          — cluster-based lookup using context_configs.json\n"
            "  classify        — classifier-based selection using classifier.pkl\n"
            "  baseline-lookup — retrieval baselines for research evaluation (R1)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # Subcommand: cluster-based lookup
    lookup = sub.add_parser(
        "lookup",
        help="Look up the precomputed context config for a query (ClusterSelector)",
    )
    lookup.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing context_configs.json",
    )
    lookup.add_argument(
        "--query", default=None,
        help="Query text: show which cluster and config would be selected",
    )
    lookup.add_argument(
        "--budget", default="medium", choices=["small", "medium", "large", "auto"],
        help="Budget level to look up, or 'auto' to estimate from query text (default: medium)",
    )
    lookup.add_argument(
        "--ordering", default="relevance", choices=["relevance", "bookend", "none"],
        help=(
            "Component ordering strategy: 'relevance' (most-relevant first, default), "
            "'bookend' (most-relevant at edges to combat lost-in-the-middle), "
            "'none' (original insertion order)"
        ),
    )

    # Subcommand: classifier-based selection
    classify = sub.add_parser(
        "classify",
        help="Predict component inclusion from an embedding (ClassifierSelector)",
    )
    classify.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing classifier.pkl",
    )
    classify.add_argument(
        "--embedding", default=None, type=Path,
        help="Path to a .npy file containing the query embedding vector",
    )
    classify.add_argument(
        "--threshold", type=float, default=0.5,
        help="Inclusion threshold for classifier output (default: 0.5)",
    )
    classify.add_argument(
        "--verbose", action="store_true",
        help="Show per-component probability scores",
    )

    # Subcommand: baseline retrieval lookup (Phase R1)
    baseline = sub.add_parser(
        "baseline-lookup",
        help="Run a query through a retrieval baseline (research evaluation Phase R1)",
    )
    baseline.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing scoring matrices (and optionally context_configs.json)",
    )
    baseline.add_argument(
        "--query", default=None,
        help="Query text to evaluate",
    )
    baseline.add_argument(
        "--method", nargs="+",
        choices=["all", "random", "tfidf", "bm25", "dense"],
        default=["tfidf"],
        help=(
            "Baseline method(s) to run.  Multiple values produce a comparison table.\n"
            "  all    — return every component (quality upper bound)\n"
            "  random — random k components (null hypothesis)\n"
            "  tfidf  — TF-IDF cosine similarity top-k\n"
            "  bm25   — BM25 top-k (no extra deps)\n"
            "  dense  — sentence-transformer cosine top-k (requires thalamus[sentence])"
        ),
    )
    baseline.add_argument(
        "--budget", default="medium", choices=["small", "medium", "large", "auto"],
        help="Budget tier (default: medium).  'auto' passes None to let the selector decide.",
    )
    baseline.add_argument(
        "--ordering", default="bookend", choices=["relevance", "bookend", "none"],
        help="Component ordering strategy (default: bookend)",
    )
    baseline.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the 'random' baseline (default: None = system entropy)",
    )

    return p
