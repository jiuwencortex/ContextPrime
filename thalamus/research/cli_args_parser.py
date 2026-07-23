"""Argument parser for the thalamus-research CLI.

All research subcommands live here, separate from the production
``thalamus-select`` CLI which handles only ``lookup`` and ``classify``.

Research subcommands
--------------------
  baseline-lookup   Run a single query through one or more baseline selectors
  eval              Benchmark all selectors over a query set, write results JSON
"""
from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m thalamus.research",
        description=(
            "THALAMUS research tools.\n\n"
            "  baseline-lookup — run a query through retrieval baselines (Phase R1)\n"
            "  eval            — benchmark selectors on a query set, write results JSON (Phase R1)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # ── baseline-lookup ───────────────────────────────────────────────────────
    baseline = sub.add_parser(
        "baseline-lookup",
        help="Run a query through a retrieval baseline (Phase R1)",
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

    # ── eval ──────────────────────────────────────────────────────────────────
    ev = sub.add_parser(
        "eval",
        help="Benchmark selectors on a query set and write a results JSON (Phase R1)",
    )
    ev.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing scoring matrices, context_configs.json, and classifier",
    )
    ev.add_argument(
        "--selectors", nargs="+",
        choices=["thalamus", "all", "random", "tfidf", "bm25", "dense"],
        default=["thalamus", "tfidf", "bm25"],
        metavar="SELECTOR",
        help=(
            "Selectors to benchmark.  One or more of: "
            "thalamus all random tfidf bm25 dense  (default: thalamus tfidf bm25)"
        ),
    )
    ev.add_argument(
        "--query", nargs="*", default=None,
        metavar="QUERY",
        help="Inline query strings.  May be combined with --queries-file.",
    )
    ev.add_argument(
        "--queries-file", default=None, type=Path,
        help=(
            "JSON file containing queries: list of {\"id\": str, \"query\": str} dicts, "
            "or a dict with a 'queries' key."
        ),
    )
    ev.add_argument(
        "--budget", default="medium", choices=["small", "medium", "large", "auto"],
        help="Budget tier passed to every selector (default: medium)",
    )
    ev.add_argument(
        "--ordering", default="bookend", choices=["relevance", "bookend", "none"],
        help="Component ordering strategy passed to every selector (default: bookend)",
    )
    ev.add_argument(
        "--n-repeats", type=int, default=5,
        help="Latency measurement repetitions per (query, selector) pair (default: 5)",
    )
    ev.add_argument(
        "--reference", default=None,
        help=(
            "Name of the reference selector for overlap statistics.  "
            "Defaults to 'thalamus' if present, otherwise the first listed selector."
        ),
    )
    ev.add_argument(
        "--output", default=None, type=Path,
        help="Write results to this JSON file (prints JSON to stdout when omitted)",
    )
    ev.add_argument(
        "--print-report", action="store_true",
        help="Print human-readable comparison table to stdout after writing JSON",
    )
    ev.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the 'random' selector baseline",
    )

    return p
