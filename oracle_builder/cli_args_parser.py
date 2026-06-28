# oracle_builder/cli_args_parser.py
# Entry point: python -m jiuwenswarm.tools.oracle_builder build
from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m jiuwenswarm.tools.oracle_builder",
        description=(
            "Build context_configs.json from pre-computed recommendation matrices.\n\n"
            "Reads all scoring_matrix_skill_*.json, scoring_matrix_mem_*.json, and\n"
            "scoring_matrix_tool_*.json files from --oracle-dir.\n\n"
            "Clusters their example queries, runs an evolutionary search over component\n"
            "combinations (no LLM calls), and writes optimal configs per cluster and budget\n"
            "to context_configs.json. A companion .pkl file stores the TF-IDF model for\n"
            "query-time cluster assignment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    build = sub.add_parser("evolve", help="Run evolutionary search to build context_configs.json")
    build.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing scoring_matrix_*.json files (output of component_scoring build)",
    )
    build.add_argument(
        "--output", type=Path,
        help="Path to write context_configs.json (default: <oracle-dir>/context_configs.json)",
    )
    build.add_argument(
        "--n-clusters", type=int, default=20,
        help="Number of query-type clusters (default: 20)",
    )
    build.add_argument(
        "--max-features", type=int, default=2000,
        help="TF-IDF vocabulary size (default: 2000)",
    )
    build.add_argument(
        "--population", type=int, default=100,
        help="Evolutionary search population size (default: 100)",
    )
    build.add_argument(
        "--generations", type=int, default=200,
        help="Number of evolutionary generations (default: 200)",
    )
    build.add_argument(
        "--mutation-rate", type=float, default=0.05,
        help="Mutation probability per bit (default: 0.05)",
    )
    build.add_argument(
        "--lambda", dest="lambda_", type=float, default=0.1,
        help="Context-size penalty weight (default: 0.1)",
    )
    build.add_argument(
        "--budget-small", type=int, default=2000,
        help="Max tokens for 'small' budget (default: 2000)",
    )
    build.add_argument(
        "--budget-medium", type=int, default=4000,
        help="Max tokens for 'medium' budget (default: 4000)",
    )
    build.add_argument(
        "--budget-large", type=int, default=8000,
        help="Max tokens for 'large' budget (default: 8000)",
    )
    build.add_argument(
        "--embedder", default="tfidf", choices=["tfidf", "sentence"],
        help=(
            "Embedding backend for query clustering: "
            "'tfidf' (default, fast, no extra deps) or "
            "'sentence' (semantic, requires sentence-transformers)"
        ),
    )
    build.add_argument(
        "--sentence-model", dest="sentence_model", default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (only used when --embedder=sentence, default: all-MiniLM-L6-v2)",
    )
    build.add_argument(
        "--validate-pareto", action="store_true",
        help=(
            "After GA, evaluate each Pareto-front config with LLM calls against "
            "representative cluster queries and re-rank before selecting the best config. "
            "Requires --eval-model and --eval-api-key."
        ),
    )
    build.add_argument(
        "--eval-model", default="gpt-4o-mini",
        help="LLM model to use for Pareto validation (default: gpt-4o-mini)",
    )
    build.add_argument(
        "--eval-api-key", dest="eval_api_key", default=None,
        help="OpenAI-compatible API key for Pareto validation (falls back to OPENAI_API_KEY env var)",
    )
    build.add_argument(
        "--eval-api-base", dest="eval_api_base", default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL (default: https://api.openai.com/v1)",
    )
    build.add_argument(
        "--eval-queries-per-cluster", type=int, default=3, dest="eval_queries_per_cluster",
        help="Number of representative queries per cluster used in Pareto validation (default: 3)",
    )
    build.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # Subcommand: train the component inclusion classifier
    train = sub.add_parser(
        "train-classifier",
        help="Train a component inclusion classifier from logged agent turns",
    )
    train.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory to write classifier.pkl",
    )
    train.add_argument(
        "--log-dir", type=Path,
        help="Directory containing JSONL turn logs (default: <oracle-dir>/online_logs)",
    )
    train.add_argument(
        "--min-turns", type=int, default=10,
        help="Minimum logged turns required to train (default: 10)",
    )
    train.add_argument(
        "--max-weeks", type=int, default=8,
        help="Number of weekly log files to scan (default: 8)",
    )
    train.add_argument(
        "--C", dest="C", type=float, default=1.0,
        help="Inverse L2 regularisation strength for LogisticRegression (default: 1.0)",
    )

    return p
