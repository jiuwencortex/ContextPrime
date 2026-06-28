from __future__ import annotations

import sys
import argparse

from .config_builder import ContextConfigBuilder


def cmd_build(args: argparse.Namespace) -> None:
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    output = args.output or (args.oracle_dir / "context_configs.json")
    budgets = {"small":  args.budget_small,
               "medium": args.budget_medium,
               "large":  args.budget_large}

    validation_config = None
    if getattr(args, "validate_pareto", False):
        from .pareto_validator import ValidationConfig
        validation_config = ValidationConfig(
            model=getattr(args, "eval_model", "gpt-4o-mini"),
            api_key=getattr(args, "eval_api_key", None),
            api_base=getattr(args, "eval_api_base", "https://api.openai.com/v1"),
            queries_per_cluster=getattr(args, "eval_queries_per_cluster", 3),
        )
        print(
            f"Pareto validation enabled: model={validation_config.model}, "
            f"queries_per_cluster={validation_config.queries_per_cluster}"
        )

    builder = ContextConfigBuilder(
        oracle_dir=args.oracle_dir,
        n_clusters=args.n_clusters,
        max_features=args.max_features,
        population_size=args.population,
        n_generations=args.generations,
        mutation_rate=args.mutation_rate,
        lambda_=args.lambda_,
        budgets=budgets,
        embedder=getattr(args, "embedder", "tfidf"),
        sentence_model=getattr(args, "sentence_model", "all-MiniLM-L6-v2"),
        validation_config=validation_config,
    )
    builder.build(output)
