"""CLI command: thalamus-select baseline-lookup

Run a single query through a baseline selector and print the result.

Usage::

    # TF-IDF top-k retrieval
    thalamus-select baseline-lookup \\
        --oracle-dir /path/to/oracle \\
        --query "Set up CI/CD pipeline" \\
        --method tfidf \\
        --budget medium \\
        --ordering bookend

    # Compare all methods side by side
    thalamus-select baseline-lookup \\
        --oracle-dir /path/to/oracle \\
        --query "Fix the login bug" \\
        --method all random tfidf bm25 \\
        --budget medium
"""
from __future__ import annotations

import argparse
import json
import time


def cmd_baseline_lookup(args: argparse.Namespace) -> None:
    oracle_dir = args.oracle_dir
    query = args.query or "What can you help me with?"
    budget = args.budget if args.budget != "auto" else None
    ordering = args.ordering
    methods: list[str] = args.method
    seed: int | None = args.seed

    results: dict[str, dict] = {}

    for method in methods:
        selector = _load_selector(method, oracle_dir, seed)
        if not selector.is_ready:
            print(f"[{method}] not ready — no scoring matrices in {oracle_dir}")
            continue

        t0 = time.perf_counter()
        result = selector.select(query, budget=budget, ordering=ordering)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result is None:
            print(f"[{method}] returned None")
            continue

        result["_latency_ms"] = round(elapsed_ms, 2)
        results[method] = result

    if len(methods) == 1:
        # Single method: pretty-print
        method = methods[0]
        if method in results:
            _print_result(method, query, results[method])
    else:
        # Multiple methods: comparison table
        _print_comparison(query, results)


def _load_selector(method: str, oracle_dir, seed):
    # Import lazily so the CLI works even if sentence-transformers is absent
    if method == "all":
        from .all_selector import AllSelector
        return AllSelector.load(oracle_dir)
    elif method == "random":
        from .random_selector import RandomSelector
        return RandomSelector.load(oracle_dir, seed=seed)
    elif method == "tfidf":
        from .tfidf_selector import TFIDFSelector
        return TFIDFSelector.load(oracle_dir)
    elif method == "bm25":
        from .bm25_selector import BM25Selector
        return BM25Selector.load(oracle_dir)
    elif method == "dense":
        from .dense_selector import DenseSelector
        return DenseSelector.load(oracle_dir)
    else:
        raise ValueError(f"Unknown baseline method: {method!r}")


def _print_result(method: str, query: str, result: dict) -> None:
    latency = result.pop("_latency_ms", None)
    print(f"\nBaseline: {method}")
    print(f"Query:    {query!r}")
    if latency is not None:
        print(f"Latency:  {latency:.2f} ms")
    print()
    print(json.dumps(result, indent=2))


def _print_comparison(query: str, results: dict[str, dict]) -> None:
    print(f"\nQuery: {query!r}\n")
    header = f"{'method':<12} {'skills':>6} {'memory':>6} {'tools':>6} {'total':>6} {'ms':>7}"
    print(header)
    print("-" * len(header))
    for method, result in results.items():
        n_skills = len(result.get("skills", []))
        n_memory = len(result.get("memory", []))
        n_tools = len(result.get("tools", []))
        latency = result.get("_latency_ms", 0.0)
        print(
            f"{method:<12} {n_skills:>6} {n_memory:>6} {n_tools:>6} "
            f"{n_skills+n_memory+n_tools:>6} {latency:>7.2f}"
        )
    print()
    for method, result in results.items():
        r = {k: v for k, v in result.items() if not k.startswith("_")}
        print(f"--- {method} ---")
        print(json.dumps(r, indent=2))
        print()
