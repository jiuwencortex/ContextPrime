# oracle_builder/evolutionary/pareto_validator.py
"""End-to-end Pareto-front validation using real LLM calls.

The proxy fitness function used during evolutionary search
(Σ mean_score × cosine_sim − λ × size) measures component-level
relevance but ignores combination synergies.  This module re-evaluates
each config on the Pareto front by sending the assembled component list
to an LLM and asking it to score how well the combination matches a
representative set of cluster queries.

The corrected score is:
    combined_score = proxy_fitness × (1 − α) + llm_score_normalized × α

where α = 0.5 by default, balancing the two signals.

Invoked only when ``--validate-pareto`` is passed to ``oracle_builder evolve``.
Requires an OpenAI-compatible API key.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from .evolution.context_genome import ContextGenome
from .component_info import ComponentInfo

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\b([1-9]|10)\b")

# Weight for the LLM quality signal vs. proxy fitness (0 = proxy only, 1 = LLM only)
_LLM_WEIGHT = 0.5


@dataclass
class ValidationConfig:
    """Settings for the LLM-based Pareto validator."""
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str = "https://api.openai.com/v1"
    queries_per_cluster: int = 3


class ParetoValidator:
    """Re-rank Pareto-front configs using LLM quality scores.

    Usage::

        cfg = ValidationConfig(model="gpt-4o-mini", api_key="sk-...")
        validator = ParetoValidator(cfg)
        re_ranked = validator.validate(pareto, components, cluster_queries)
        # re_ranked is a list of (genome, combined_score) sorted best-first
    """

    def __init__(self, config: ValidationConfig):
        self._cfg = config
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "ParetoValidator requires an API key.  Pass --eval-api-key or "
                "set the OPENAI_API_KEY environment variable."
            )

    def validate(
        self,
        pareto: list[ContextGenome],
        components: list[ComponentInfo],
        cluster_queries: list[str],
    ) -> list[tuple[ContextGenome, float]]:
        """Evaluate each Pareto config and return (genome, combined_score) pairs.

        Parameters
        ----------
        pareto:
            Pareto-front genomes from the evolutionary search.
        components:
            Full component list (needed to decode genome bits).
        cluster_queries:
            Representative query texts for this cluster.

        Returns
        -------
        list of (ContextGenome, combined_score)
            Sorted by combined_score descending (best first).
        """
        if not pareto:
            return []

        # Sample at most N representative queries
        queries = cluster_queries[: self._cfg.queries_per_cluster]
        if not queries:
            logger.warning("No representative queries for Pareto validation; skipping")
            return [(g, g.fitness) for g in pareto]

        results: list[tuple[ContextGenome, float]] = []

        for genome in pareto:
            config_dict = genome.to_config(components)
            llm_scores: list[float] = []

            for query in queries:
                score = self._evaluate_one(query, config_dict)
                if score is not None:
                    llm_scores.append(score)

            if llm_scores:
                avg_llm = sum(llm_scores) / len(llm_scores)
                # Normalize LLM score from [1, 10] to [0, 1]
                llm_normalized = (avg_llm - 1.0) / 9.0
                combined = genome.fitness * (1 - _LLM_WEIGHT) + llm_normalized * _LLM_WEIGHT
            else:
                combined = genome.fitness

            results.append((genome, combined))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _evaluate_one(self, query: str, config_dict: dict) -> float | None:
        """Ask the LLM to score one (query, config) pair.  Returns 1–10 or None on error."""
        skills  = config_dict.get("skills", [])
        memory  = config_dict.get("memory", [])
        tools   = config_dict.get("tools", [])

        prompt = (
            "You are evaluating whether a context configuration is appropriate for an AI agent query.\n\n"
            f'Query: "{query}"\n\n'
            "Selected context components:\n"
            f"  Skills:          {skills or '(none)'}\n"
            f"  Memory sections: {memory or '(none)'}\n"
            f"  Tools:           {tools or '(none)'}\n\n"
            "Rate how well this combination matches the query's needs.\n"
            "Consider relevance, completeness, and conciseness.\n"
            "Reply with only a single integer from 1 (very poor) to 10 (excellent)."
        )

        try:
            response_text = self._chat(prompt)
            match = _SCORE_RE.search(response_text.strip())
            if match:
                return float(match.group(1))
            logger.warning("Could not parse LLM score from: %r", response_text)
            return None
        except Exception as exc:
            logger.warning("LLM evaluation failed: %s", exc)
            return None

    def _chat(self, user_message: str) -> str:
        """Make a single chat-completion call and return the assistant's text."""
        url = f"{self._cfg.api_base.rstrip('/')}/chat/completions"
        payload = json.dumps({
            "model":      self._cfg.model,
            "messages":   [{"role": "user", "content": user_message}],
            "max_tokens": 10,
            "temperature": 0.0,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        return body["choices"][0]["message"]["content"]
