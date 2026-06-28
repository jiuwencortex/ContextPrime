# recommendation_matrix/shared/all_items_evaluator.py
# AllItemsEvaluator: evaluate every component against its queries, write matrix files, build states.
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from openjiuwen.core.foundation.llm import Model

from .fingerprint import ComponentRecord, component_fingerprint
from .metrics.metrics_list import FITNESS_METRICS
from .single_evaluator import SingleEvaluator
from .state import ComponentState, now_iso

logger = logging.getLogger(__name__)


class AllItemsEvaluator:
    """Evaluate all components against their generated queries and write scoring matrix files."""

    def __init__(
        self,
        model: Model,
        model_name: str,
        matrix_dir: Path,
        max_parallel: int,
        cross_eval: bool = False,
        timeout: float = 3600.0,
        temperature: float = 0.2,
        max_tokens: int = 57000,
        file_prefix: str = "skill_",
        component_type: str = "skill",
    ):
        self._matrix_dir = matrix_dir
        self._max_parallel = max_parallel
        self._cross_eval = cross_eval
        self._file_prefix = file_prefix
        self._component_type = component_type
        self._single = SingleEvaluator(
            model, model_name, timeout=timeout, temperature=temperature, max_tokens=max_tokens
        )

    async def evaluate_all(
        self,
        gen_results: list[tuple[ComponentRecord, list[dict]]],
    ) -> tuple[dict[str, ComponentState], int]:
        """Run every component against its queries, write scoring_matrix_*.json files.

        Returns {component_name: ComponentState} and total LLM call count.
        """
        all_pairs = self._collect_all_pairs(gen_results)
        sem = asyncio.Semaphore(self._max_parallel)

        new_states: dict[str, ComponentState] = {}
        llm_calls = len(gen_results)

        for component, pairs in gen_results:
            eval_pairs = all_pairs if self._cross_eval else pairs
            rows = await self._single.evaluate_component(component.body, eval_pairs, sem)
            self._write_matrix_file(component.name, rows)
            llm_calls += len(eval_pairs)

            mean_score = (
                sum(r["scores"]["f1"] for r in rows) / len(rows) if rows else 0.0
            )
            new_states[component.name] = ComponentState(
                fingerprint=component_fingerprint(component),
                n_examples=len(rows),
                built_at=now_iso(),
                mean_score=round(mean_score, 4),
            )
            logger.info(
                "component=%s: wrote %d rows, mean_f1=%.3f",
                component.name, len(rows), mean_score,
            )

        return new_states, llm_calls

    # Backward-compatible alias used by skill_matrix compat shim
    async def evaluate_all_skills(
        self,
        gen_results: list[tuple[ComponentRecord, list[dict]]],
    ) -> tuple[dict[str, ComponentState], int]:
        return await self.evaluate_all(gen_results)

    def _collect_all_pairs(
        self, gen_results: list[tuple[ComponentRecord, list[dict]]]
    ) -> list[dict]:
        all_pairs: list[dict] = []
        for _, pairs in gen_results:
            all_pairs.extend(pairs)
        return all_pairs

    def _write_matrix_file(self, component_name: str, rows: list[dict]) -> Path:
        self._matrix_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w-]", "_", component_name)
        dest = self._matrix_dir / f"scoring_matrix_{self._file_prefix}{safe}.json"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        payload = {
            "run_id": f"{self._file_prefix}{safe}_matrix_{ts}",
            "component_type": self._component_type,
            "component_name": component_name,
            "fitness_metrics": FITNESS_METRICS,
            "baseline_cross_eval": [
                {
                    "example_id": f"{self._file_prefix}{safe}_{i:04d}",
                    "example_input": r["example_input"],
                    "example_expected": r["example_expected"],
                    "candidate_output": r["candidate_output"],
                    "scores": r["scores"],
                }
                for i, r in enumerate(rows)
            ],
            "evolved_cross_eval": [],
        }
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest


# Backward-compatible alias
AllSkillsEvaluator = AllItemsEvaluator
