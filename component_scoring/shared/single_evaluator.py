# recommendation_matrix/shared/single_evaluator.py
# SingleEvaluator: evaluates one component against all its (query, answer) pairs.
from __future__ import annotations

import asyncio
import logging

from openjiuwen.core.foundation.llm import Model, SystemMessage, UserMessage

from .metrics.metric_bag_of_words import bag_of_words
from .metrics.metric_bigram_f1 import bigram_f1
from .metrics.metric_length_ratio import length_ratio
from .metrics.metric_token_f1 import token_f1

logger = logging.getLogger(__name__)


class SingleEvaluator:
    """Evaluate how helpful a component is for a single prompt.

    Given:
      - component_body: the component content (SKILL.md body, memory section, tool description)
      - query: a user prompt
      - expected: the expected/desired output

    Runs one LLM call with the component body as system message and the query as user message.
    """

    def __init__(
        self,
        model: Model,
        model_name: str,
        timeout: float = 3600.0,
        temperature: float = 0.2,
        max_tokens: int = 57000,
    ):
        self._model = model
        self._model_name = model_name
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def evaluate_component(
        self,
        component_body: str,
        pairs: list[dict],
        sem: asyncio.Semaphore,
    ) -> list[dict]:
        """Evaluate all (query, answer) pairs for one component in parallel."""

        async def eval_one(pair: dict) -> dict:
            async with sem:
                return await self.evaluate_pair(
                    component_body=component_body,
                    query=pair["query"],
                    expected=pair["answer"],
                )

        return list(await asyncio.gather(*[eval_one(p) for p in pairs]))

    # Backward-compatible alias
    async def evaluate_skill(
        self,
        skill_body: str,
        pairs: list[dict],
        sem: asyncio.Semaphore,
    ) -> list[dict]:
        return await self.evaluate_component(skill_body, pairs, sem)

    async def evaluate_pair(self, component_body: str, query: str, expected: str) -> dict:
        """Run one (component, query) pair through the LLM and score the result."""
        actual = await self._invoke(component_body, query)
        scores = self._compute_scores(actual, expected)
        return {
            "example_input": query,
            "example_expected": expected,
            "candidate_output": actual,
            "scores": scores,
        }

    async def _invoke(self, component_body: str, query: str) -> str:
        """Run one (component, query) pair through the LLM. Returns response text."""
        try:
            response = await asyncio.wait_for(
                self._model.invoke(
                    [SystemMessage(content=component_body), UserMessage(content=query)],
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                ),
                timeout=self._timeout,
            )
            return (response.content or "").strip()
        except asyncio.TimeoutError:
            logger.warning("Execution timed out: query=%r", query[:60])
            return ""
        except Exception as e:
            logger.warning("Execution error: %s", e)
            return ""

    @staticmethod
    def _compute_scores(candidate_output: str, expected: str) -> dict[str, float]:
        """Return {metric: score} for all FITNESS_METRICS."""
        return {
            "f1": token_f1(candidate_output, expected),
            "bigram_f1": bigram_f1(candidate_output, expected),
            "bag_of_words": bag_of_words(candidate_output, expected),
            "length_ratio": length_ratio(candidate_output, expected),
        }


# Backward-compatible alias
SingleSkillEvaluator = SingleEvaluator
