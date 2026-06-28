# recommendation_matrix/shared/queries_generator.py
# Stage 1: generate (query, answer) pairs for each component in parallel.
from __future__ import annotations

import asyncio
import json
import logging

from openjiuwen.core.foundation.llm import Model, SystemMessage, UserMessage

from .fingerprint import ComponentRecord

logger = logging.getLogger(__name__)


_SYSTEM = "You are a test-data generator for AI evaluation."


class QueriesGenerator:
    """Generate (query, answer) pairs for components using an LLM."""

    def __init__(
        self,
        model: Model,
        model_name: str,
        n_examples: int,
        max_parallel: int,
        prompt_template: str | None = None,
    ):
        self._model = model
        self._model_name = model_name
        self._n = n_examples
        self._max_parallel = max_parallel
        self._template = prompt_template

    async def generate_for_items(
        self, components: list[ComponentRecord]
    ) -> list[tuple[ComponentRecord, list[dict]]]:
        """Call the LLM once per component to invent N (query, answer) pairs in parallel."""
        sem = asyncio.Semaphore(self._max_parallel)

        async def generate_with_sem(component: ComponentRecord) -> tuple[ComponentRecord, list[dict]]:
            async with sem:
                pairs = await self._generate_single(component)
                logger.info("component=%s: %d pairs generated", component.name, len(pairs))
                return component, pairs

        return list(await asyncio.gather(*[generate_with_sem(c) for c in components]))

    # Backward-compatible alias used by skill_matrix compat shim
    async def generate_for_skills(
        self, skills: list[ComponentRecord]
    ) -> list[tuple[ComponentRecord, list[dict]]]:
        return await self.generate_for_items(skills)

    async def _generate_single(self, component: ComponentRecord) -> list[dict]:
        """Call LLM once to invent N (query, answer) pairs for one component."""
        prompt = self._template.format(
            n=self._n,
            name=component.name,
            description=component.description,
            body=component.body,
            # extra fields for memory template (ignored if not in template)
            source_file=getattr(component, "source_file", ""),
        )
        response = await self._model.invoke(
            [SystemMessage(content=_SYSTEM), UserMessage(content=prompt)],
            model=self._model_name,
            temperature=0.8,
            max_tokens=4096,
        )
        return self._parse(response.content or "", component.name)

    def _parse(self, text: str, component_name: str) -> list[dict]:
        try:
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1]
                if stripped.endswith("```"):
                    stripped = stripped.rsplit("```", 1)[0]
            pairs = json.loads(stripped.strip())
            if not isinstance(pairs, list):
                raise ValueError("Expected a JSON array")
            valid = [
                p for p in pairs
                if isinstance(p, dict)
                and isinstance(p.get("query"), str) and len(p["query"]) > 10
                and isinstance(p.get("answer"), str) and len(p["answer"]) > 5
            ]
            if len(valid) < self._n // 2:
                logger.warning(
                    "component=%s: only %d/%d valid pairs generated",
                    component_name, len(valid), self._n,
                )
            return valid[:self._n]
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("component=%s: failed to parse LLM output: %s", component_name, e)
            logger.debug("component=%s: raw LLM output was: %r", component_name, text[:500])
            return []
