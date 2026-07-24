"""Phase R5 — Warm-start new deployments from the knowledge base.

The :class:`TransferInitializer` matches components in a *new* deployment
against the cross-deployment knowledge base (KB) using SHA-256 fingerprints
and writes a ``transfer_priors.json`` file to the oracle directory.  The GA
can read this file to seed its initial fitness scores rather than using flat
priors, reducing cold-start time.

**Transfer protocol**

1. Fingerprint all components in the new deployment's ``context_configs.json``.
2. For each component, look up its fingerprint in the KB.
3. If found: extract ``mean_outcome_when_included`` as the prior mean score.
4. Write ``{component_name: prior_score}`` to ``transfer_priors.json``.
5. Components without a KB match fall back to the current scoring matrix.

The GA's fitness function then uses::

    fitness_i = α × kb_prior_i + (1 - α) × current_score_i

where α decays as turns accumulate (default: α = exp(-n_turns / τ), τ=200).

Usage::

    from thalamus_research.meta_learning.transfer_initializer import TransferInitializer

    ti = TransferInitializer(kb_path="/shared/knowledge_base.json")
    result = ti.transfer(new_oracle_dir="/oracle/new_deployment")
    print(f"Warm-started {result.n_matched} / {result.n_total} components")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_PRIORS_FILE = "transfer_priors.json"


@dataclass
class TransferResult:
    """Summary of a warm-start transfer operation."""

    n_total: int          # total components in the new deployment
    n_matched: int        # components matched in the KB
    n_unmatched: int      # components with no KB entry
    match_rate: float     # n_matched / n_total
    prior_scores: dict[str, float]   # component_name → prior quality score

    def to_dict(self) -> dict:
        return asdict(self)

    def print_report(self) -> None:
        print(f"\nPhase R5 — Transfer Initializer")
        print(f"Components: {self.n_total}  |  "
              f"KB matches: {self.n_matched} ({self.match_rate * 100:.1f}%)")
        if self.prior_scores:
            print()
            print(f"{'Component':<35} {'Prior Score':>12}")
            print("-" * 49)
            for name, score in sorted(self.prior_scores.items(), key=lambda x: -x[1])[:20]:
                print(f"{name:<35} {score:>12.4f}")


class TransferInitializer:
    """Warm-start a new oracle deployment from the cross-deployment KB.

    Parameters
    ----------
    kb_path:
        Path to the ``knowledge_base.json`` file produced by
        :class:`~thalamus_research.meta_learning.knowledge_base.KnowledgeBase`.
    """

    def __init__(self, kb_path: str | Path) -> None:
        from .knowledge_base import KnowledgeBase

        self._kb = KnowledgeBase(kb_path)
        logger.info("TransferInitializer: KB has %d entries", len(self._kb))

    # ── public API ────────────────────────────────────────────────────────────

    def transfer(
        self,
        new_oracle_dir: str | Path,
        write_priors: bool = True,
    ) -> TransferResult:
        """Match components in *new_oracle_dir* against the KB and produce priors.

        Parameters
        ----------
        new_oracle_dir:
            Directory of the new deployment (must have ``context_configs.json``).
        write_priors:
            If True (default), write ``transfer_priors.json`` to *new_oracle_dir*.

        Returns
        -------
        :class:`TransferResult`
        """
        from .component_fingerprint import fingerprint_catalog

        oracle_dir = Path(new_oracle_dir)
        fps = fingerprint_catalog(oracle_dir)   # {name: fingerprint}

        prior_scores: dict[str, float] = {}
        n_matched = 0

        for name, fp in fps.items():
            entry = self._kb.get(fp)
            if entry is None:
                continue
            score = entry.get("mean_outcome_when_included")
            if score is not None:
                prior_scores[name] = float(score)
                n_matched += 1

        n_total = len(fps)
        match_rate = n_matched / max(n_total, 1)

        result = TransferResult(
            n_total=n_total,
            n_matched=n_matched,
            n_unmatched=n_total - n_matched,
            match_rate=round(match_rate, 4),
            prior_scores=prior_scores,
        )

        if write_priors:
            priors_path = oracle_dir / _PRIORS_FILE
            priors_path.write_text(
                json.dumps(prior_scores, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "transfer_priors.json written to %s (%d entries)",
                oracle_dir, len(prior_scores),
            )

        logger.info(
            "Transfer: %d / %d components matched (%.1f%%)",
            n_matched, n_total, match_rate * 100,
        )
        return result
