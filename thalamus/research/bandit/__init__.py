"""Phase R3b — Contextual bandit formalization.

Research goal: formalize context component selection as a multi-label contextual
bandit problem and derive the minimum off-policy exploration rate required for
the Path B classifier to converge to a policy not dominated by Path A.

Planned implementation
----------------------
- Formal model: state = query embedding, action = component bitmask, reward = outcome
- Prove: without exploration (ε=0), Path B converges to Path A's policy
- Derive ε* = f(|component_pool|, n_clusters, divergence(Path A action distribution,
              uniform distribution))
- Empirical: measure Path B quality as a function of ε on jiuwenswarm task suite
- CLI flag: ``thalamus-oracle tune --auto-exploration``

Prerequisite: Phase R1 complete (baselines + evaluation harness).
"""
