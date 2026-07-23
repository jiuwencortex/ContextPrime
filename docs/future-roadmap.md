# Future Roadmap — Beyond Step 8

> **Context:** This document describes what remains after Steps 1–8 of the
> implementation plan are complete. Steps 1–5 are done. Steps 6–8 are the
> next planned work. Everything here is post-Step-8 territory — not yet
> planned for implementation.

---

## Where Steps 1–8 Leave the System

After all eight implementation steps are complete, THALAMUS has:

- Components scored with real LLM-judge quality signals (Step 1)
- A versioned, evaluated classifier with held-out validation (Step 2)
- Auto-tuned hyperparameters for C, thresholds, K, and λ (Step 3)
- Semantic BERTScore / LLM-judge metrics replacing lexical ones (Step 4)
- Drift detection and automatic rebuild triggers (Step 5)
- Oracle warm-start for the classifier; classifier signal in GA fitness (Step 6)
- A learned gradient-boosting fitness function replacing the hand-crafted formula (Step 7)
- A joint multi-label classifier capturing component co-inclusion patterns (Step 8)

For a single production deployment with one agent and one component library,
this is the correct stopping point. The system is genuinely AutoML-like for
its specific problem.

---

## Two Remaining Structural Gaps

These are not addressable within the current architecture. They require
fundamentally different approaches.

### Gap A — Interaction Modeling at the Set Level

The Step-8 joint classifier models co-inclusion but does not model *why*
two components work better together. It learns correlations from data, not
causal structure.

A deeper model would:
- Take the full selected component set as input (not individual embeddings)
- Predict outcome quality for the set as a whole
- Learn combination effects directly: "Skill A + Memory B = good; either alone = mediocre"

This requires a set-level model — a transformer over component embeddings,
or a graph neural network over a component interaction graph. These are
significantly more complex than a linear or gradient-boosting model, require
substantially more data to train, and lose the interpretability of the current
logistic regression weights.

**When this becomes worthwhile:** when the component library exceeds ~100
components, when single-component scores become poor predictors of combination
outcomes, and when sufficient turn data exists to train and validate a model
of this complexity (likely 10,000+ turns).

### Gap B — Meta-Learning Across Deployments

THALAMUS starts from scratch for every new agent deployment. If a second
agent shares 60% of its skill library with an existing agent, none of the
learned configurations, classifier weights, or fitness model transfer.

A meta-learning layer would:
- Maintain a cross-deployment knowledge base indexed by component identity
  (fingerprint or embedding)
- When a new deployment starts, warm-start its oracle and classifier from
  configurations learned on other deployments with overlapping components
- Reduce cold-start time from weeks of data collection to days

**When this becomes worthwhile:** when operating at platform scale — multiple
agents, shared component libraries, centralized deployment infrastructure.
Not relevant for a single-agent deployment.

---

## Priority

If the system reaches Step-8 completion, Gap A (interaction modeling) is the
higher-value next step. It directly improves selection quality for the
existing deployment. Gap B is only relevant if the system is being operated
as a multi-tenant platform.

Neither gap should be started before Step 8 is complete and the joint
classifier has accumulated enough operational data to validate whether
interaction modeling would actually improve on it.
