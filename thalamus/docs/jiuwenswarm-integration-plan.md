# Thalamus × JiuwenSwarm Integration Plan

**Goal:** When a user sends a prompt to JiuwenSwarm, Thalamus's precomputed oracle selects
the optimal subset of skills, memory, and tools for that specific query — replacing the
current "load everything" strategy with targeted context assembly in <10 ms, before the
agent starts running.

---

## 1. What Thalamus Produces (Inputs to This Integration)

After running `thalamus-score` + `thalamus-oracle`, the `oracle_dir` contains:

| File | What it is |
|---|---|
| `context_configs.json` | Cluster → {skills, memory, tools} lookup table, built by genetic algorithm |
| `context_configs.pkl` | Fitted TF-IDF K-means or sentence-transformer clusterer |
| `classifier_current.pkl` | Logistic regression trained on real turn logs (available after ~100-500 turns) |
| `classifier_thresholds.json` | Per-component tuned inclusion thresholds (from hyperparameter search) |
| `scoring_matrix_skill_*.json` | Per-skill evaluation data (used offline to build the above, not used at runtime) |

**Runtime selection paths:**

- **Path A — Cluster lookup** (`ClusterSelector`): embed query → nearest K-means cluster →
  look up pre-computed optimal component set. Fast, no LLM, available immediately after
  `thalamus-oracle` runs.
- **Path B — Classifier** (`ClassifierSelector`): embed query → logistic regression per
  component → threshold → include/exclude. More accurate, requires real operational turn
  logs to train. Takes over from Path A once sufficient data accumulates.

The selector returns:
```python
{
    "skills":  ["devops-toolkit", "testing-guide"],   # ordered by relevance
    "memory":  ["Architecture", "TeamConventions"],
    "tools":   ["bash_exec", "git_cmd"],
    "fitness": 3.12,           # Path A only
    "context_tokens": 3654,    # Path A only
    # Path B also includes: "probabilities", "confidence", "source"
}
```

---

## 2. Problem Being Solved

JiuwenSwarm today injects ALL skills into the system prompt (`skill_mode=all`), or lets the
agent call a `list_skill` tool at its own discretion (`auto_list`), or runs a TF-IDF search
over scoring matrices (`recommendation`). None of these are:

1. Multi-component (skills only, not memory + tools jointly)
2. Jointly optimized (each component chosen in isolation, not as a bundle)
3. Budget-aware (no concept of context window cost)
4. Fast enough to run on every message without an LLM call

Thalamus solves all four. The integration makes it the primary context assembly mechanism.

---

## 3. Architecture Overview

```
User prompt
    │
    ▼
ThalamusContextRail.on_message_start(query)          ← NEW (jiuwenswarm)
    │
    ├─ Path B available?  →  ClassifierSelector.select(embed(query))
    │                                                         │
    ├─ Path A available?  →  ClusterSelector.select(query, budget, ordering="bookend")
    │                                                         │
    └─ neither?           →  None  (fall through to original behavior)
    │
    ▼
session_state["thalamus_context"] = {
    "skills":  ["devops-toolkit", ...],
    "memory":  ["Architecture", ...],
    "tools":   ["bash_exec", ...],
}
    │
    ├──► SkillUseRail (skill_mode="thalamus")          ← MODIFIED (agent-core)
    │       reads skills list from session_state
    │       injects ONLY those skills into system prompt
    │
    ├──► MemoryRail                                    ← MODIFIED (jiuwenswarm)
    │       reads memory list from session_state
    │       loads ONLY those memory sections
    │
    └──► Tool registry                                 ← MODIFIED (jiuwenswarm)
            enables ONLY those tools for the session
    │
    ▼
Agent runs with focused, budget-aware context
```

---

## 4. New and Modified Components

### 4.1 Thalamus — `ContextSelector` facade (NEW, Thalamus repo)

**File:** `../thalamus/selection/context_selector.py`

A thin facade that owns the Path B → Path A → None fallback and handles optional imports.
Neither ClusterSelector nor ClassifierSelector implement this fallback themselves.

```python
class ContextSelector:
    """Unified selector: tries Path B (classifier), falls back to Path A (cluster)."""

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "ContextSelector":
        """Load whichever paths are available in oracle_dir."""

    def select(
        self,
        query: str,
        budget: str | None = None,   # None → auto via BudgetEstimator
        ordering: str = "bookend",   # mitigates lost-in-the-middle
    ) -> dict | None:
        """
        Returns {skills, memory, tools, ...} or None if no oracle available.
        Path B is preferred when classifier_current.pkl exists.
        Path A is the fallback.
        """

    @property
    def active_path(self) -> str:
        """Returns "classifier", "cluster", or "none"."""
```

**Why `ordering="bookend"` as default:** Thalamus pre-sorts components by relevance; the
bookend pattern places the most-relevant at the beginning and end of the list, counteracting
the LLM "lost-in-the-middle" attention decay documented in the Thalamus architecture.

---

### 4.2 agent-core — `SKILL_MODE_THALAMUS` in `SkillUseRail` (MODIFIED)

**File:** `openjiuwen/harness/rails/skills/skill_use_rail.py`

Add a fourth mode. In this mode, `SkillUseRail` does NOT build its own skill list —
it reads the list injected by `ThalamusContextRail` from session state.

```python
SKILL_MODE_THALAMUS = "thalamus"
_VALID_SKILL_MODES = {SKILL_MODE_ALL, SKILL_MODE_AUTO_LIST,
                      SKILL_MODE_RECOMMENDATION, SKILL_MODE_THALAMUS}

# In init():
elif self.skill_mode == self.SKILL_MODE_THALAMUS:
    # No tool registered — context was pre-selected upstream.
    # Skills will be read from session state in _build_skills_section().
    pass

# In _build_skills_section():
elif self.skill_mode == self.SKILL_MODE_THALAMUS:
    thalamus_ctx = self._session.state.get("thalamus_context") or {}
    selected = thalamus_ctx.get("skills") or []
    # filter self.skills to only those in selected, preserving order
    visible = [s for s in self.skills if s.name in selected]
    content = build_all_mode_skill_prompt(visible, language)  # reuse existing builder
```

**Why reuse `build_all_mode_skill_prompt`:** Selected skills are injected inline (same as
`all` mode), since Thalamus has already done the relevance filtering. No tool call needed.

---

### 4.3 jiuwenswarm — `ThalamusContextRail` (NEW)

**File:** `jiuwenswarm/agents/swarm/providers/thalamus_rail.py`

A new rail that runs before SkillUseRail. It performs the Thalamus query at the start of
each message and writes results to session state for downstream rails to read.

```python
class ThalamusContextRail(DeepAgentRail):
    """Pre-selects skills, memory, and tools via Thalamus oracle before agent runs."""

    priority = 50  # Must run before SkillUseRail (priority=100) and MemoryRail

    SESSION_KEY = "thalamus_context"

    def __init__(self, oracle_dir: str | Path, budget: str | None = None,
                 ordering: str = "bookend"):
        self._oracle_dir = Path(oracle_dir)
        self._budget = budget  # None → auto
        self._ordering = ordering
        self._selector: ContextSelector | None = None

    def init(self, session):
        super().init(session)
        try:
            from selection import ContextSelector
            self._selector = ContextSelector.load(self._oracle_dir)
            logger.info("Thalamus oracle loaded (path=%s)", self._selector.active_path)
        except Exception:
            logger.warning("Thalamus oracle unavailable at %s; using fallback",
                           self._oracle_dir)

    def on_message_start(self, message, **_):
        if self._selector is None:
            return
        query = message.content if hasattr(message, "content") else str(message)
        try:
            result = self._selector.select(query, budget=self._budget,
                                           ordering=self._ordering)
            if result:
                self._session.state[self.SESSION_KEY] = result
                logger.debug("Thalamus selected: skills=%s memory=%s tools=%s",
                             result.get("skills"), result.get("memory"),
                             result.get("tools"))
        except Exception:
            logger.warning("Thalamus selection failed; falling back to full context",
                           exc_info=True)
```

**Priority 50** ensures it runs before `SkillUseRail` (priority 100). Failures are logged
and swallowed — the agent falls back to its configured behavior.

---

### 4.4 jiuwenswarm — Memory rail integration (MODIFIED)

**File:** `jiuwenswarm/agents/swarm/providers/` (whichever file owns memory injection)

After Thalamus runs, `session_state["thalamus_context"]["memory"]` contains an ordered list
of memory section names. The memory rail should:

1. Check `session_state.get("thalamus_context", {}).get("memory")`.
2. If present and non-empty: load only those sections, in that order.
3. If absent (Thalamus unavailable or returned nothing): existing behavior unchanged.

```python
# Pseudocode inside memory rail's build method:
thalamus_ctx = self._session.state.get("thalamus_context") or {}
selected_memory = thalamus_ctx.get("memory")
if selected_memory:
    sections = [s for s in all_sections if s.name in selected_memory]
    sections.sort(key=lambda s: selected_memory.index(s.name))  # preserve Thalamus order
else:
    sections = all_sections  # original behavior
```

---

### 4.5 jiuwenswarm — Tool registry integration (MODIFIED)

**File:** wherever tools are registered per-session in jiuwenswarm

Same pattern: check `session_state["thalamus_context"]["tools"]` and restrict enabled tools.

```python
thalamus_ctx = self._session.state.get("thalamus_context") or {}
selected_tools = thalamus_ctx.get("tools")
if selected_tools:
    tools = [t for t in all_tools if t.name in selected_tools]
else:
    tools = all_tools
```

---

### 4.6 jiuwenswarm — Config wiring (NEW/MODIFIED)

**`config.yaml`** — new section under `react:`:
```yaml
react:
  skill_mode: thalamus          # enable Thalamus mode
  oracle_dir: ~/.jiuwenswarm/agent/workspace/oracle
  thalamus:
    budget: null                # null → auto (BudgetEstimator), or "small"/"medium"/"large"
    ordering: bookend           # bookend | relevance | none
    selector: auto              # auto | cluster | classifier
```

**`config_specs.py`** — new helpers:
```python
def _thalamus_enabled(config: dict) -> bool:
    return _skill_mode(config) == SkillUseRail.SKILL_MODE_THALAMUS

def _thalamus_config(config: dict) -> dict:
    return _config_section(config, "react").get("thalamus") or {}
```

**Provider registry** — register `THALAMUS_CONTEXT` element:
```python
registry.THALAMUS_CONTEXT: lambda c: {
    "oracle_dir": _oracle_dir(c),
    "budget": _thalamus_config(c).get("budget"),
    "ordering": _thalamus_config(c).get("ordering", "bookend"),
},
```

**`code_rails.py`** — build `ThalamusContextRail` when skill_mode is thalamus:
```python
def build_thalamus_context_rail(params, ctx):
    inp = ThalamusContextInput.resolve(params, ctx)
    return ThalamusContextRail(
        oracle_dir=inp.oracle_dir,
        budget=inp.budget,
        ordering=inp.ordering,
    )
```

---

## 5. Runtime Flow (End-to-End)

```
1. User sends: "Set up CI/CD pipeline with Docker"

2. ThalamusContextRail.on_message_start():
   - BudgetEstimator.estimate(query) → "large"  (multi-step query detected)
   - Path B available? No (no real turn logs yet)
   - Path A: ClusterSelector.select(query, budget="large", ordering="bookend")
     → embed "Set up CI/CD pipeline with Docker" → cluster_id=7
     → lookup cluster 7 / budget_large
     → {
         skills:  ["devops-toolkit", "docker-deployment", "testing-guide", "kubernetes-guide"],
         memory:  ["Architecture", "TeamConventions", "DeploymentChecklist"],
         tools:   ["bash_exec", "git_cmd", "docker_tool", "kubectl_tool"],
         fitness: 3.89,
         context_tokens: 7423
       }
   - Write to session_state["thalamus_context"]

3. SkillUseRail (skill_mode="thalamus"):
   - Reads ["devops-toolkit", "docker-deployment", "testing-guide", "kubernetes-guide"]
   - Filters self.skills to only those 4 (out of potentially 50+ installed skills)
   - Injects their SKILL.md bodies into system prompt

4. MemoryRail:
   - Reads ["Architecture", "TeamConventions", "DeploymentChecklist"]
   - Loads only those 3 memory sections
   - Order preserved (bookended for attention)

5. Tool registry:
   - Enables only ["bash_exec", "git_cmd", "docker_tool", "kubectl_tool"]
   - Agent cannot accidentally call unrelated tools

6. Agent runs with focused, budget-capped context (~7400 tokens) instead of
   all 50+ skills + all memory + all tools.
```

---

## 6. Fallback Chain

```
Thalamus oracle available?
    ├─ YES: classifier_current.pkl exists?
    │         ├─ YES → Path B (ClassifierSelector) — most accurate
    │         └─ NO  → Path A (ClusterSelector)   — pre-computed, fast
    │
    └─ NO (oracle_dir missing, files absent, import error, selection error):
           session_state["thalamus_context"] is NOT set
           └─ SkillUseRail falls back to its configured skill_mode
              (e.g., "auto_list" or "all" if thalamus is unavailable)
```

Configuring a safe fallback:
```yaml
react:
  skill_mode: thalamus
  thalamus_fallback_mode: auto_list   # NEW — used when Thalamus unavailable
```

`SkillUseRail` checks for the session state key; if absent, uses `thalamus_fallback_mode`.

---

## 7. Component Name Mapping

Thalamus component names must match jiuwenswarm's actual skill/memory/tool names.

**Potential issue:** Thalamus scoring uses whatever names were passed to `thalamus-score
--type skill --skills-dir <path>`. If those names match the directory names of installed
skills, they match automatically. If not, a mapping table is needed.

**Recommended approach:** Run `thalamus-score` pointing at the same `skills_dir` that
JiuwenSwarm uses (i.e., `get_agent_skills_dir()`). This ensures names are consistent.

```bash
thalamus-score build \
  --type skill \
  --skills-dir ~/.jiuwenswarm/agent/workspace/skills \
  --oracle-dir ~/.jiuwenswarm/agent/workspace/oracle
```

---

## 8. Implementation Sequence

### Phase 1 — Thalamus repo (no jiuwenswarm/agent-core changes)
1. Add `ContextSelector` facade (`../thalamus/selection/context_selector.py`)
   - Path B → Path A → None fallback
   - Export from `__init__.py`

### Phase 2 — agent-core changes
2. Add `SKILL_MODE_THALAMUS` to `SkillUseRail`
   - Read from `session_state["thalamus_context"]["skills"]`
   - Reuse `build_all_mode_skill_prompt()` for injection
3. Add `thalamus_fallback_mode` parameter to `SkillUseRail.__init__`
4. Unit tests for thalamus mode (mock session state)

### Phase 3 — jiuwenswarm changes
5. Add `ThalamusContextRail` (`providers/thalamus_rail.py`)
6. Register in provider registry
7. Add `ThalamusContextInput` dataclass
8. Update `config.yaml` (new `thalamus:` subsection)
9. Update `config_specs.py` (`_thalamus_config()`, `_thalamus_enabled()`)
10. Integrate memory rail (read `session_state["thalamus_context"]["memory"]`)
11. Integrate tool registry (read `session_state["thalamus_context"]["tools"]`)

### Phase 4 — Validation
12. Run `thalamus-score + thalamus-oracle` against real skills dir
13. Set `skill_mode: thalamus` in config.yaml with oracle_dir
14. Smoke-test with a representative set of queries
15. Verify that Path B activates after accumulating turn logs

---

## 9. What This Does NOT Change

- `SkillForge` integration and `scoring_matrix_*.json` files: unchanged. Those files are
  the input to `thalamus-score`, not to jiuwenswarm directly.
- `skill_mode: all | auto_list | recommendation`: all three remain fully functional. The
  new `thalamus` mode is additive.
- The existing `RecommendSkillTool` (TF-IDF on scoring matrices): unchanged, remains
  available via `skill_mode: recommendation` for deployments without a Thalamus oracle.

---

## 10. Key Properties of the Integrated System

| Property | Value |
|---|---|
| Latency added per request | <10 ms (Path A cluster lookup) |
| LLM calls added per request | 0 |
| Components covered | Skills + Memory + Tools jointly |
| Context budget enforcement | Yes (small/medium/large tiers) |
| Relevance ordering | Bookend (mitigates lost-in-the-middle) |
| Graceful degradation | Yes — any failure falls back to original behavior |
| Cold start | Path A available immediately after `thalamus-oracle` |
| Self-improving | Yes — Path B improves as turn logs accumulate |
