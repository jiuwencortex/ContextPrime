# THALAMUS

Adaptive context selection for production AI agents.

THALAMUS solves **Context Saturation**: the failure mode where an agent's context window fills with irrelevant skill instructions, memory sections, and tool definitions because the system loads everything, every time. It precomputes the optimal component set for each query type and token budget, then retrieves it at runtime in under 10 ms with no LLM calls.

See [`docs/thalamus.md`](docs/thalamus.md) for the full system design and rationale.

---

## How It Works

Two paths, computed offline and looked up at runtime:

```
PATH A — EVOLUTIONARY (cold start, < ~500 turns)

  Offline:
    1. Component Scoring   — LLM generates (query, answer) pairs per component;
                             evaluates each component against those pairs.
                             Output: scoring_matrix_*.json per component.

    2. Score Enrichment    — Bayesian blend of synthetic scores with real turn
                             data as it accumulates (optional, improves over time).

    3. Query Clustering    — K-means over component example queries (TF-IDF or
                             sentence-transformer). Groups query space into types.

    4. Evolutionary Search — Genetic algorithm (no LLM calls) finds optimal
                             component bitmask per cluster × token budget.
                             Pareto front balances quality vs. token cost.

    5. Write Output        — context_configs.json  (lookup table)
                             context_configs.pkl   (fitted clusterer)

  Runtime:  vectorize query → predict cluster → JSON lookup  (<10 ms)


PATH B — CLASSIFIER (after sufficient turn logs accumulate)

  Offline:
    6. Turn Logging        — agent logs (query_embedding, context_config, outcome)
                             per turn to weekly JSONL files.

    7. Classifier Training — logistic regression per component trained on logged
                             turns; versioned, evaluated, promotion-gated.

  Runtime:  vectorize query → per-component logistic regression  (<1 ms)
```

Path A runs first (cold start). Path B takes over once enough operational data exists. The system degrades gracefully: if the oracle is missing, it returns an empty selection; if the classifier is missing, it falls back to Path A.

---

## Prerequisites

- Python 3.10+
- `openjiuwen` parent package — provides the LLM client (`openjiuwen.core.foundation.llm`)
- An LLM API key (OpenAI-compatible: OpenAI, Kimi, etc.)

---

## Installation

```bash
pip install -e .
```

For semantic embeddings (better cluster quality, requires a transformer model download):

```bash
pip install -e ".[sentence]"
```

For BERTScore evaluation metric:

```bash
pip install -e ".[bertscore]"
```

---

## Running

The fastest way to run the full pipeline is via the shell scripts in `runners/`:

```bash
# Full pipeline: score components → build oracle → validate with a test query
OPENAI_API_KEY=sk-...         \
SKILLS_DIR=/path/to/skills    \
PROJECT_DIR=/path/to/project  \
TOOLS_DIR=/path/to/tools      \
ORACLE_DIR=/path/to/oracle    \
bash runners/run_all.sh
```

Individual steps:

| Script | What it runs |
|---|---|
| `runners/run_01_score.sh` | Phase 1–2: LLM scores all components |
| `runners/run_02_oracle.sh` | Phase 3: genetic algorithm builds `context_configs.json` |
| `runners/run_03_classifier.sh` | Phase 4: trains logistic regression from turn logs |
| `runners/run_04_select.sh` | Runtime: resolves a single query to a context config |

All scripts read configuration from environment variables (paths, API keys, tuning params). See each script's header for the full list.

The underlying CLI commands are also directly available after `pip install`:

```bash
thalamus-score  build --type all  --skills-dir ... --matrix-dir ... --model gpt-4o-mini --api-key ...
thalamus-oracle evolve            --oracle-dir ...
thalamus-select lookup            --oracle-dir ... --query "..." --budget auto
```

Or via `python -m`:

```bash
python -m thalamus.scoring  build   --type all ...
python -m thalamus.oracle     evolve  --oracle-dir ...
python -m thalamus.selection  lookup  --oracle-dir ... --query "..."
```

---

## Python API

```python
# Select context for a query at runtime (Path A — cluster lookup)
from thalamus.selection.by_clusters.cluster_selector import ClusterSelector

selector = ClusterSelector.load("/path/to/oracle")
config = selector.select("Set up a CI pipeline", budget="auto", ordering="bookend")
# config = {"skills": [...], "memory": [...], "tools": [...], "context_tokens": 2140, ...}


# Log an agent turn for classifier training (Path B)
from thalamus._shared.turn_logger import TurnLogger

logger = TurnLogger("/path/to/oracle/online_logs")
turn_id = logger.log_turn(
    query_embedding=embedding_vector,  # numpy array, not raw text
    context_config=config,
    exploration_rate=0.05,
    all_component_names={"skills": [...], "memory": [...], "tools": [...]}
)

# After the agent completes the task:
logger.update_outcome(
    turn_id,
    task_completed=True,
    follow_up_correction=False,
    conversation_length=3,
    skills_used=["devops-toolkit"],
    tools_called=["bash_exec"]
)
```

---

## Project Structure

```
thalamus/
  scoring/                # Phase 1–2: scan, evaluate, score components
    skills/               #   SKILL.md scanner + composer
    memory/               #   Markdown section scanner + composer
    tools/                #   Python AST tool scanner + composer
    enrichment/           #   Bayesian blend of real turn data into scores
    shared/               #   Generic pipeline: generator, evaluator, metrics
      metrics/            #   F1, bigram F1, bag-of-words, length ratio,
                          #   optional BERTScore and LLM-judge
  oracle/                 # Phase 3–4: build lookup table and train classifier
    evolutionary/         #   K-means clustering + genetic algorithm oracle
    classifier/           #   Logistic regression classifier, versioning, tuning
    hyperparameters_tuner/ #  Auto-tune K, λ, C, thresholds from turn data
    rebuild_recommender/  #   Drift detection + staleness checker
  selection/              # Runtime: select context for a query
    by_clusters/          #   Path A: cluster lookup (ClusterSelector)
    by_classifier/        #   Path B: per-component probabilities (ClassifierSelector)
  _shared/                # Cross-package utilities
    turn_logger.py        #   Log agent turns to JSONL
    outcome_scorer.py     #   Quality signal from implicit/explicit signals
    query_clusterer.py    #   TF-IDF / sentence-transformer clustering backend
    context_orderer.py    #   Relevance and bookend ordering strategies

docs/
  thalamus.md             # Full system design paper
  thalamus_slides.md      # Slide-deck version of the paper
  IMPLEMENTATION_PLAN.md  # Development status and step-by-step plan (Steps 1–8)
  future-roadmap.md       # Post-Step-8 directions (interaction modeling, meta-learning)
  SkillRouter.md          # Comparison with SkillRouter (external paper)

runners/
  run_all.sh              # Full pipeline (score → oracle → validate)
  run_01_score.sh         # Phase 1–2: component scoring
  run_02_oracle.sh        # Phase 3: evolutionary oracle
  run_03_classifier.sh    # Phase 4: classifier training
  run_04_select.sh        # Runtime: single query lookup

tests/
  test_skill_discovery.py # Skill parsing, ranking, cosine similarity (19 tests)
  run_tests.py            # pytest runner
```

---

## Tests

```bash
python -m pytest tests/ -v
# or
python tests/run_tests.py
```

---

## Oracle Directory Layout

After a full run, the oracle directory contains:

```
<oracle-dir>/
  scoring_matrix_skill_<name>.json     # one per skill
  scoring_matrix_mem_<name>.json       # one per memory section
  scoring_matrix_tool_<name>.json      # one per tool / tool group
  matrix_state_skills.json             # fingerprint state (skip unchanged)
  matrix_state_memory.json
  matrix_state_tools.json
  context_configs.json                 # Path A lookup table (cluster → config)
  context_configs.pkl                  # fitted clusterer for query → cluster
  classifier_current.pkl               # Path B active classifier (if trained)
  classifier_registry.json             # version history with eval metrics
  online_logs/
    turns_YYYY-WNN.jsonl               # weekly turn logs for classifier training
```

---

## Implementation Status

Steps 1–5 are complete (scoring, evaluation, hyperparameter tuning, semantic
metrics, drift detection). Steps 6–8 (cross-path learning, learned fitness
function, joint classifier) are not started — they require operational turn
logs from a deployed system. See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).
