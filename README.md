# Thalamus — monorepo

Two packages, one repository.

| Package | Purpose | Docs |
|---------|---------|------|
| [`thalamus/`](thalamus/README.md) | Production runtime — offline scoring, genetic algorithm oracle, sub-10ms context selection | [`thalamus/docs/`](thalamus/docs/) |
| [`thalamus_research/`](thalamus_research/README.md) | Research tooling — baselines, evaluation harness, ablations, R1–R5 experiments | [`thalamus_research/docs/`](thalamus_research/docs/) |

---

## Setup

```bash
git clone <repo>
cd Thalamus
export PYTHONPATH="$PWD"   # makes both packages importable immediately
uv sync                    # optional: registers all CLI scripts in .venv
```

## CLI scripts (after `uv sync`)

```bash
thalamus-score  ...    # Phase 1: score components via LLM
thalamus-oracle ...    # Phase 2: build evolutionary oracle
thalamus-select ...    # Runtime: resolve query → context config (<10ms)
thalamus-research ...  # Research: baselines, ablations, R1-R5 experiments
```

## Workspace layout

```
Thalamus/
  pyproject.toml              ← uv workspace root (no package of its own)

  thalamus/                   ← production package
    pyproject.toml
    README.md
    docs/
    runners/
    tests/
    __init__.py
    _shared/ oracle/ scoring/ selection/ skills/

  thalamus_research/          ← research package
    pyproject.toml
    README.md
    docs/
    runners/
    tests/
    __init__.py
    baselines/ evaluation/ ablations/ cross_path/
    bandit/ set_quality/ meta_learning/
```
