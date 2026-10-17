# Contributing to knowledge-fabric

Thank you for your interest in contributing. This document explains the process,
architecture, and conventions to follow.

## Table of Contents
- [Development setup](#development-setup)
- [Architecture overview](#architecture-overview)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [How to add an embedding provider](#how-to-add-an-embedding-provider)
- [How to write a source adapter](#how-to-write-a-source-adapter)
- [Pull request process](#pull-request-process)

---

## Development setup

### Prerequisites
- Python ≥ 3.11
- Docker (for the Postgres + pgvector service)
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

```bash
# Clone and install
git clone https://github.com/invarcore/knowledge-fabric.git
cd knowledge-fabric
pip install uv
uv sync --all-extras

# Start Postgres (runs migrations 001-005 automatically)
docker compose up -d

# Or apply schema manually:
psql -h localhost -U knowledge_fabric -d knowledge_fabric \
  -f db/schema/001_extensions.sql \
  -f db/schema/002_core_tables.sql \
  -f db/schema/003_multi_tenancy.sql \
  -f db/schema/004_dimension_guard.sql \
  -f db/schema/005_partitioning_and_hnsw.sql
```

---

## Architecture overview

```
enterprise-adapters (private)
    └── SourceAdapter.fetch_resource()
          ↓
knowledge-fabric (this repo)
    ├── ingestion/       — document parsing, chunking
    ├── embeddings/      — EmbeddingProvider protocol + implementations
    ├── db/              — Postgres repository (documents, chunks, audit_events)
    ├── retrieval/       — lexical + vector search + RRF fusion
    ├── reranking/       — optional neural reranker (cross-encoder, Cohere)
    ├── evidence/        — EvidencePackage data model
    └── mcp/             — MCP server (FastMCP, tools: retrieve_evidence etc.)
          ↓
intent-fabric (separate repo)
    ├── planning/        — RuleBasedPlanner or LLMPlanner (ollama/openai/gemini/foundry)
    ├── policies/        — PolicyEngine (YAML-driven rules, hot-reload)
    └── approvals/       — ApprovalRequest lifecycle
```

The stack is explicitly phase-separated — no layer reaches up or skips a layer.

---

## Running tests

```bash
# All tests (requires Postgres running)
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=knowledge_fabric --cov-report=term-missing
```

---

## Code style

- **Formatter**: `ruff format` (configured in `pyproject.toml`)
- **Linter**: `ruff check`
- **Type checker**: `pyright`
- All public functions and classes must have docstrings
- No `print()` in library code — use `logging` or `warnings`
- New modules must be importable without network calls at import time

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pyright src/
```

---

## How to add an embedding provider

1. Add a `@dataclass(slots=True)` class in [`providers.py`](src/knowledge_fabric/embeddings/providers.py)
2. Implement the `EmbeddingProvider` protocol: `dimension` property + `embed_texts()`
3. Add a `from_env()` classmethod that reads config from environment variables
4. Register in `build_embedding_provider()` factory
5. Add a test in `tests/test_embeddings.py`
6. Document in `config/settings.yaml` under `# embeddings:`

**Rules**: no API calls at import time; use `urllib.request` for HTTP (no extra deps); `dimension` must match the model's actual output dimension.

---

## How to write a source adapter

Community adapters use the public contracts package — no access to the private enterprise repo required:

```bash
pip install knowledge-fabric-adapters
```

```python
from knowledge_fabric_adapters import KnowledgeSourceAdapter, ReadOnlyResource

class SlackAdapter:
    """Fetches messages from a Slack workspace channel."""

    def list_resources(self) -> list[ReadOnlyResource]:
        ...

    def fetch_resource(self, resource_id: str) -> dict[str, object]:
        # Must return: {"resource_id": str, "content": str, "metadata": dict}
        ...
```

---

## Pull request process

1. Fork and create a branch: `git checkout -b feat/my-feature`
2. Write tests for new code — PRs without tests will be asked to add them
3. Run `uv run pytest tests/ -v` and `uv run ruff check src/` before opening a PR
4. Keep PRs focused — one feature or fix per PR
5. Fill in the PR template

PRs are reviewed within 48 hours on business days.
