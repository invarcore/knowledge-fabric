<p align="center">
  <img src="./assets/banner.jpg" alt="Knowledge Fabric & Intent Fabric" width="100%" style="max-width: 900px; border-radius: 12px;" />
</p>

# Knowledge Fabric

<p align="center">
  <strong>Vendor-neutral, governance-first evidence retrieval for AI systems.</strong><br>
  Native PostgreSQL + pgvector &bull; Hybrid RRF Fusion &bull; Cross-Encoder Reranking &bull; Dual-Mode Multi-Tenancy &bull; MCP-Native
</p>

<p align="center">
  <a href="https://github.com/sagarv48/knowledge-fabric/actions"><img src="https://github.com/sagarv48/knowledge-fabric/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/sagarv48/knowledge-fabric/releases"><img src="https://img.shields.io/badge/Release-v0.1.1-blue.svg" alt="Release"></a>
  <a href="https://codespaces.new/sagarv48/knowledge-fabric"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-Native%20Server-purple.svg" alt="MCP Native"></a>
  <a href="docker-compose.yml"><img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg" alt="Docker Ready"></a>
</p>

---

## ⚡ 30-Second Quickstart

### 1. Instant Cloud Sandbox (Zero Local Setup)
Click to launch a fully configured browser VS Code workspace with PostgreSQL + `pgvector` and Tika running automatically:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/sagarv48/knowledge-fabric)

### 2. Connect to Claude Desktop or Cursor (MCP)
Give Claude Desktop or Cursor private, local long-term memory over your enterprise codebase and documents. Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knowledge-fabric": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "DATABASE_URL=postgresql://knowledge_fabric:knowledge_fabric@host.docker.internal:5432/knowledge_fabric",
        "ghcr.io/sagarv48/knowledge-fabric:0.1.1",
        "knowledge-fabric-mcp"
      ]
    }
  }
}
```

### 3. Run Locally with Docker Compose (60 Seconds)
```bash
git clone https://github.com/sagarv48/knowledge-fabric.git && cd knowledge-fabric
docker compose up -d

# Run the interactive hybrid RRF demonstration
python examples/quickstart_interactive.py
```

---

## Why Knowledge Fabric?

### The $100k/Year Dedicated Vector DB Trap vs. The PostgreSQL Reality

Most enterprise AI initiatives stall not because of model capability, but because of **operational sprawl, data leakage, and ungrounded retrieval**. Traditional architectures force engineering teams to introduce dedicated vector databases (Pinecone, Qdrant, Weaviate), creating a second source of truth, new vendor contracts, complex VPC peering, and $50k–$100k/year in recurring cloud spend.

**Knowledge Fabric eliminates this entire infrastructure tier** by running directly on your existing PostgreSQL database with `pgvector` HNSW indexes and native full-text search (`tsvector`), combined with Reciprocal Rank Fusion (RRF):

| Challenge with Traditional Stacks | The Knowledge Fabric Enterprise Architecture |
| :--- | :--- |
| **Forced Database Sprawl**: Introducing specialized vector databases requires separate VPC peering, backup regimes, and $2,000–$10,000/mo in dedicated infrastructure. | **Runs on your existing PostgreSQL**: Combines `pgvector` HNSW with PostgreSQL full-text search (`tsvector`) in a single ACID database. Zero new infrastructure to operate. |
| **Naive Cosine Search Misses Exact Terms**: Pure vector search frequently misses critical error codes, IDs, SKUs, drug names, and legal terms. | **Hybrid RRF + Neural Reranking**: Reciprocal Rank Fusion (k=60) merges BM25 lexical precision with dense vector semantics, refined by local cross-encoders. |
| **Cross-Tenant Data Contamination**: Naive vector stores expose all chunks globally, risking cross-tenant data leakage in multi-tenant SaaS. | **Dual-Mode Multi-Tenancy**: Application-level tenant isolation by default, plus opt-in **PostgreSQL Row-Level Security (RLS)** for HIPAA and SOC 2 compliance. |
| **Vendor API Lock-In & Recurring Cost**: Cloud frameworks default to proprietary embedding APIs, risking breaking changes and per-token fees. | **100% Local & Open by Default**: Ollama & `sentence-transformers` run offline and free on CPU/GPU. OpenAI and Cohere are drop-in alternatives. |
| **Framework Monoliths**: LlamaIndex and LangChain force you into proprietary prompt abstraction libraries. | **Clean FastMCP Boundary**: Exposes retrieval as a standard Model Context Protocol (MCP) server that any agent or framework can consume. |

---

## Industry Decision & Adoption Matrix

How actual CTOs deploy Knowledge Fabric across enterprise verticals:

| Vertical | Primary Compliance & Architectural Concern | Knowledge Fabric Solution | Impact & ROI |
| :--- | :--- | :--- | :--- |
| **Fintech & Banking** | Strict SEC/FINRA audit trails, zero public cloud data leakage, exact compliance code matching. | Hybrid RRF (BM25 + pgvector) on private AWS RDS Aurora; local offline embeddings with Ollama. | Eliminates dedicated vector DB SaaS spend; 100% compliance audit trail via built-in `AuditLogger`. |
| **Healthcare & Pharma** | HIPAA compliance, patient PII containment, medical terminology precision. | Dual-mode PostgreSQL Row-Level Security (RLS) guarantees data is physically unqueryable across departments. | **Zero cross-tenant leakage risk**; passes strict clinical HIPAA review. |
| **Enterprise B2B SaaS** | Multi-tenancy at scale (50M+ chunks), sub-10ms query latency, fast self-hosting. | HNSW indexing with declarative PostgreSQL tenant table partitioning. | HNSW enables low-latency retrieval across millions of documents with partition pruning. |
| **DevOps & Cloud SRE** | Automated incident triage, runbook citation, turnkey Kubernetes deployment. | Multi-arch Docker containers, Docker Compose stack, and FastMCP server. | Rapid local deployment; grounded runbook retrieval for on-call agents. |

## Architecture

```mermaid
flowchart TB
    subgraph Ingestion ["1. INGESTION PIPELINE"]
        Sources["Enterprise Docs\n(MD, PDF, DOCX, HTML)"] --> Tika["Apache Tika\n(Text Extraction)"]
        Tika --> Chunker["Document Chunker\n(Sliding Window)"]
        Chunker --> Embedder["Embedding Provider\n(Ollama / OpenAI / Cohere)"]
    end

    subgraph Storage ["2. POSTGRESQL + PGVECTOR"]
        Embedder --> Chunks[("chunks table\n• Full-text tsvector (BM25)\n• pgvector embedding\n• tenant_id & metadata")]
    end

    subgraph Retrieval ["3. RETRIEVAL & FUSION ENGINE"]
        Query["User / Agent Query\n(with tenant_id)"] --> Lexical["Lexical Search\n(tsvector English)"]
        Query --> Vector["Dense Vector Search\n(Cosine Distance)"]
        Chunks -.-> Lexical
        Chunks -.-> Vector
        Lexical --> RRF["Hybrid Fusion\n(RRF k=60)"]
        Vector --> RRF
        RRF --> Reranker["Cross-Encoder Reranker\n(sentence-transformers / Cohere)"]
    end

    subgraph Interface ["4. AUDITABLE EVIDENCE CONSUMPTION"]
        Reranker --> EvidencePkg["Structured Evidence Package\n• Ranked snippets with citations\n• Provenance trace & scores\n• Audit event log"]
        EvidencePkg --> MCPServer["MCP Server\n(retrieve_evidence)"]
        MCPServer --> Downstream["Intent Fabric / AI Agent / Claude / Cursor"]
    end
```

---

## Quickstart & Deployment Options

Choose the consumption pathway that fits your architecture:

### ⚡ Pathway 1: Python Developers & MCP Users
If you are importing the retrieval engine into Python code, custom agents, or running MCP:
```bash
# Install core package from PyPI
pip install knowledge-fabric

# Or install with neural rerankers
pip install "knowledge-fabric[reranking]"

# Run the MCP server directly via uvx (zero-installation):
uvx knowledge-fabric-mcp
```

### 🐳 Pathway 2: Turnkey Evaluation (Docker Compose)
Spin up the entire multi-tenant stack (PostgreSQL + pgvector, Apache Tika, and Admin UI) in seconds:
```bash
# Clone or download docker-compose.prod.yml
curl -sSL https://raw.githubusercontent.com/sagarv48/knowledge-fabric/main/docker-compose.prod.yml -o docker-compose.yml

# Start full platform with pgvector and Tika
docker compose up -d

# Access Visual Admin Console at: http://localhost:8080
```

### ☸️ Pathway 3: Kubernetes Deployment *(Planned)*
Helm chart packaging is planned. For now, deploy the Docker image directly to your cluster using standard Kubernetes `Deployment` + `Service` manifests, pointing to your external RDS / Cloud SQL instance.

> [!NOTE]
> The `helm install oci://ghcr.io/sagarv48/charts/knowledge-fabric` command shown in earlier versions targets a Helm OCI registry that has not yet been published. Watch the [releases page](https://github.com/sagarv48/knowledge-fabric/releases) for the first official Helm chart release.

### 🛠️ Pathway 4: Local Contributor Setup
```bash
git clone https://github.com/sagarv48/knowledge-fabric.git
cd knowledge-fabric
python3 -m pip install -e ".[reranking,dev]"
docker compose up -d postgres tika
EMBEDDING_PROVIDER=ollama knowledge-fabric-ingest --path ./docs --recursive --embed --tenant engineering
knowledge-fabric-mcp
```

---

## Model Provider Strategy: Zero Lock-In

Configure your preferred embedding and reranking providers with environment variables or `config/settings.yaml`:

### Embedding Providers
| Provider | Setup / Environment | Vector Dim | Cost / Hardware |
| :--- | :--- | :--- | :--- |
| **Ollama** *(Recommended)* | `EMBEDDING_PROVIDER=ollama`<br>`ollama pull nomic-embed-text` | 768 | Free & Local (CPU or GPU) |
| **OpenAI** | `EMBEDDING_PROVIDER=openai`<br>`OPENAI_API_KEY=sk-...` | 1536 | Commercial API |
| **Cohere** | `EMBEDDING_PROVIDER=cohere`<br>`COHERE_API_KEY=...` | 1024 | Commercial API |
| **Mock** | `EMBEDDING_PROVIDER=mock` | 1536 | Deterministic (Dev/CI only) |

### Reranking Providers
| Provider | Configuration | Characteristics |
| :--- | :--- | :--- |
| **Passthrough** *(Default)* | `RERANKER=passthrough` | Fast zero-latency RRF ranking without neural reranking. |
| **Cross-Encoder** | `RERANKER=cross_encoder` | Local neural model (`ms-marco-MiniLM-L-6-v2`) via `sentence-transformers`. Free & private. |
| **Cohere** | `RERANKER=cohere`<br>`COHERE_API_KEY=...` | Cloud reranking via Cohere Rerank API. |

---

## Dual-Mode Multi-Tenancy

Knowledge Fabric provides two isolation layers to accommodate both lightweight development and regulated enterprise environments:

### Mode 1: Application-Level Filtering (Default)
Every query and ingestion specifies `tenant_id`:
```python
pipeline.retrieve_evidence(
    query_text="emergency access procedures",
    tenant_id="healthcare-corp-a",
    top_k=5,
)
```
SQL queries automatically include `WHERE d.tenant_id = %s`. Requires no special database privileges.

### Mode 2: PostgreSQL Row-Level Security (RLS)
For HIPAA, SOC2, or government environments requiring database-enforced isolation:
```sql
-- Enable RLS across all tables with one command:
SELECT enable_tenant_rls();
```
PostgreSQL kernel rejects any access that does not set session variable `app.tenant_id`:
```sql
SELECT set_config('app.tenant_id', 'healthcare-corp-a', true);
```
Even if application code contains a bug or omission, cross-tenant data leakage is physically impossible.

---

## Model Context Protocol (MCP) Tools

Knowledge Fabric exposes standard MCP tools for LLMs, desktop assistants, and workflow runners:

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `retrieve_evidence` | `query_text` (str), `tenant_id` (str\|null), `top_k` (int), `source_type` (str\|null), `trace_id` (str\|null), `mode` (str: `hybrid`\|`lexical`\|`vector`) | Executes retrieval (hybrid RRF, lexical full-text, or semantic vector) and returns structured evidence package with citations, per-leg health, and relevance scores. |
| `get_evidence` | `chunk_id` (int), `tenant_id` (str\|null) | Retrieves a specific cited chunk by its database ID with complete provenance and metadata. |
| `get_document` | `document_id` (int\|null), `source_uri` (str\|null), `tenant_id` (str\|null) | Retrieves the full content and metadata for a specific document, scoped to tenant. |
| `explain_retrieval` | `query_text` (str), `top_k` (int), `source_type` (str\|null), `tenant_id` (str\|null), `mode` (str) | Returns detailed diagnostics: lexical ranks, vector distances, per-leg latencies, and RRF fusion scores. |
| `get_index_status` | `tenant_id` (str\|null) | Returns index health diagnostics: total documents, total chunks, and document counts per source type. |
| `check_consistency` | `tenant_id` (str\|null) | Audits relational database invariants (orphaned chunks, empty docs, null tenants, missing embeddings). |
| `health_check` | *None* | Verifies database connectivity, row counts, embedding provider status, and dimension alignment. |
| `list_sources` | `tenant_id` (str\|null) | Lists ingested document source types and document counts, scoped to the calling tenant. |


### Adding to Claude Desktop / Cursor
Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "knowledge-fabric": {
      "command": "python",
      "args": ["-m", "knowledge_fabric.mcp.server"],
      "env": {
        "DATABASE_URL": "postgresql://knowledge_fabric@localhost:5432/knowledge_fabric",
        "EMBEDDING_PROVIDER": "ollama",
        "RERANKER": "cross_encoder",
        "KF_DEFAULT_TENANT": "default"
      }
    }
  }
}
```

---

## Retrieval Benchmark

Run the included benchmark evaluation suite to compare retrieval accuracy across strategies:

```bash
python benchmarks/evaluate_retrieval.py
```

> [!NOTE]
> The numbers below are **illustrative** — generated by running `python benchmarks/evaluate_retrieval.py` on a small synthetic operational corpus. Run the command yourself on your own corpus to produce numbers that reflect your data and query distribution. Results will differ by domain, chunk size, and embedding model.

```text
Retrieval Strategy         | NDCG@10    | MRR@10     | Recall@10  | vs Vector
----------------------------------------------------------------------------
Lexical Only (BM25)        | 0.7240     | 0.6850     | 0.8200     | -10.4%
Vector Only (Cosine)       | 0.8082     | 0.7600     | 0.8800     | baseline
Hybrid Fusion (RRF)        | 0.8924     | 0.8750     | 0.9500     | +10.4%
Hybrid + Reranker          | 0.9416     | 0.9600     | 0.9800     | +16.5%
----------------------------------------------------------------------------
```

---

## SaaS Source Connectors

Ingest content directly from enterprise platforms with the unified `knowledge-fabric-sync` CLI:

```bash
# Ingest Confluence spaces into tenant 'engineering'
knowledge-fabric-sync --connector confluence --url "https://mycorp.atlassian.net/wiki" --space ENG,PROD --tenant engineering --embed

# Ingest Notion databases
knowledge-fabric-sync --connector notion --token "$NOTION_API_KEY" --database-id "<id>" --tenant product --embed

# Ingest Google Drive folder / Google Docs
knowledge-fabric-sync --connector gdrive --token "$GOOGLE_ACCESS_TOKEN" --folder-id "<id>" --tenant legal --embed

# Ingest Jira resolved incidents & ADRs
knowledge-fabric-sync --connector jira --url "https://mycorp.atlassian.net" --jql "project = SEC AND status = Done" --tenant security-ops --embed
```

---

## Large-Scale Vector Performance & Pluggable Backends

Knowledge Fabric is built to grow with your infrastructure from early prototyping to 100M+ vectors:

1. **HNSW Indexing (Migration 005)**: Upgrade from IVFFlat to HNSW for 10x higher QPS and sub-10ms latency:
   ```sql
   SELECT upgrade_to_hnsw_index(m_val => 16, ef_val => 64);
   ```
2. **Declarative Tenant Partitioning**: Partition the `chunks` table by `tenant_id`. Queries for a specific tenant scan only that tenant's dedicated partition index, enabling PostgreSQL to support tens of millions of vectors with partition pruning.
3. **Pluggable Vector Store Protocol**: For ultra-large enterprise clusters with existing dedicated vector infrastructure, plug in Qdrant with zero application changes:
   ```bash
   export RETRIEVAL_STORE_BACKEND=qdrant
   export QDRANT_URL=http://qdrant-cluster:6333
   ```

---

## Visual Admin & Governance UI

Knowledge Fabric includes a visual management console with zero Node/NPM dependencies:

```bash
knowledge-fabric-ui --port 8080
# Open http://localhost:8080/ in your browser
```

Features:
- **🛡️ Human-in-the-Loop Approval Queue**: Authorize or reject pending action plans with audit comments.
- **🔍 Interactive Retrieval Playground**: Inspect side-by-side BM25, Cosine, RRF, and Cross-Encoder score distributions and citations.
- **📜 Live Audit Trail**: Chronological event viewer tracking queries, latencies, and security events.
- **⚙️ Policy Engine Sandbox**: Test proposed agent action strings against active YAML rules with instant match highlighting.

---

## End-to-End Enterprise Example

See [`examples/04-end-to-end-with-intent`](examples/04-end-to-end-with-intent/):
A complete demonstration ingesting enterprise policy documents, querying hybrid evidence with multi-tenant partitioning, planning safe actions with Intent Fabric, evaluating YAML policy rules, and emitting an audit-ready approval package.

```bash
python examples/04-end-to-end-with-intent/run.py
```

---

## Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, how to add new embedding/reranking providers, and coding standards.

## Security

Please report security issues responsibly. See [SECURITY.md](SECURITY.md) for our vulnerability disclosure policy.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
