# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-11-12

### Added
- **RLS-Safe Connection Pooling (`RLSSafeConnectionPool`)**: Reuses physical PostgreSQL connections safely across tenants with startup `is_superuser` caching, bounded `max_active` sockets, and `conn.rollback()` + `SELECT set_config('app.tenant_id', '', false)` scrubbing on every return.
- **CASE-Bench Code Security Classifier & Reranker (`chunking/case_guard.py`, `CaseGuardReranker`)**: AST-based Python and polyglot security classification (`is_security_safeguard`, `has_insecure_exemplar`), safeguard RRF boosting, insecure-exemplar quarantine, parameterized auth decorator inspection, `hashlib` `usedforsecurity=False` handling, and `DocumentIngestionService` / `DocumentChunkingService` support for TypeScript, JavaScript, Go, and Rust files.

## [0.1.3] - 2026-11-07

### Added
- Migration `007_tenant_scoped_uri.sql`: Tenant-scoped document URI uniqueness (`UNIQUE (tenant_id, source_uri)`), resolving cross-tenant document ingestion collisions and Row-Level Security (RLS) privilege violations.
- Token-efficient retrieval output formats in `retrieve_evidence`: Added `format: 'json' | 'compact' | 'markdown'`. `format='markdown'` emits pre-formatted Markdown citations reducing context window footprint by >80%, and `format='compact'` eliminates duplicate chunk text payload.
- Automatic default tenant fallback in `create_mcp_server` and `run_mcp_server`, guaranteeing plug-and-play reliability for Claude Desktop, Cursor, and AI agents when `tenant_id` is omitted.
- Dedicated end-user remediation test suite `tests/test_end_user_evaluation_remediations.py`.

### Changed
- Hardened `RetrievalPipeline.retrieve_with_trace` against telemetry/audit failure cascade: audit logger exceptions are safely caught and recorded in warnings, allowing degraded retrieval to succeed even during database telemetry disruptions.
- Truthful `health_check` status reporting: sets `status = "error"` when database is unreachable, and `status = "degraded"` when embedding dimensions mismatch.
- Standardized `docker-compose.yml`: removed obsolete `version` specification and configured `KF_DEFAULT_TENANT: default` across services.

### Security
- Eliminated cross-tenant document overwrite and ingestion denial-of-service by scoping document identity to `(tenant_id, source_uri)` in PostgreSQL and `KnowledgeRepository.upsert_document`.

## [0.1.2] - 2026-09-28

### Added
- Extensible `EmbeddingProviderRegistry` supporting custom provider registration via `register(name, factory)` and dynamic lookup.
- Native `GeminiEmbeddingProvider` implementing Google Gemini `text-embedding-004` (768d) with zero extra dependencies via standard library HTTP.
- Local `SentenceTransformerEmbeddingProvider` supporting Hugging Face models (`all-MiniLM-L6-v2`, `bge-base-en-v1.5`, etc.).
- Cryptographic evidence package HMAC signing (`package_signature` and `compute_package_signature`) for tamper-evident provenance.
- Apache Tika container service and healthchecks in `docker-compose.yml`.
- Non-superuser least-privilege role migration `006_application_role.sql` (`kf_app`) and `FORCE ROW LEVEL SECURITY` on tables.
- Standalone `reembed_chunks` helper in `knowledge_fabric.db.dimension_guard` for database vector migrations.

### Changed
- Fixed PostgreSQL RLS parameterization by replacing `SET LOCAL app.tenant_id = %s` with `SELECT set_config('app.tenant_id', %s, true)`.
- Enforced fail-closed tenant scoping in MCP server (`_require_tenant`) preventing unscoped cross-tenant data leaks.
- Fixed audit logging permission errors under RLS by scoping `retrieval_runs` and `audit_events` with `tenant_id`.
- Added `skip_audit=True` option to `RetrievalPipeline.retrieve_with_trace` so `explain_retrieval` runs without audit side-effects.
- Normalized Tika endpoint URL handling to prevent duplicate `/tika/tika` paths.
- Bound all Docker Compose ports to loopback (`127.0.0.1`) to eliminate public interface exposure.

### Security
- Enforced non-superuser connection roles (`kf_app`) in `create_postgres_connection_factory` with static and dynamic `is_superuser` validation (KF-01).
- Enforced fail-closed session tenant isolation in `_fetch_rows`, `upsert_document`, and `replace_chunks`, ensuring unauthenticated or cross-tenant queries return empty under RLS.
- Added comprehensive RLS isolation test suite `tests/test_rls_isolation.py`.


## [0.1.1] - 2026-09-06

### Added
- Visual Admin & Governance Console (`knowledge-fabric-ui`) with zero frontend build dependencies, serving an interactive approval queue, live retrieval playground, and policy test sandbox.
- Production Helm chart (`deploy/helm/knowledge-fabric`) with flexible embedded pgvector or external DB options, HPA, Ingress TLS, and Kubernetes secret management.
- Multi-stage production `Dockerfile` with non-root user (`UID 10001`), healthchecks, and `docker-compose.prod.yml` turnkey deployment.
- High-scale vector search optimizations with HNSW indexing migration (`005_hnsw_indexes.sql`), PostgreSQL declarative tenant partitioning, and zero-dependency `QdrantRetrievalStore` adapter.
- STRIDE security hardening: strict tenant ID validation (`^[a-zA-Z0-9_.:@-]{1,128}$`), query length bounds, top_k bounds [1, 100], reranker candidate pool capping, and HMAC approval signature validation.
- Backwards-compatible CLI argument aliases (`--source-path`, `--collection`, `--source-type`) in `knowledge_fabric.ingestion.cli`.

### Changed
- Standardized ruff and pyright CI workflows with Python 3.11 and 3.12 support.
- Configured production database user default to `knowledge_fabric`.

## [0.1.0] - 2026-07-12

### Added
- Core Knowledge Fabric retrieval pipeline with hybrid RRF fusion (lexical FTS + vector similarity via pgvector).
- Pluggable embedding providers (Ollama, OpenAI, Cohere, Mock).
- Document ingestion pipeline supporting Markdown, text, and JSON with chunking and metadata extraction.
- FastMCP tool server integration (`retrieve_evidence`, `get_document`, `explain_retrieval`, `health_check`, `list_sources`).
