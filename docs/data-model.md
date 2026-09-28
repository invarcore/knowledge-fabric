# Knowledge Fabric: Canonical Data Model Specification

**Status:** Current Architecture & Verified Schema Invariants  
**Document Type:** Technical Specification and Schema Reference  
**Applicable Versions:** Knowledge Fabric v0.1.x+

---

## 1. Overview and Core Invariants

Knowledge Fabric acts as the *librarian* of the AI assistant ecosystem. It provides persistent storage for normalized source documents, extracted chunk-level retrieval units with preserved provenance, dense vector embeddings, full-text tsvector search columns, and retrieval audit telemetry.

### Core Data Invariants:
1. **Source Provenance:** Every chunk maintains a strict foreign key relationship back to its parent document (`chunks.document_id REFERENCES documents(id) ON DELETE CASCADE`).
2. **Idempotent Document Ingestion:** Documents are uniquely identified by their canonical `source_uri`. Re-ingesting an unchanged or updated source item performs an idempotent upsert (`ON CONFLICT (source_uri) DO UPDATE`) and atomic chunk replacement.
3. **Cascading Clean Deletion:** Deleting a document predictably removes all derived chunks, tsvector indexes, and vector embeddings without leaving orphaned retrieval units in the database.
4. **Tenant Isolation:** Every primary data and operational table (`documents`, `chunks`, `retrieval_runs`, `audit_events`) carries a `tenant_id` column with composite indexing (`idx_documents_tenant_source`).
5. **Dual-Mode Security Isolation:**
   - **Mode 1 (Application Filtering — Always On):** All queries explicitly include `WHERE tenant_id = %s`.
   - **Mode 2 (PostgreSQL Row-Level Security — Opt-in):** Database-level security policies (`tenant_isolation_*`) restrict reads and writes using session variable `app.tenant_id` via `SELECT set_config('app.tenant_id', %s, true)`.
6. **Vector Dimension Guard:** A strict dimension assertion (`DimensionGuard`) verifies that the active embedding provider dimension matches the PostgreSQL `VECTOR(N)` column specification before any vector search executes.

---

## 2. Logical Entities & Schema Definitions

```text
┌─────────────────────────────────────────────────────────────┐
│                          documents                          │
├─────────────────────────────────────────────────────────────┤
│ id: BIGSERIAL PRIMARY KEY                                   │
│ source_uri: TEXT NOT NULL UNIQUE                            │
│ source_type: TEXT NOT NULL                                  │
│ title: TEXT                                                 │
│ metadata: JSONB NOT NULL DEFAULT '{}'                       │
│ content_text: TEXT NOT NULL                                 │
│ tenant_id: TEXT NOT NULL DEFAULT 'default'                  │
│ created_at: TIMESTAMPTZ NOT NULL DEFAULT NOW()              │
│ updated_at: TIMESTAMPTZ NOT NULL DEFAULT NOW()              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ 1 : N (ON DELETE CASCADE)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                            chunks                           │
├─────────────────────────────────────────────────────────────┤
│ id: BIGSERIAL PRIMARY KEY                                   │
│ document_id: BIGINT NOT NULL REFERENCES documents(id)       │
│ chunk_index: INTEGER NOT NULL                               │
│ chunk_text: TEXT NOT NULL                                   │
│ metadata: JSONB NOT NULL DEFAULT '{}'                       │
│ embedding: VECTOR(1536)                                     │
│ search_tsv: tsvector GENERATED ALWAYS AS (to_tsvector(...)) │
│ tenant_id: TEXT NOT NULL DEFAULT 'default'                  │
│ created_at: TIMESTAMPTZ NOT NULL DEFAULT NOW()              │
│ UNIQUE (document_id, chunk_index)                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        retrieval_runs                       │
├─────────────────────────────────────────────────────────────┤
│ id: BIGSERIAL PRIMARY KEY                                   │
│ query_text: TEXT NOT NULL                                   │
│ top_k: INTEGER NOT NULL                                     │
│ retrieval_mode: TEXT NOT NULL                               │
│ filters: JSONB NOT NULL DEFAULT '{}'                        │
│ latency_ms: INTEGER                                         │
│ result_count: INTEGER NOT NULL DEFAULT 0                    │
│ tenant_id: TEXT NOT NULL DEFAULT 'default'                  │
│ created_at: TIMESTAMPTZ NOT NULL DEFAULT NOW()              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         audit_events                        │
├─────────────────────────────────────────────────────────────┤
│ id: BIGSERIAL PRIMARY KEY                                   │
│ event_type: TEXT NOT NULL                                   │
│ actor: TEXT NOT NULL DEFAULT 'system'                       │
│ trace_id: TEXT                                              │
│ event_payload: JSONB NOT NULL DEFAULT '{}'                  │
│ tenant_id: TEXT NOT NULL DEFAULT 'default'                  │
│ created_at: TIMESTAMPTZ NOT NULL DEFAULT NOW()              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Lifecycle Operations

Knowledge Fabric implements a formal data lifecycle covering document registration, chunk replacement, targeted deletion, and tenant purging.

### 3.1 Idempotent Upsert & Atomic Chunk Replacement
When content is scanned by `knowledge-fabric-ingest` or connected sync adapters:
```sql
INSERT INTO documents (source_uri, source_type, title, metadata, content_text, tenant_id)
VALUES ($1, $2, $3, $4::jsonb, $5, $6)
ON CONFLICT (source_uri)
DO UPDATE SET
  source_type = EXCLUDED.source_type,
  title = EXCLUDED.title,
  metadata = EXCLUDED.metadata,
  content_text = EXCLUDED.content_text,
  tenant_id = EXCLUDED.tenant_id,
  updated_at = NOW()
RETURNING id;
```
Following document upsert, existing chunks for that document are cleared and replaced within a single transaction:
```sql
DELETE FROM chunks WHERE document_id = $1;
INSERT INTO chunks (document_id, chunk_index, chunk_text, metadata, embedding, tenant_id)
VALUES (...);
```

### 3.2 Targeted Document Deletion
To delete an individual document by its database ID or canonical URI:
```python
repository.delete_document(document_id=123, tenant_id="team-alpha")
# or
repository.delete_document(source_uri="file:///docs/policy.md", tenant_id="team-alpha")
```
Under the hood:
```sql
DELETE FROM documents WHERE id = $1 AND tenant_id = $2;
```
Due to the foreign key constraint `REFERENCES documents (id) ON DELETE CASCADE`, all corresponding rows in `chunks` (along with their `search_tsv` and `embedding` vector indexes) are removed immediately and atomically.

### 3.3 Source-Level Deletion
When a knowledge source (e.g. Confluence space or local directory) is disconnected or purged:
```python
repository.delete_by_source(source_type="confluence", tenant_id="team-alpha")
```
```sql
DELETE FROM documents WHERE source_type = $1 AND tenant_id = $2;
```

### 3.4 Full Tenant Purge (GDPR / Compliance)
When offboarding an organization or purging a sandbox environment:
```python
repository.purge_tenant(tenant_id="client-omega")
```
The purge sequence executes within an active transaction:
1. `DELETE FROM documents WHERE tenant_id = $1;` (cascades to all `chunks` and vector indexes)
2. `DELETE FROM retrieval_runs WHERE tenant_id = $1;`
3. `DELETE FROM audit_events WHERE tenant_id = $1;`

The operation returns a verified summary of deleted rows:
```json
{
  "tenant_id": "client-omega",
  "documents_deleted": 42,
  "retrieval_runs_deleted": 128,
  "audit_events_deleted": 156
}
```

---

## 4. Indexing Strategy

| Index Name | Table | Strategy / Type | Purpose |
| :--- | :--- | :--- | :--- |
| `idx_documents_source_type` | `documents` | B-Tree (`source_type`) | Filtering by source kind |
| `idx_documents_tenant_id` | `documents` | B-Tree (`tenant_id`) | Primary tenant lookup |
| `idx_documents_tenant_source` | `documents` | B-Tree (`tenant_id`, `source_type`) | Scoped domain filtering |
| `idx_chunks_document_id` | `chunks` | B-Tree (`document_id`) | Document-to-chunk join and cascading delete |
| `idx_chunks_tenant_id` | `chunks` | B-Tree (`tenant_id`) | Tenant-isolated chunk access |
| `idx_chunks_search_tsv` | `chunks` | GIN (`search_tsv`) | Sub-millisecond full-text BM25/tsquery search |
| `idx_chunks_embedding` | `chunks` | IVFFlat / HNSW (`embedding vector_cosine_ops`) | Approximate nearest-neighbor vector retrieval |
| `idx_retrieval_runs_tenant_id` | `retrieval_runs` | B-Tree (`tenant_id`) | Tenant run analytics |
| `idx_audit_events_tenant_id` | `audit_events` | B-Tree (`tenant_id`) | Tenant security auditing |

---

## 5. Sample Verification SQL

```sql
-- 1. Verify foreign key cascade
SELECT count(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents);
-- Expected: 0 orphaned chunks

-- 2. Verify tenant integrity
SELECT d.tenant_id, count(c.id) as chunk_count
FROM documents d
JOIN chunks c ON c.document_id = d.id
WHERE d.tenant_id != c.tenant_id
GROUP BY d.tenant_id;
-- Expected: 0 rows (all chunks match document tenant)

-- 3. Verify dimension alignment
SELECT atttypmod as vector_dimensions
FROM pg_attribute
WHERE attrelid = 'chunks'::regclass AND attname = 'embedding';
-- Expected: 1536 (or configured dimension)
```
