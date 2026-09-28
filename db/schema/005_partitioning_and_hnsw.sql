-- Migration 005: HNSW Vector Indexing & Declarative Tenant Partitioning
--
-- This migration addresses large-scale vector search performance:
--
-- 1. HNSW INDEXING:
--    Replaces IVFFlat with HNSW (Hierarchical Navigable Small World).
--    Benefits over IVFFlat:
--      - 10x-20x higher QPS at equivalent recall
--      - Works on empty tables (IVFFlat requires training centroids on populated data)
--      - Constant sub-10ms latency up to millions of vectors per table
--
-- 2. DECLARATIVE TENANT PARTITIONING RECIPE:
--    For deployments with tens of millions of vectors across tenants,
--    partitioning chunks by `tenant_id` enables PostgreSQL partition pruning:
--    a query for tenant 'acme' scans only Acme's isolated 100k-row index,
--    completely bypassing the other 49.9M vectors in the database.

-- Helper to safely upgrade IVFFlat to HNSW
CREATE OR REPLACE FUNCTION upgrade_to_hnsw_index(m_val INT DEFAULT 16, ef_val INT DEFAULT 64)
RETURNS void AS $$
BEGIN
    -- Drop old IVFFlat index if present
    DROP INDEX IF EXISTS idx_chunks_embedding;
    DROP INDEX IF EXISTS idx_chunks_hnsw;

    -- Create HNSW index on vector cosine distance
    EXECUTE format(
        'CREATE INDEX idx_chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = %s, ef_construction = %s)',
        m_val,
        ef_val
    );

    -- Update metadata registry
    INSERT INTO schema_metadata (key, value, updated_at)
    VALUES ('vector_index_type', 'hnsw', NOW())
    ON CONFLICT (key)
    DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();

    RAISE NOTICE 'Successfully upgraded chunks.embedding index to HNSW (m=%, ef_construction=%).', m_val, ef_val;
END;
$$ LANGUAGE plpgsql;

-- Apply HNSW upgrade on migration execution
SELECT upgrade_to_hnsw_index();

-- ── Reference Partitioning Pattern (for 10M+ multi-tenant clusters) ─────────
-- To convert to partitioned storage for massive scale:
--
-- CREATE TABLE IF NOT EXISTS chunks_partitioned (
--   id BIGSERIAL,
--   document_id BIGINT NOT NULL,
--   tenant_id TEXT NOT NULL,
--   chunk_index INTEGER NOT NULL,
--   chunk_text TEXT NOT NULL,
--   metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
--   embedding VECTOR(768),
--   search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
--   created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
--   PRIMARY KEY (tenant_id, id)
-- ) PARTITION BY LIST (tenant_id);
--
-- -- Default partition for standard/unspecified tenants:
-- CREATE TABLE IF NOT EXISTS chunks_default PARTITION OF chunks_partitioned DEFAULT;
--
-- -- Dedicated partition for high-volume enterprise tenant:
-- -- CREATE TABLE chunks_acme PARTITION OF chunks_partitioned FOR VALUES IN ('acme_corp');
-- -- CREATE INDEX idx_chunks_acme_hnsw ON chunks_acme USING hnsw (embedding vector_cosine_ops);
