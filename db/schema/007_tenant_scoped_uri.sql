-- Migration 007: Tenant-Scoped Document URI Uniqueness
--
-- Replaces global UNIQUE(source_uri) constraint with composite UNIQUE(tenant_id, source_uri).
-- In a multi-tenant deployment, multiple tenants may legitimately have documents
-- with identical relative paths (e.g. README.md, architecture.md) without cross-tenant
-- collision or Row-Level Security (RLS) privilege violations.

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_uri_key;
DROP INDEX IF EXISTS idx_documents_tenant_uri;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_tenant_uri ON documents (tenant_id, source_uri);

