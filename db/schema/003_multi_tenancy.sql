-- Migration 003: Multi-tenancy with dual-mode isolation
--
-- This migration adds tenant_id columns to all data tables and provides
-- TWO isolation modes that can be used independently or together:
--
-- MODE 1 — Application-level filtering (default, simpler):
--   The application passes tenant_id in all queries (WHERE tenant_id = %s).
--   Easy to set up, no superuser required, works with any Postgres.
--
-- MODE 2 — PostgreSQL Row-Level Security (opt-in, stronger):
--   Postgres enforces isolation at the DB level even if app code has bugs.
--   Requires: SUPERUSER or BYPASSRLS privilege to set up policies.
--   Enable after running this migration by calling: SELECT enable_tenant_rls();
--
-- For regulated industries (HIPAA, SOC2, FedRAMP) we recommend Mode 2.
-- For other deployments, Mode 1 is sufficient.

-- ── Step 1: Add tenant_id columns ──────────────────────────────────────────

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE retrieval_runs
  ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE audit_events
  ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

-- ── Step 2: Indexes for efficient per-tenant filtering ──────────────────────

CREATE INDEX IF NOT EXISTS idx_documents_tenant_id
  ON documents (tenant_id);

CREATE INDEX IF NOT EXISTS idx_chunks_tenant_id
  ON chunks (tenant_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_tenant_id
  ON retrieval_runs (tenant_id);

CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_id
  ON audit_events (tenant_id);

-- Composite indexes for the most common query pattern: tenant + filter
CREATE INDEX IF NOT EXISTS idx_documents_tenant_source
  ON documents (tenant_id, source_type);

-- ── Step 3: RLS policies (Mode 2) — opt-in via enable_tenant_rls() ─────────
--
-- RLS provides database-level enforcement: even a buggy or compromised query
-- cannot read another tenant's data as long as the session variable is set.
--
-- Usage after enabling:
--   SELECT set_config('app.tenant_id', 'your-tenant-id', true);   -- before each query
--   (The application does this automatically; see postgres.py)

CREATE OR REPLACE FUNCTION enable_tenant_rls() RETURNS void AS $$
BEGIN
  -- Documents
  ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
  ALTER TABLE documents FORCE ROW LEVEL SECURITY;
  DROP POLICY IF EXISTS tenant_isolation_documents ON documents;
  CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = current_setting('app.tenant_id', true));

  -- Chunks
  ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
  ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
  DROP POLICY IF EXISTS tenant_isolation_chunks ON chunks;
  CREATE POLICY tenant_isolation_chunks ON chunks
    USING (tenant_id = current_setting('app.tenant_id', true));

  -- Retrieval runs
  ALTER TABLE retrieval_runs ENABLE ROW LEVEL SECURITY;
  ALTER TABLE retrieval_runs FORCE ROW LEVEL SECURITY;
  DROP POLICY IF EXISTS tenant_isolation_retrieval_runs ON retrieval_runs;
  CREATE POLICY tenant_isolation_retrieval_runs ON retrieval_runs
    USING (tenant_id = current_setting('app.tenant_id', true));

  -- Audit events
  ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
  ALTER TABLE audit_events FORCE ROW LEVEL SECURITY;
  DROP POLICY IF EXISTS tenant_isolation_audit_events ON audit_events;
  CREATE POLICY tenant_isolation_audit_events ON audit_events
    USING (tenant_id = current_setting('app.tenant_id', true));

  RAISE NOTICE 'RLS tenant isolation enabled on all tables.';
END;
$$ LANGUAGE plpgsql;

-- To disable RLS (e.g. admin tooling that needs cross-tenant access):
CREATE OR REPLACE FUNCTION disable_tenant_rls() RETURNS void AS $$
BEGIN
  ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
  ALTER TABLE chunks DISABLE ROW LEVEL SECURITY;
  ALTER TABLE retrieval_runs DISABLE ROW LEVEL SECURITY;
  ALTER TABLE audit_events DISABLE ROW LEVEL SECURITY;
  RAISE NOTICE 'RLS tenant isolation disabled on all tables.';
END;
$$ LANGUAGE plpgsql;
