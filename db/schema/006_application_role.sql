-- Migration 006: Dedicated Least-Privilege Application Role (kf_app)
--
-- Security best practice:
-- Connect using a dedicated non-superuser application role rather than
-- a database superuser. This prevents table schema drops and strictly enforces RLS.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'kf_app') THEN
        CREATE ROLE kf_app WITH LOGIN PASSWORD 'kf_app_password' NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    ELSE
        ALTER ROLE kf_app WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;

    -- Automatically enable RLS if function exists
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'enable_tenant_rls') THEN
        PERFORM enable_tenant_rls();
    END IF;
END
$$;

-- Grant least-privilege DML permissions on application tables
GRANT USAGE ON SCHEMA public TO kf_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO kf_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO kf_app;

-- Default privileges for future tables and sequences
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO kf_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO kf_app;
