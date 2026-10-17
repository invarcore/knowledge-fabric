# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Test suite proving Row-Level Security (RLS) isolation and non-superuser role restrictions.

Validates KF-01:
1. Superuser connection restriction: Prevents connecting as superuser roles that bypass RLS.
2. Unauthenticated query returns empty: Unset or empty app.tenant_id returns 0 rows under RLS.
3. Cross-tenant query returns empty: app.tenant_id for tenant-B never returns tenant-A records.
4. Same-tenant authenticated query succeeds.
5. Live PostgreSQL engine verification (when a live PostgreSQL instance is available).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from knowledge_fabric.config import DatabaseSettings
from knowledge_fabric.db.connection import (
    PROHIBITED_SUPERUSER_ROLES,
    create_postgres_connection_factory,
)
from knowledge_fabric.db.repository import KnowledgeRepository
from knowledge_fabric.ingestion.models import Chunk, Document, SourceFormat
from knowledge_fabric.retrieval.postgres import PostgresRetrievalStore


# ==============================================================================
# 1. Non-Superuser Connection Role Enforcement Tests (KF-01)
# ==============================================================================


def test_prohibited_superuser_names_rejected_statically() -> None:
    """Connecting as postgres, root, superuser, or admin is blocked before connection."""
    for bad_user in PROHIBITED_SUPERUSER_ROLES:
        settings = DatabaseSettings(
            host="localhost",
            port=5432,
            name="knowledge_fabric",
            user=bad_user,
            password="secret",
        )
        factory = create_postgres_connection_factory(settings, enforce_non_superuser=True)
        with pytest.raises(PermissionError, match=f"'{bad_user}'.*prohibited superuser role"):
            factory()


def test_superuser_privilege_detected_and_rejected_at_connection() -> None:
    """Even if username looks benign, is_superuser=on triggers connection termination."""
    settings = DatabaseSettings(
        host="localhost",
        port=5432,
        name="knowledge_fabric",
        user="custom_dba",
        password="secret",
    )

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("on",)
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=None)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("psycopg.connect", return_value=mock_conn):
        factory = create_postgres_connection_factory(settings, enforce_non_superuser=True)
        with pytest.raises(PermissionError, match="has superuser privileges.*KF-01"):
            factory()

    assert mock_conn.close.called


def test_least_privilege_role_connection_allowed() -> None:
    """Dedicated application role kf_app with is_superuser=off connects cleanly."""
    settings = DatabaseSettings(
        host="localhost",
        port=5432,
        name="knowledge_fabric",
        user="kf_app",
        password="kf_app_password",
    )

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("off",)
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=None)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("psycopg.connect", return_value=mock_conn):
        factory = create_postgres_connection_factory(settings, enforce_non_superuser=True)
        conn = factory()
        assert conn is mock_conn


def test_superuser_enforcement_bypassable_for_admin_tooling() -> None:
    """KNOWLEDGE_ENFORCE_NON_SUPERUSER=false permits superuser connections for migrations."""
    settings = DatabaseSettings(
        host="localhost",
        port=5432,
        name="knowledge_fabric",
        user="postgres",
        password="secret",
    )

    mock_conn = MagicMock()
    with patch("psycopg.connect", return_value=mock_conn):
        # Explicit argument override
        factory = create_postgres_connection_factory(settings, enforce_non_superuser=False)
        conn = factory()
        assert conn is mock_conn

    # Environment variable override
    with patch.dict(os.environ, {"KNOWLEDGE_ENFORCE_NON_SUPERUSER": "false"}):
        with patch("psycopg.connect", return_value=mock_conn):
            factory_env = create_postgres_connection_factory(settings)
            conn_env = factory_env()
            assert conn_env is mock_conn


# ==============================================================================
# 2. Simulated RLS Engine (Fidelity Mock for Hermetic Unit Testing)
# ==============================================================================


class _SimulatedRLSPostgresEngine:
    """Simulates PostgreSQL 16 table storage and Row-Level Security (RLS) enforcement.

    Implements:
    - documents and chunks tables with tenant_id columns.
    - session variable 'app.tenant_id' manipulated via set_config.
    - RLS policy: tenant_id = current_setting('app.tenant_id', true).
    - non-superuser vs superuser RLS bypass semantics.
    """

    def __init__(self, is_superuser: bool = False) -> None:
        self.is_superuser = is_superuser
        self.session_variables: dict[str, str] = {}
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.executions: list[tuple[str, list[Any]]] = []

    def set_config(self, key: str, value: str) -> None:
        self.session_variables[key] = value

    def current_setting(self, key: str) -> str:
        return self.session_variables.get(key, "")

    def _applies_rls_filter(self, row_tenant: str) -> bool:
        if self.is_superuser:
            return True  # Superusers unconditionally bypass RLS
        current_tenant = self.current_setting("app.tenant_id")
        if not current_tenant:
            return False  # Unset or empty tenant context returns empty
        return row_tenant == current_tenant

    def execute_query(self, sql: str, params: list[Any]) -> list[tuple[Any, ...]]:
        self.executions.append((sql, params))

        # Handle set_config
        if "set_config" in sql:
            key = params[0] if len(params) > 0 else ""
            val = params[1] if len(params) > 1 else ""
            # If called as set_config('app.tenant_id', %s, true), val is params[0]
            if "app.tenant_id" in sql and len(params) == 1:
                key = "app.tenant_id"
                val = params[0]
            self.set_config(key, val)
            return [(val,)]

        # Handle chunks query (lexical or vector search)
        if "FROM chunks" in sql:
            results: list[tuple[Any, ...]] = []
            for c in self.chunks:
                # 1. RLS DB-level kernel check
                if not self._applies_rls_filter(c["tenant_id"]):
                    continue
                # 2. Application-level WHERE clause check (Mode 1)
                if "AND d.tenant_id = %s" in sql:
                    app_filter_tenant = params[2] if len(params) > 2 else None
                    if c["tenant_id"] != app_filter_tenant:
                        continue
                results.append(
                    (
                        c["id"],
                        c["document_id"],
                        c["source_uri"],
                        c["chunk_index"],
                        c["chunk_text"],
                        c["metadata"],
                        0.95,  # score
                    )
                )
            return results

        # Handle documents query
        if "FROM documents" in sql:
            results = []
            for d in self.documents:
                if not self._applies_rls_filter(d["tenant_id"]):
                    continue
                results.append(
                    (
                        d["id"],
                        d["source_uri"],
                        d["source_type"],
                        d["title"],
                        d["metadata"],
                        d["content_text"],
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                    )
                )
            return results

        return []


class _SimulatedCursor:
    def __init__(self, engine: _SimulatedRLSPostgresEngine) -> None:
        self.engine = engine
        self._last_result: list[tuple[Any, ...]] = []

    def __enter__(self) -> _SimulatedCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        self._last_result = self.engine.execute_query(sql, params or [])

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._last_result

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._last_result[0] if self._last_result else None


class _SimulatedConnection:
    def __init__(self, engine: _SimulatedRLSPostgresEngine) -> None:
        self.engine = engine
        self.commit_count = 0

    def cursor(self) -> _SimulatedCursor:
        return _SimulatedCursor(self.engine)

    def commit(self) -> None:
        self.commit_count += 1


# ==============================================================================
# 3. Unauthenticated and Cross-Tenant Query Isolation Tests
# ==============================================================================


@pytest.fixture
def rls_engine() -> _SimulatedRLSPostgresEngine:
    engine = _SimulatedRLSPostgresEngine(is_superuser=False)
    # Seed Tenant A records
    engine.documents.append(
        {
            "id": 1,
            "source_uri": "confluence://tenant-alpha/security-runbook",
            "source_type": "confluence",
            "title": "Alpha Incident Runbook",
            "metadata": {"tenant_id": "tenant-alpha"},
            "content_text": "Confidential Alpha key credentials and cluster topology.",
            "tenant_id": "tenant-alpha",
        }
    )
    engine.chunks.append(
        {
            "id": 101,
            "document_id": 1,
            "source_uri": "confluence://tenant-alpha/security-runbook",
            "chunk_index": 0,
            "chunk_text": "Alpha cluster production secrets: AWS_KEY=AKIAIOSFODNN7EXAMPLE",
            "metadata": {"tenant_id": "tenant-alpha"},
            "tenant_id": "tenant-alpha",
        }
    )
    return engine


def test_unauthenticated_lexical_search_returns_empty(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """An unauthenticated query (no tenant_id supplied) returns 0 results under RLS."""
    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(rls_engine))

    hits = store.lexical_search(query_text="production secrets", tenant_id=None)
    assert hits == []
    # Verify set_config cleared tenant context to fail closed
    assert rls_engine.current_setting("app.tenant_id") == ""


def test_unauthenticated_vector_search_returns_empty(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """An unauthenticated vector search returns 0 results under RLS."""
    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(rls_engine))

    hits = store.vector_search(query_embedding=[0.1, 0.2, 0.3], tenant_id=None)
    assert hits == []
    assert rls_engine.current_setting("app.tenant_id") == ""


def test_unauthenticated_get_document_returns_none(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """An unauthenticated get_document lookup returns None under RLS."""
    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(rls_engine))

    doc = store.get_document(document_id=1, tenant_id=None)
    assert doc is None
    assert rls_engine.current_setting("app.tenant_id") == ""


def test_cross_tenant_lexical_search_returns_empty(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """A query authenticated as tenant-beta CANNOT read tenant-alpha records."""
    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(rls_engine))

    hits = store.lexical_search(query_text="production secrets", tenant_id="tenant-beta")
    assert hits == []
    assert rls_engine.current_setting("app.tenant_id") == "tenant-beta"


def test_cross_tenant_vector_search_returns_empty(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """A vector query authenticated as tenant-beta CANNOT read tenant-alpha records."""
    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(rls_engine))

    hits = store.vector_search(query_embedding=[0.1, 0.2, 0.3], tenant_id="tenant-beta")
    assert hits == []
    assert rls_engine.current_setting("app.tenant_id") == "tenant-beta"


def test_cross_tenant_get_document_returns_none(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """Lookup of a tenant-alpha document from tenant-beta context returns None."""
    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(rls_engine))

    doc = store.get_document(document_id=1, tenant_id="tenant-beta")
    assert doc is None
    assert rls_engine.current_setting("app.tenant_id") == "tenant-beta"


def test_cross_tenant_get_chunk_returns_none(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """Lookup of a tenant-alpha chunk from tenant-beta context returns None."""
    repo = KnowledgeRepository(connection_factory=lambda: _SimulatedConnection(rls_engine))

    chunk = repo.get_chunk(101, tenant_id="tenant-beta")
    assert chunk is None
    assert rls_engine.current_setting("app.tenant_id") == "tenant-beta"


def test_authenticated_same_tenant_query_succeeds(rls_engine: _SimulatedRLSPostgresEngine) -> None:
    """A query authenticated as tenant-alpha successfully returns tenant-alpha records."""
    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(rls_engine))

    hits = store.lexical_search(query_text="production secrets", tenant_id="tenant-alpha")
    assert len(hits) == 1
    assert hits[0].chunk_id == 101
    assert "AKIAIOSFODNN7EXAMPLE" in hits[0].chunk_text
    assert rls_engine.current_setting("app.tenant_id") == "tenant-alpha"

    doc = store.get_document(document_id=1, tenant_id="tenant-alpha")
    assert doc is not None
    assert doc["title"] == "Alpha Incident Runbook"


def test_superuser_rls_bypass_demonstrates_vulnerability_when_unrestricted() -> None:
    """Empirical proof of KF-01: When superuser bypasses RLS, an unauthenticated query leaks data."""
    superuser_engine = _SimulatedRLSPostgresEngine(is_superuser=True)
    superuser_engine.chunks.append(
        {
            "id": 999,
            "document_id": 1,
            "source_uri": "confluence://tenant-alpha/secret",
            "chunk_index": 0,
            "chunk_text": "Classified Tenant Alpha Data",
            "metadata": {"tenant_id": "tenant-alpha"},
            "tenant_id": "tenant-alpha",
        }
    )

    store = PostgresRetrievalStore(connection_factory=lambda: _SimulatedConnection(superuser_engine))

    # Without non-superuser enforcement, an unauthenticated query returns Tenant A data
    leaked_hits = store.lexical_search(query_text="Classified", tenant_id=None)
    assert len(leaked_hits) == 1
    assert leaked_hits[0].chunk_id == 999
    # This proves why create_postgres_connection_factory enforces non-superuser roles!


# ==============================================================================
# 4. Live PostgreSQL Engine Integration (Opt-in when DB container is up)
# ==============================================================================


def _is_live_postgres_available() -> bool:
    try:
        import psycopg

        host = os.environ.get("KF_DB_HOST", "localhost")
        port = int(os.environ.get("KF_DB_PORT", "5432"))
        dbname = os.environ.get("KF_DB_NAME", "knowledge_fabric_test")
        user = os.environ.get("KF_DB_USER", "kf_app")
        password = os.environ.get("KF_DB_PASSWORD", "kf_app_password")

        with psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=1,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False


@pytest.mark.skipif(not _is_live_postgres_available(), reason="Live PostgreSQL not available")
def test_live_postgres_rls_isolation_end_to_end() -> None:
    """Full database-level proof against real PostgreSQL engine with RLS active."""
    settings = DatabaseSettings(
        host=os.environ.get("KF_DB_HOST", "localhost"),
        port=int(os.environ.get("KF_DB_PORT", "5432")),
        name=os.environ.get("KF_DB_NAME", "knowledge_fabric_test"),
        user=os.environ.get("KF_DB_USER", "kf_app"),
        password=os.environ.get("KF_DB_PASSWORD", "kf_app_password"),
    )

    factory = create_postgres_connection_factory(settings, enforce_non_superuser=True)
    repo = KnowledgeRepository(connection_factory=factory)
    store = PostgresRetrievalStore(connection_factory=factory)

    # 1. Seed Tenant Alpha Document & Chunks
    doc_alpha = Document(
        source_uri="file:///confidential/alpha.txt",
        source_format=SourceFormat.TXT,
        content_text="Alpha Tenant Top Secret Credentials",
        title="Alpha Top Secret",
        metadata={"classification": "restricted"},
    )
    doc_alpha_id = repo.upsert_document(doc_alpha, tenant_id="tenant-alpha")
    repo.replace_chunks(
        document_id=doc_alpha_id,
        chunks=[
            Chunk(
                document_uri=doc_alpha.source_uri,
                chunk_index=0,
                chunk_text="Alpha production password is SuperSecretAlpha123!",
                metadata={"tenant_id": "tenant-alpha"},
            )
        ],
        tenant_id="tenant-alpha",
    )

    # 2. Unauthenticated query must return empty
    unauth_hits = store.lexical_search("SuperSecretAlpha123", tenant_id=None)
    assert unauth_hits == []

    # 3. Cross-tenant query (tenant-beta) must return empty
    cross_hits = store.lexical_search("SuperSecretAlpha123", tenant_id="tenant-beta")
    assert cross_hits == []

    # 4. Same-tenant query (tenant-alpha) must return the chunk
    auth_hits = store.lexical_search("SuperSecretAlpha123", tenant_id="tenant-alpha")
    assert len(auth_hits) >= 1
    assert "SuperSecretAlpha123" in auth_hits[0].chunk_text

    # 5. Cleanup
    repo.delete_document(document_id=doc_alpha_id, tenant_id="tenant-alpha")
