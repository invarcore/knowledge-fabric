# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Verification test suite for end-user architectural remediations:
1. Tenant-scoped document URI uniqueness (Migration 007).
2. Telemetry and audit failure resilience in degraded retrieval pipeline.
3. Truthful health_check status reporting (error vs degraded vs ok).
4. Token-efficient retrieval output formats (compact JSON & markdown citations).
5. MCP server default tenant fallback for autonomous agent plug-and-play.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from knowledge_fabric.embeddings.providers import MockEmbeddingProvider
from knowledge_fabric.evidence.models import EvidenceItem, EvidencePackage
from knowledge_fabric.ingestion.models import Document, SourceFormat
from knowledge_fabric.mcp.server import create_mcp_server, run_mcp_server
from knowledge_fabric.mcp.tools import KnowledgeFabricMCPTools
from knowledge_fabric.retrieval.models import RetrievalHit
from knowledge_fabric.retrieval.pipeline import RetrievalPipeline


# ── 1. Tenant-Scoped Document URI Uniqueness ──────────────────────────────────


def test_tenant_scoped_document_upsert_query_syntax() -> None:
    """KnowledgeRepository emits ON CONFLICT (tenant_id, source_uri)."""
    from knowledge_fabric.db.repository import KnowledgeRepository

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (42,)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    repo = KnowledgeRepository(connection_factory=lambda: mock_conn)
    doc = Document(
        source_uri="specs/auth.md",
        source_format=SourceFormat.MARKDOWN,
        content_text="Tenant Auth Specs",
        title="Auth Specs",
    )

    doc_id = repo.upsert_document(doc, tenant_id="tenant-beta")
    assert doc_id == 42
    executed_sql = mock_cursor.execute.call_args_list[1][0][0]
    assert "ON CONFLICT (tenant_id, source_uri)" in executed_sql


# ── 2. Telemetry / Audit Failure Resilience ───────────────────────────────────


def test_audit_failure_does_not_crash_degraded_pipeline() -> None:
    """When DB/audit logger fails, pipeline logs warning and returns package."""
    mock_store = MagicMock()
    # Lexical leg succeeds
    mock_hit = RetrievalHit(
        chunk_id=1,
        document_id=1,
        document_uri="doc://1",
        chunk_index=0,
        chunk_text="lexical result",
        metadata={},
        score=0.9,
        source="lexical",
    )
    mock_store.lexical_search.return_value = [mock_hit]

    # Audit logger explodes (e.g. database disconnect)
    mock_audit = MagicMock()
    mock_audit.log_retrieval.side_effect = RuntimeError("Deadlock / Connection lost")

    pipeline = RetrievalPipeline(
        retrieval_store=mock_store,
        embedding_provider=MockEmbeddingProvider(_dimension=8),
        audit_logger=mock_audit,
    )

    package, trace = pipeline.retrieve_with_trace(
        query_text="auth",
        mode="lexical",
    )

    assert len(package.items) == 1
    assert any("audit_logging_failed" in w for w in trace.warnings)
    assert package.items[0].snippet == "lexical result"


# ── 3. Truthful Health Check Status ───────────────────────────────────────────


def test_health_check_reports_error_when_db_fails() -> None:
    """When DB connection raises an error, overall status is 'error', not 'ok'."""
    tools = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=MagicMock(),
        embedding_provider=MockEmbeddingProvider(_dimension=8),
        connection_factory=MagicMock(side_effect=ConnectionError("Postgres unreachable")),
    )
    health = tools.health_check()
    assert health["status"] == "error"
    assert health["db_status"] == "error"
    assert "Postgres unreachable" in str(health.get("db_error"))


def test_health_check_reports_degraded_when_dimension_mismatches() -> None:
    """When embedding dimension does not match DB, overall status is 'degraded'."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (5, 20, 100)
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    tools = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=MagicMock(),
        embedding_provider=MockEmbeddingProvider(_dimension=1536),
        connection_factory=lambda: mock_conn,
    )

    with patch(
        "knowledge_fabric.db.dimension_guard.check_embedding_dimension",
        side_effect=ValueError("Dimension mismatch: 768 vs 1536"),
    ):
        health = tools.health_check()
        assert health["status"] == "degraded"
        assert health["dimension_check"] == "mismatch"


# ── 4. Token-Efficient Compact and Markdown Formats ───────────────────────────


def test_evidence_package_compact_mode_omits_duplicate_chunks() -> None:
    """compact=True removes redundant chunks[] array and duplicate text."""
    pkg = EvidencePackage(
        query_text="test query",
        items=[
            EvidenceItem(
                chunk_id=1,
                document_id=10,
                document_uri="docs/guide.md",
                chunk_index=0,
                snippet="Lorem ipsum dolor sit amet.",
                score=0.95,
                retrieval_sources=["lexical"],
                metadata={"heading_path": "Guide > Setup"},
                provenance_hash="a" * 64,
            )
        ],
    )

    full = pkg.to_dict(compact=False)
    assert "chunks" in full
    assert "items" in full

    compact = pkg.to_dict(compact=True)
    assert "chunks" not in compact
    assert "items" in compact
    assert compact["items"][0]["snippet"] == "Lorem ipsum dolor sit amet."
    assert compact["items"][0]["heading_path"] == "Guide > Setup"


def test_retrieve_evidence_markdown_format() -> None:
    """retrieve_evidence format='markdown' returns structured citations."""
    mock_pipeline = MagicMock()
    pkg = EvidencePackage(
        query_text="encryption",
        items=[
            EvidenceItem(
                chunk_id=1,
                document_id=10,
                document_uri="docs/crypto.md",
                chunk_index=0,
                snippet="AES-GCM is used for at-rest payload encryption.",
                score=0.98,
                metadata={"heading_path": "Security > Encryption"},
            )
        ],
    )
    mock_pipeline.retrieve_evidence.return_value = pkg

    tools = KnowledgeFabricMCPTools(
        retrieval_pipeline=mock_pipeline,
        retrieval_store=MagicMock(),
    )

    md_resp = tools.retrieve_evidence(query_text="encryption", format="markdown")
    assert md_resp["format"] == "markdown"
    assert "citations" in md_resp
    assert "docs/crypto.md (Security > Encryption)" in md_resp["citations"]
    assert "AES-GCM" in md_resp["citations"]


# ── 5. MCP Server Default Tenant Fallback ─────────────────────────────────────


def test_create_mcp_server_uses_default_tenant_when_omitted(monkeypatch) -> None:
    """When default_tenant is supplied, omitted tenant_id does not raise ValueError."""
    from types import ModuleType

    class _FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self, name: str):
            def _dec(fn):
                self.tools[name] = fn
                return fn
            return _dec

    fake_module = ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = _FakeFastMCP
    monkeypatch.setitem(__import__("sys").modules, "mcp.server.fastmcp", fake_module)
    monkeypatch.delenv("KF_DEFAULT_TENANT", raising=False)

    fake_tools = MagicMock()
    fake_tools.retrieve_evidence.return_value = {"ok": True}

    server = create_mcp_server(fake_tools, default_tenant="org-default")
    tools = server.tools

    # Does not raise ValueError
    resp = tools["retrieve_evidence"](query_text="test")
    assert resp == {"ok": True}
    assert fake_tools.retrieve_evidence.call_args[1]["tenant_id"] == "org-default"
