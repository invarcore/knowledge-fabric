"""Comprehensive test coverage expansion for Knowledge Fabric.

Covers:
- DocumentChunkingService edge cases & strategies
- SaaS sync_cli connector parsing and ingestion loop
- Embedding providers (Ollama, OpenAI, Cohere, Gemini, SentenceTransformer, Registry, Factory)
- MCP tools and FastMCP server entrypoints
- Admin UI HTTP server routes & handlers
- Qdrant retrieval store (vector, lexical, health check)
- Postgres retrieval store methods
- AuditLogger telemetry persistence
- Config environment variable overrides and _as_dict validation
- Ingestion CLI deletion and file resolution edge cases
- Retrieval pipeline degradation & fail-closed modes
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
import io
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request

import pytest

from knowledge_fabric.chunking import ChunkingConfig, DocumentChunkingService
from knowledge_fabric.config import Settings, _as_dict, load_settings
from knowledge_fabric.db.audit import AuditLogger
from knowledge_fabric.embeddings.providers import (
    CohereEmbeddingProvider,
    EmbeddingProviderRegistry,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_embedding_provider,
)
from knowledge_fabric.ingestion.cli import _build_parser, _resolve_input_files, main as ingestion_main
from knowledge_fabric.ingestion.models import Document, SourceFormat
from knowledge_fabric.ingestion.sync_cli import _build_adapter, main as sync_cli_main
from knowledge_fabric.mcp.server import _require_tenant, build_tools_from_settings, create_mcp_server, run_mcp_server
from knowledge_fabric.mcp.tools import KnowledgeFabricMCPTools, _get_mock_provider_type
from knowledge_fabric.retrieval.models import RetrievalHit
from knowledge_fabric.retrieval.pipeline import RetrievalPipeline
from knowledge_fabric.retrieval.postgres import PostgresRetrievalStore
from knowledge_fabric.retrieval.qdrant import QdrantRetrievalStore
from knowledge_fabric.ui.server import DashboardRequestHandler, _resolve_static_dir, main as ui_main, run_ui_server


# ============================================================================
# 1. DocumentChunkingService Edge Cases & Strategies
# ============================================================================

def test_chunking_config_validation():
    with pytest.raises(ValueError, match="max_chars must be greater than 0"):
        DocumentChunkingService(max_chars=0)

    with pytest.raises(ValueError, match="overlap_chars must be >= 0"):
        DocumentChunkingService(overlap_chars=-1)

    with pytest.raises(ValueError, match="overlap_chars must be smaller than max_chars"):
        DocumentChunkingService(max_chars=100, overlap_chars=100)

    cfg = ChunkingConfig(max_chars=500, overlap_chars=50)
    service = DocumentChunkingService(config=cfg)
    assert service.max_chars == 500
    assert service.overlap_chars == 50


def test_chunking_markdown_large_sections_and_fences():
    service = DocumentChunkingService(max_chars=150, overlap_chars=30)
    markdown_content = (
        "# Main Architecture\n\n"
        "Here is a short paragraph.\n\n"
        "```python\n"
        "# Very long code block that should be kept intact or split cleanly\n"
        "def compute_hash(val):\n"
        "    return hashlib.sha256(val.encode()).hexdigest()\n"
        "```\n\n"
        "## Section Two\n\n"
        "| Header 1 | Header 2 |\n"
        "| -------- | -------- |\n"
        "| Cell 1   | Cell 2   |\n\n"
        "Another very long paragraph with detailed prose explanation that will definitely exceed "
        "the 150 character limit imposed on max_chars for chunking service."
    )
    doc = Document(
        source_uri="file:///architecture.md",
        source_format=SourceFormat.MARKDOWN,
        content_text=markdown_content,
    )
    chunks = service.chunk_document(doc)
    assert len(chunks) >= 3
    assert any("compute_hash" in c.chunk_text for c in chunks)
    assert any(c.heading_level == 2 for c in chunks)


def test_chunking_page_aware():
    service = DocumentChunkingService(max_chars=100, overlap_chars=20)
    text = (
        "--- Page 1 ---\n"
        "First slide overview and introductory remarks.\n"
        "--- Page 2 ---\n"
        "Second slide with architecture diagrams and deeper details.\n"
        "\f"
        "Third slide concluding thoughts and next milestones."
    )
    doc = Document(
        source_uri="file:///presentation.pptx",
        source_format=SourceFormat.PPTX,
        content_text=text,
    )
    chunks = service.chunk_document(doc)
    assert len(chunks) >= 3
    pages = [c.metadata.get("page_number") for c in chunks]
    assert 1 in pages or 2 in pages


def test_chunking_fallback_empty_and_long():
    service = DocumentChunkingService(max_chars=50, overlap_chars=10)
    doc_empty = Document(
        source_uri="file:///empty.txt",
        source_format=SourceFormat.TXT,
        content_text="",
    )
    chunks_empty = service.chunk_document(doc_empty)
    assert chunks_empty == []

    doc_long = Document(
        source_uri="file:///long.txt",
        source_format=SourceFormat.TXT,
        content_text="A" * 120,
    )
    chunks_long = service.chunk_document(doc_long)
    assert len(chunks_long) >= 2


# ============================================================================
# 2. SaaS sync_cli Connector Parsing and Ingestion Loop
# ============================================================================

def test_sync_cli_build_adapter_confluence():
    args_missing_url = argparse.Namespace(connector="confluence", url="", space="ENG", email="a@b.com", token="tok")
    args_missing_space = argparse.Namespace(connector="confluence", url="https://corp.atlassian.net", space="", email="a@b.com", token="tok")
    args_valid = argparse.Namespace(connector="confluence", url="https://corp.atlassian.net", space="ENG,PROD", email="a@b.com", token="tok")

    mock_mod = MagicMock()
    with patch.dict("sys.modules", {
        "knowledge_fabric_adapters": MagicMock(),
        "knowledge_fabric_adapters.connectors": MagicMock(),
        "knowledge_fabric_adapters.connectors.confluence": mock_mod,
    }):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="CONFLUENCE_URL"):
                _build_adapter(args_missing_url)

            with pytest.raises(ValueError, match="CONFLUENCE_SPACES"):
                _build_adapter(args_missing_space)

        adapter = _build_adapter(args_valid)
        assert adapter is not None


def test_sync_cli_build_adapter_notion():
    args_missing_token = argparse.Namespace(connector="notion", token="", database_id="db1")
    args_valid = argparse.Namespace(connector="notion", token="secret", database_id="db1,db2")

    mock_mod = MagicMock()
    with patch.dict("sys.modules", {
        "knowledge_fabric_adapters": MagicMock(),
        "knowledge_fabric_adapters.connectors": MagicMock(),
        "knowledge_fabric_adapters.connectors.notion": mock_mod,
    }):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="NOTION_API_KEY"):
                _build_adapter(args_missing_token)

        adapter = _build_adapter(args_valid)
        assert adapter is not None


def test_sync_cli_build_adapter_gdrive():
    args_missing_token = argparse.Namespace(connector="gdrive", token="", folder_id="root")
    args_valid = argparse.Namespace(connector="google_drive", token="oauth-tok", folder_id="folder123")

    mock_mod = MagicMock()
    with patch.dict("sys.modules", {
        "knowledge_fabric_adapters": MagicMock(),
        "knowledge_fabric_adapters.connectors": MagicMock(),
        "knowledge_fabric_adapters.connectors.google_drive": mock_mod,
    }):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_ACCESS_TOKEN"):
                _build_adapter(args_missing_token)

        adapter = _build_adapter(args_valid)
        assert adapter is not None


def test_sync_cli_build_adapter_jira():
    args_missing_url = argparse.Namespace(connector="jira", url="", jql="", email="a@b.com", token="tok")
    args_valid = argparse.Namespace(connector="jira", url="https://jira.corp.com", jql="status=Done", email="a@b.com", token="tok")

    mock_mod = MagicMock()
    with patch.dict("sys.modules", {
        "knowledge_fabric_adapters": MagicMock(),
        "knowledge_fabric_adapters.connectors": MagicMock(),
        "knowledge_fabric_adapters.connectors.jira": mock_mod,
    }):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="JIRA_URL"):
                _build_adapter(args_missing_url)

        adapter = _build_adapter(args_valid)
        assert adapter is not None


def test_sync_cli_build_adapter_unknown():
    args_unknown = argparse.Namespace(connector="slack")
    with pytest.raises(ValueError, match="Unknown connector"):
        _build_adapter(args_unknown)


def test_sync_cli_main_execution():
    fake_resource_ok = MagicMock(resource_id="res_1", name="Doc One")
    fake_resource_err = MagicMock(resource_id="res_2", name="Doc Err")

    mock_adapter = MagicMock()
    mock_adapter.list_resources.return_value = [fake_resource_ok, fake_resource_err]
    mock_adapter.fetch_resource.side_effect = [
        {"content": "# Doc One\nContent body here.", "path": "confluence://res_1", "metadata": {"author": "alice"}},
        RuntimeError("Resource fetch timeout"),
    ]

    mock_repo = MagicMock()
    mock_repo.upsert_document.return_value = 101
    mock_repo.replace_chunks.return_value = 2

    with patch("knowledge_fabric.ingestion.sync_cli._build_adapter", return_value=mock_adapter), \
         patch("knowledge_fabric.ingestion.sync_cli.load_settings"), \
         patch("knowledge_fabric.ingestion.sync_cli.create_postgres_connection_factory"), \
         patch("knowledge_fabric.ingestion.sync_cli.KnowledgeRepository", return_value=mock_repo), \
         patch("knowledge_fabric.ingestion.sync_cli.build_embedding_provider") as mock_emb_fn:

        mock_emb = MagicMock()
        mock_emb.embed_texts.return_value = [[0.1, 0.2]]
        mock_emb_fn.return_value = mock_emb

        exit_code = sync_cli_main([
            "--connector", "confluence",
            "--url", "https://mock.atlassian.net",
            "--space", "ENG",
            "--email", "test@test.com",
            "--token", "secret",
            "--embed",
            "--tenant", "tenant-test",
        ])
        assert exit_code == 0
        assert mock_repo.upsert_document.call_count == 1
        assert mock_repo.replace_chunks.call_count == 1


# ============================================================================
# 3. Embedding Providers & Registry
# ============================================================================

def test_ollama_embedding_provider():
    with patch.dict(os.environ, {"OLLAMA_EMBED_MODEL": "nomic-embed-text", "OLLAMA_BASE_URL": "http://localhost:11434"}):
        provider = OllamaEmbeddingProvider.from_env()
        assert provider.dimension == 768

    mock_response = io.BytesIO(json.dumps({"embedding": [0.1] * 768}).encode("utf-8"))
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result = provider.embed_texts(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 768


def test_openai_embedding_provider():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(OSError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingProvider.from_env()

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key", "OPENAI_EMBED_MODEL": "text-embedding-3-small"}):
        provider = OpenAIEmbeddingProvider.from_env()
        assert provider.dimension == 1536

        mock_response = io.BytesIO(json.dumps({"data": [{"embedding": [0.2] * 1536}]}).encode("utf-8"))
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            res = provider.embed_texts(["test text"])
            assert len(res) == 1
            assert len(res[0]) == 1536


def test_cohere_embedding_provider():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(OSError, match="COHERE_API_KEY"):
            CohereEmbeddingProvider.from_env()

    with patch.dict(os.environ, {"COHERE_API_KEY": "coh-mock", "COHERE_EMBED_MODEL": "embed-english-v3.0"}):
        provider = CohereEmbeddingProvider.from_env()
        assert provider.dimension == 1024

        mock_response = io.BytesIO(json.dumps({"embeddings": {"float": [[0.3] * 1024]}}).encode("utf-8"))
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            res = provider.embed_texts(["test text"])
            assert len(res) == 1
            assert len(res[0]) == 1024


def test_gemini_embedding_provider():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(OSError, match="GEMINI_API_KEY"):
            GeminiEmbeddingProvider.from_env()

    with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-mock", "GEMINI_EMBED_MODEL": "text-embedding-004"}):
        provider = GeminiEmbeddingProvider.from_env()
        assert provider.dimension == 768
        assert provider.embed_texts([]) == []

        mock_response = io.BytesIO(json.dumps({"embeddings": [{"values": [0.4] * 768}]}).encode("utf-8"))
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            res = provider.embed_texts(["hello gemini"])
            assert len(res) == 1
            assert len(res[0]) == 768


def test_sentence_transformer_embedding_provider():
    with patch.dict(os.environ, {"SENTENCE_TRANSFORMER_MODEL": "all-MiniLM-L6-v2"}):
        provider = SentenceTransformerEmbeddingProvider.from_env()
        assert provider.dimension == 384
        assert provider.embed_texts([]) == []

        # Test ImportError
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers is not installed"):
                provider._ensure_loaded()

        # Test successful load and encode
        mock_st = MagicMock()
        mock_vec = MagicMock()
        mock_vec.tolist.return_value = [0.5] * 384
        mock_st.encode.return_value = [mock_vec]
        mock_st.get_sentence_embedding_dimension.return_value = 384
        mock_mod = MagicMock(SentenceTransformer=MagicMock(return_value=mock_st))
        with patch.dict("sys.modules", {"sentence_transformers": mock_mod}):
            provider._model = None
            res = provider.embed_texts(["sentence test"])
            assert len(res) == 1
            assert len(res[0]) == 384


def test_embedding_provider_registry_and_factory():
    assert EmbeddingProviderRegistry.is_registered("mock")
    assert "mock" in EmbeddingProviderRegistry.list_providers()

    with pytest.raises(KeyError, match="Unknown embedding provider"):
        EmbeddingProviderRegistry.get("nonexistent_provider")

    # Factory with TypeError handling
    EmbeddingProviderRegistry.register("const_mock", lambda: MockEmbeddingProvider(_dimension=16))
    inst = EmbeddingProviderRegistry.get("const_mock", dimension=32)
    assert inst.dimension == 16

    # build_embedding_provider
    prov1 = build_embedding_provider("mock", dimension=64)
    assert prov1.dimension == 64

    prov2 = build_embedding_provider("unknown_fallback", dimension=128)
    assert prov2.dimension == 128


# ============================================================================
# 4. MCP Tools & FastMCP Server
# ============================================================================

def test_mcp_tools_health_check_variations():
    # 1. No connection factory
    tools_no_conn = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=MagicMock(),
        embedding_provider=MockEmbeddingProvider(_dimension=8),
        connection_factory=None,
    )
    health1 = tools_no_conn.health_check()
    assert health1["status"] == "ok"
    assert health1["db_status"] == "unknown"

    # 2. Connection factory connected with dimension match
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (10, 50, 5)
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    tools_conn = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=MagicMock(),
        embedding_provider=MockEmbeddingProvider(_dimension=8),
        connection_factory=lambda: mock_conn,
    )
    with patch("knowledge_fabric.db.dimension_guard.check_embedding_dimension"):
        health2 = tools_conn.health_check()
        assert health2["db_status"] == "connected"
        assert health2["document_count"] == 10
        assert health2["dimension_check"] == "match"

    # 3. Connection factory error
    tools_err = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=MagicMock(),
        embedding_provider=MockEmbeddingProvider(_dimension=8),
        connection_factory=MagicMock(side_effect=RuntimeError("DB unreachable")),
    )
    health3 = tools_err.health_check()
    assert health3["db_status"] == "error"
    assert "DB unreachable" in str(health3.get("db_error"))


def test_mcp_tools_list_sources_and_store_fallbacks():
    # list_sources without store
    tools_no_store = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=None,
    )
    assert "No retrieval store configured" in str(tools_no_store.list_sources()["error"])

    mock_store = MagicMock()
    mock_store.list_sources_for_tenant.side_effect = RuntimeError("Query error")
    tools_with_store = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=mock_store,
    )
    assert "Query error" in str(tools_with_store.list_sources()["error"])

    # store without optional methods
    bare_store = object()
    tools_bare = KnowledgeFabricMCPTools(
        retrieval_pipeline=MagicMock(),
        retrieval_store=bare_store,
    )
    assert "error" in tools_bare.get_index_status()
    assert tools_bare.get_evidence(1) is None
    assert tools_bare.check_consistency()["status"] == "unsupported_by_backend"


def test_mcp_server_helpers_and_tool_registration():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="tenant_id is required"):
            _require_tenant(None)

    with patch.dict(os.environ, {"KF_DEFAULT_TENANT": "env-tenant"}):
        assert _require_tenant(None) == "env-tenant"

    # build_tools_from_settings with qdrant backend
    settings = load_settings("config/settings.yaml")
    settings.retrieval.backend = "qdrant"
    tools = build_tools_from_settings(settings)
    assert tools is not None

    # Server registration and tool calls
    server = create_mcp_server(tools)
    assert server is not None

    # run_mcp_server
    with patch("knowledge_fabric.mcp.server.create_mcp_server") as mock_create:
        mock_srv = MagicMock()
        mock_create.return_value = mock_srv
        run_mcp_server("config/settings.yaml")
        mock_srv.run.assert_called_once()

    # _get_mock_provider_type
    assert _get_mock_provider_type() is MockEmbeddingProvider


# ============================================================================
# 5. Admin UI HTTP Server Routes & Handlers
# ============================================================================

def test_ui_server_static_dir_and_routes():
    static_dir = _resolve_static_dir()
    assert static_dir.exists()

    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # GET /
        with urllib.request.urlopen(f"{base_url}/") as resp:
            assert resp.status == HTTPStatus.OK
            assert b"<!DOCTYPE html>" in resp.read() or b"<html" in resp.read()

        # GET /api/health
        with urllib.request.urlopen(f"{base_url}/api/health") as resp:
            assert resp.status == HTTPStatus.OK
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"

        # GET /api/approvals
        with urllib.request.urlopen(f"{base_url}/api/approvals") as resp:
            assert resp.status == HTTPStatus.OK
            data = json.loads(resp.read().decode())
            assert isinstance(data, list)

        # GET /api/audit
        with urllib.request.urlopen(f"{base_url}/api/audit") as resp:
            assert resp.status == HTTPStatus.OK
            data = json.loads(resp.read().decode())
            assert isinstance(data, list)

        # GET /api/policies
        with urllib.request.urlopen(f"{base_url}/api/policies") as resp:
            assert resp.status == HTTPStatus.OK
            data = json.loads(resp.read().decode())
            assert "rules" in data

        # POST /api/approvals/appr_123/decision
        post_data = json.dumps({"decision": "approved", "comment": "all good", "reviewer": "alice@corp.com"}).encode()
        req = urllib.request.Request(f"{base_url}/api/approvals/appr_123/decision", data=post_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == HTTPStatus.OK
            data = json.loads(resp.read().decode())
            assert data["approval_id"] == "appr_123"
            assert "signature" in data

        # POST /api/retrieval/query - valid
        post_query = json.dumps({"query": "firewall rules", "tenant_id": "corp_tenant"}).encode()
        req = urllib.request.Request(f"{base_url}/api/retrieval/query", data=post_query, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == HTTPStatus.OK
            data = json.loads(resp.read().decode())
            assert len(data["hits"]) > 0

        # POST /api/retrieval/query - invalid tenant_id
        post_bad_tenant = json.dumps({"query": "test", "tenant_id": "bad/tenant!*#"}).encode()
        req = urllib.request.Request(f"{base_url}/api/retrieval/query", data=post_bad_tenant, headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == HTTPStatus.BAD_REQUEST

        # POST /api/policies/test - syntax error
        post_bad_action = json.dumps({"action": "invalid action spaces!"}).encode()
        req = urllib.request.Request(f"{base_url}/api/policies/test", data=post_bad_action, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["decision"] == "deny"
            assert "syntax validation" in data["reason"]

        # POST /api/policies/test - db_* destructive
        post_db = json.dumps({"action": "db_drop_table"}).encode()
        req = urllib.request.Request(f"{base_url}/api/policies/test", data=post_db, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["decision"] == "deny"

        # POST /api/policies/test - ticket_* requires approval
        post_ticket = json.dumps({"action": "ticket_create_incident"}).encode()
        req = urllib.request.Request(f"{base_url}/api/policies/test", data=post_ticket, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["decision"] == "requires_approval"

        # POST /api/policies/test - allow
        post_allow = json.dumps({"action": "inspect_metrics"}).encode()
        req = urllib.request.Request(f"{base_url}/api/policies/test", data=post_allow, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["decision"] == "allow"

        # POST /api/unknown 404
        req = urllib.request.Request(f"{base_url}/api/unknown", data=b"{}", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == HTTPStatus.NOT_FOUND

    finally:
        server.shutdown()
        server.server_close()


def test_ui_server_main_and_cli():
    with patch("knowledge_fabric.ui.server.run_ui_server") as mock_run:
        exit_code = ui_main(["--host", "127.0.0.1", "--port", "8888"])
        assert exit_code == 0
        mock_run.assert_called_once_with(host="127.0.0.1", port=8888)


# ============================================================================
# 6. Qdrant Retrieval Store
# ============================================================================

def test_qdrant_retrieval_store():
    store = QdrantRetrievalStore(url="http://localhost:6333", collection="test_col", api_key="qdrant-secret")
    headers = store._headers()
    assert headers["api-key"] == "qdrant-secret"

    # vector search success
    search_resp = {
        "result": [
            {
                "id": 42,
                "score": 0.95,
                "payload": {
                    "document_uri": "doc://1",
                    "source_type": "markdown",
                    "chunk_text": "Sample qdrant text",
                    "metadata": {"section": "intro"},
                },
            }
        ]
    }
    with patch.object(store, "_request", return_value=search_resp):
        hits = store.vector_search([0.1, 0.2], top_k=5, filters={"section": "intro"}, tenant_id="tenant-1")
        assert len(hits) == 1
        assert hits[0].chunk_id == 42
        assert hits[0].score == 0.95

    # vector search exception fallback
    with patch.object(store, "_request", side_effect=RuntimeError("Qdrant connection refused")):
        hits_err = store.vector_search([0.1, 0.2])
        assert hits_err == []

    # lexical search scroll success
    scroll_resp = {
        "result": {
            "points": [
                {
                    "id": 99,
                    "payload": {
                        "document_uri": "doc://2",
                        "source_type": "text",
                        "chunk_text": "Lexical scroll hit",
                    },
                }
            ]
        }
    }
    with patch.object(store, "_request", return_value=scroll_resp):
        lex_hits = store.lexical_search("scroll hit", top_k=5, filters={"key": "val"}, tenant_id="tenant-1")
        assert len(lex_hits) == 1
        assert lex_hits[0].chunk_id == 99

    # lexical search exception fallback
    with patch.object(store, "_request", side_effect=RuntimeError("Scroll error")):
        assert store.lexical_search("text") == []

    # health check success & error
    with patch.object(store, "_request", return_value={"result": {"status": "green", "vectors_count": 1000}}):
        h_ok = store.health_check()
        assert h_ok["status"] == "connected"
        assert h_ok["vectors_count"] == 1000

    with patch.object(store, "_request", side_effect=RuntimeError("Timeout")):
        h_err = store.health_check()
        assert h_err["status"] == "error"


# ============================================================================
# 7. Postgres Retrieval Store
# ============================================================================

def test_postgres_retrieval_store_methods():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    store = PostgresRetrievalStore(connection_factory=lambda: mock_conn)

    # get_document validations
    with pytest.raises(ValueError, match="document_id or source_uri must be provided"):
        store.get_document()

    # get_document by id not found
    mock_cursor.fetchall.return_value = []
    assert store.get_document(document_id=123, tenant_id="t1") is None

    # get_document by uri found
    mock_cursor.fetchall.return_value = [
        (1, "file:///doc.md", "markdown", "Title Doc", {"k": "v"}, "Content text", "2026-01-01", "2026-01-02")
    ]
    doc = store.get_document(source_uri="file:///doc.md", tenant_id="t1")
    assert doc is not None
    assert doc["id"] == 1
    assert doc["title"] == "Title Doc"

    # list_sources_for_tenant
    mock_cursor.fetchall.return_value = [("markdown", 10), ("pdf", 5)]
    sources = store.list_sources_for_tenant(tenant_id="t1")
    assert len(sources) == 2

    # delete_document and purge_tenant delegating to repo
    with patch("knowledge_fabric.db.repository.KnowledgeRepository.delete_document", return_value=True) as mock_del:
        assert store.delete_document(document_id=1, tenant_id="t1") is True
        mock_del.assert_called_once()

    with patch("knowledge_fabric.db.repository.KnowledgeRepository.purge_tenant", return_value={"purged": 10}) as mock_purge:
        res = store.purge_tenant(tenant_id="t1")
        assert res["purged"] == 10
        mock_purge.assert_called_once()


# ============================================================================
# 8. Audit Logger Telemetry
# ============================================================================

def test_audit_logger_log_retrieval():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    logger = AuditLogger(connection_factory=lambda: mock_conn)
    logger.log_retrieval(
        query_text="emergency access procedure",
        top_k=5,
        retrieval_mode="hybrid",
        result_count=3,
        latency_ms=12,
        trace_id="trace_test_123",
        tenant_id="tenant_ops",
        details={"model": "mock", "cached": False},
    )

    assert mock_cursor.execute.call_count == 3
    mock_conn.commit.assert_called_once()


# ============================================================================
# 9. Config Overrides and Helpers
# ============================================================================

def test_config_env_overrides_and_as_dict(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist.yaml"
    settings = load_settings(str(nonexistent))
    assert settings.app.name == "knowledge-fabric"

    # DATABASE_URL override
    env_vars = {
        "DATABASE_URL": "postgres://custom_user:custom_pass@custom_host:5433/custom_db",
        "TIKA_ENDPOINT": "http://tika-server:9998",
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_DIMENSION": "768",
        "RETRIEVAL_BACKEND": "qdrant",
        "RERANKER": "bge",
    }
    with patch.dict(os.environ, env_vars):
        settings_override = load_settings(str(nonexistent))
        assert settings_override.database.host == "custom_host"
        assert settings_override.database.port == 5433
        assert settings_override.database.name == "custom_db"
        assert settings_override.database.user == "custom_user"
        assert settings_override.database.password == "custom_pass"
        assert settings_override.tika.endpoint == "http://tika-server:9998"
        assert settings_override.embeddings.provider == "gemini"
        assert settings_override.embeddings.dimension == 768
        assert settings_override.retrieval.backend == "qdrant"
        assert settings_override.reranking.provider == "bge"

    # _as_dict error
    with pytest.raises(ValueError, match="Expected mapping for 'database'"):
        _as_dict({"database": "not_a_dict"}, "database")


# ============================================================================
# 10. Ingestion CLI Deletion and File Resolution Edge Cases
# ============================================================================

def test_ingestion_cli_edge_cases(tmp_path: Path):
    # Nonexistent path
    with pytest.raises(FileNotFoundError, match="Input path does not exist"):
        _resolve_input_files(str(tmp_path / "missing_dir"), recursive=False)

    # Unsupported single file
    unsupported_file = tmp_path / "data.bin"
    unsupported_file.write_bytes(b"\x00\x01\x02")
    assert _resolve_input_files(str(unsupported_file), recursive=False) == []

    # Supported file
    md_file = tmp_path / "guide.md"
    md_file.write_text("# Title\nBody.", encoding="utf-8")
    assert _resolve_input_files(str(md_file), recursive=False) == [md_file]

    # Directory with recursive=False vs True
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    sub_md = sub_dir / "nested.md"
    sub_md.write_text("# Nested\nContent.", encoding="utf-8")

    files_non_rec = _resolve_input_files(str(tmp_path), recursive=False)
    assert md_file in files_non_rec
    assert sub_md not in files_non_rec

    files_rec = _resolve_input_files(str(tmp_path), recursive=True)
    assert md_file in files_rec
    assert sub_md in files_rec

    # Test main() with missing path and no lifecycle options
    exit_code = ingestion_main(["--tenant", "tenant1"])
    assert exit_code == 1

    # Test main() with --delete-source-type
    with patch("knowledge_fabric.ingestion.cli.load_settings"), \
         patch("knowledge_fabric.ingestion.cli.create_postgres_connection_factory"), \
         patch("knowledge_fabric.ingestion.cli.KnowledgeRepository") as mock_repo_cls:
        mock_repo = MagicMock()
        mock_repo.delete_by_source.return_value = 5
        mock_repo_cls.return_value = mock_repo
        code = ingestion_main(["--delete-source-type", "confluence", "--tenant", "tenant1"])
        assert code == 0
        mock_repo.delete_by_source.assert_called_once_with(source_type="confluence", tenant_id="tenant1")


# ============================================================================
# 11. Retrieval Pipeline Degradation & Edge Cases
# ============================================================================

def test_pipeline_degraded_and_fail_closed_legs():
    mock_store = MagicMock()
    mock_emb = MockEmbeddingProvider(_dimension=8)

    pipeline = RetrievalPipeline(
        retrieval_store=mock_store,
        embedding_provider=mock_emb,
    )

    # 1. Lexical leg fails with fail_closed=True
    mock_store.lexical_search.side_effect = RuntimeError("Lexical DB down")
    with pytest.raises(RuntimeError, match="Lexical DB down"):
        pipeline.retrieve_evidence(query_text="test query", mode="lexical", fail_closed=True)

    # 2. Degraded vector: lexical fails, vector succeeds in hybrid mode
    mock_store.lexical_search.side_effect = RuntimeError("Lexical timeout")
    mock_hit = RetrievalHit(chunk_id=1, document_uri="doc://1", source_type="vector", snippet="hit", score=0.9)
    mock_store.vector_search.side_effect = None
    mock_store.vector_search.return_value = [mock_hit]

    res_degraded_vec = pipeline.retrieve_evidence(query_text="test query", mode="hybrid", fail_closed=False)
    assert res_degraded_vec.retrieval_summary.get("is_degraded") is True
    assert res_degraded_vec.retrieval_summary.get("strategy") == "degraded_vector"
    assert len(res_degraded_vec.items) == 1

    # 3. Both legs fail in hybrid mode
    mock_store.vector_search.side_effect = RuntimeError("Vector timeout")
    res_both_fail = pipeline.retrieve_evidence(query_text="test query", mode="hybrid", fail_closed=False)
    assert res_both_fail.retrieval_summary.get("is_degraded") is True
    assert res_both_fail.retrieval_summary.get("strategy") == "failed"
    assert len(res_both_fail.items) == 0
