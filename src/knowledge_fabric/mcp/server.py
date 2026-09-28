"""MCP server entrypoint for Knowledge Fabric tools."""

from __future__ import annotations

import logging
import os
from typing import Any

from knowledge_fabric.config import Settings, load_settings
from knowledge_fabric.db import AuditLogger, create_postgres_connection_factory
from knowledge_fabric.embeddings import build_embedding_provider
from knowledge_fabric.mcp.tools import KnowledgeFabricMCPTools
from knowledge_fabric.reranking import build_reranker
from knowledge_fabric.retrieval.pipeline import RetrievalPipeline, validate_tenant_id
from knowledge_fabric.retrieval.postgres import PostgresRetrievalStore

logger = logging.getLogger(__name__)


def build_tools_from_settings(settings: Settings) -> KnowledgeFabricMCPTools:
    """Build MCP tools with configured retrieval dependencies."""
    if settings.retrieval.backend == "qdrant":
        logger.warning(
            "retrieval.backend=qdrant is configured but Qdrant is not yet a complete "
            "ingestion and MCP path. Falling back to PostgreSQL. "
            "See docs/architecture.md for the verified reference path."
        )
    connection_factory = create_postgres_connection_factory(settings.database)
    retrieval_store = PostgresRetrievalStore(connection_factory=connection_factory)
    embedding_provider = build_embedding_provider(
        provider_name=settings.embeddings.provider,
        dimension=settings.embeddings.dimension,
    )
    # Honour the reranking.provider from settings.yaml (env var RERANKER overrides).
    reranker = build_reranker(settings.reranking.provider)
    audit_logger = AuditLogger(connection_factory=connection_factory)
    pipeline = RetrievalPipeline(
        retrieval_store=retrieval_store,
        embedding_provider=embedding_provider,
        audit_logger=audit_logger,
        reranker=reranker,
        rrf_k=settings.retrieval.rrf_k,
        lexical_weight=settings.retrieval.lexical_weight,
        vector_weight=settings.retrieval.vector_weight,
    )
    return KnowledgeFabricMCPTools(
        retrieval_pipeline=pipeline,
        retrieval_store=retrieval_store,
        embedding_provider=embedding_provider,
        connection_factory=connection_factory,
    )



def _require_tenant(tenant_id: str | None) -> str:
    effective_tenant = validate_tenant_id(tenant_id or os.environ.get("KF_DEFAULT_TENANT") or None)
    if not effective_tenant:
        raise ValueError(
            "tenant_id is required; specify tenant_id parameter or configure KF_DEFAULT_TENANT"
        )
    return effective_tenant


def create_mcp_server(tools: KnowledgeFabricMCPTools) -> Any:
    """Create MCP server with retrieval tool registration."""
    try:
        from mcp.server.fastmcp import FastMCP
    except (ImportError, ModuleNotFoundError):
        from mcp.server.mcpserver import MCPServer as FastMCP

    server = FastMCP("knowledge-fabric")

    @server.tool(name="health_check")
    def health_check() -> dict[str, object]:
        """Check server health and return runtime configuration.

        Returns embedding provider name and dimension, DB connectivity status,
        and approximate document/chunk counts. Useful for AI agents to orient
        themselves before querying.
        """
        return tools.health_check()

    @server.tool(name="list_sources")
    def list_sources(
        tenant_id: str | None = None,
    ) -> dict[str, object]:
        """List ingested source types and document counts for a tenant.

        Returns distinct source_type values with document counts so an AI agent
        can discover what content domains are available before issuing a query.
        Use source_type as a filter in retrieve_evidence to scope retrieval.
        Scoped to tenant_id when provided.
        """
        effective_tenant = _require_tenant(tenant_id)
        return tools.list_sources(tenant_id=effective_tenant)

    @server.tool(name="retrieve_evidence")
    def retrieve_evidence(
        query_text: str,
        top_k: int = 10,
        source_type: str | None = None,
        trace_id: str | None = None,
        tenant_id: str | None = None,
        mode: str = "hybrid",
    ) -> dict[str, object]:
        """Execute evidence retrieval across indexed content.

        Parameters:
        - query_text: The natural-language or keyword search query.
        - top_k: Maximum number of ranked evidence chunks to return (default: 10).
        - source_type: Optional filter by source kind (e.g., 'markdown', 'confluence').
        - trace_id: Optional client-supplied correlation ID for audit tracing.
        - tenant_id: Tenant namespace identifier (scopes search to tenant data).
        - mode: Retrieval strategy: 'hybrid' (lexical + vector RRF), 'lexical' (full-text only),
          or 'vector' (semantic embeddings only).
        """
        effective_tenant = _require_tenant(tenant_id)
        return tools.retrieve_evidence(
            query_text=query_text,
            top_k=top_k,
            source_type=source_type,
            trace_id=trace_id,
            tenant_id=effective_tenant,
            mode=mode,
        )

    @server.tool(name="get_document")
    def get_document(
        document_id: int | None = None,
        source_uri: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, object] | None:
        effective_tenant = _require_tenant(tenant_id)
        return tools.get_document(
            document_id=document_id,
            source_uri=source_uri,
            tenant_id=effective_tenant,
        )

    @server.tool(name="explain_retrieval")
    def explain_retrieval(
        query_text: str,
        top_k: int = 10,
        source_type: str | None = None,
        tenant_id: str | None = None,
        mode: str = "hybrid",
    ) -> dict[str, object]:
        """Return diagnostic tracing and candidate scoring explanations for a query.

        Explains candidate counts, per-leg latencies, fusion scores, and degradation
        status without persisting audit logs.
        """
        effective_tenant = _require_tenant(tenant_id)
        return tools.explain_retrieval(
            query_text=query_text,
            top_k=top_k,
            source_type=source_type,
            tenant_id=effective_tenant,
            mode=mode,
        )

    @server.tool(name="get_index_status")
    def get_index_status(
        tenant_id: str | None = None,
    ) -> dict[str, object]:
        """Return diagnostic index breakdown: document/chunk counts, sources, and storage status.

        Scoped to tenant_id when provided.
        """
        effective_tenant = _require_tenant(tenant_id)
        return tools.get_index_status(tenant_id=effective_tenant)

    @server.tool(name="get_evidence")
    def get_evidence(
        chunk_id: int,
        tenant_id: str | None = None,
    ) -> dict[str, object] | None:
        """Fetch a specific evidence chunk by its integer chunk ID, scoped to tenant."""
        effective_tenant = _require_tenant(tenant_id)
        return tools.get_evidence(chunk_id=chunk_id, tenant_id=effective_tenant)

    @server.tool(name="check_consistency")
    def check_consistency(
        tenant_id: str | None = None,
    ) -> dict[str, object]:
        """Run database consistency audits (detecting orphaned chunks, empty documents, null tenants)."""
        effective_tenant = _require_tenant(tenant_id)
        return tools.check_consistency(tenant_id=effective_tenant)

    return server


def run_mcp_server(settings_path: str = "config/settings.yaml") -> None:
    """Run the MCP server using stdio transport."""
    settings = load_settings(settings_path)
    tools = build_tools_from_settings(settings)
    server = create_mcp_server(tools)
    server.run()


if __name__ == "__main__":
    run_mcp_server()

