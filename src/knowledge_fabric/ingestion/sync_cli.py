# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""CLI for syncing content from SaaS source connectors into Knowledge Fabric."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from knowledge_fabric.chunking import DocumentChunkingService
from knowledge_fabric.config import load_settings
from knowledge_fabric.db import KnowledgeRepository, create_postgres_connection_factory
from knowledge_fabric.embeddings import build_embedding_provider
from knowledge_fabric.ingestion.models import Document, SourceFormat


def _build_adapter(args: argparse.Namespace) -> Any:
    connector = args.connector.lower()

    if connector == "confluence":
        from knowledge_fabric_adapters.connectors.confluence import ConfluenceSourceAdapter
        base_url = args.url or os.environ.get("CONFLUENCE_URL", "")
        if not base_url:
            raise ValueError("--url or CONFLUENCE_URL environment variable is required")
        spaces = [s.strip() for s in (args.space or os.environ.get("CONFLUENCE_SPACES", "")).split(",") if s.strip()]
        if not spaces:
            raise ValueError("--space or CONFLUENCE_SPACES is required")
        email = args.email or os.environ.get("CONFLUENCE_EMAIL", "")
        token = args.token or os.environ.get("CONFLUENCE_TOKEN", "")
        return ConfluenceSourceAdapter(base_url=base_url, space_keys=spaces, email=email, api_token=token)

    elif connector == "notion":
        from knowledge_fabric_adapters.connectors.notion import NotionSourceAdapter
        token = args.token or os.environ.get("NOTION_API_KEY", "")
        if not token:
            raise ValueError("--token or NOTION_API_KEY environment variable is required")
        db_ids = [d.strip() for d in (args.database_id or os.environ.get("NOTION_DATABASE_IDS", "")).split(",") if d.strip()]
        return NotionSourceAdapter(api_key=token, database_ids=db_ids or None)

    elif connector in ("google_drive", "gdrive"):
        from knowledge_fabric_adapters.connectors.google_drive import GoogleDriveSourceAdapter
        token = args.token or os.environ.get("GOOGLE_ACCESS_TOKEN", "")
        if not token:
            raise ValueError("--token or GOOGLE_ACCESS_TOKEN environment variable is required")
        folder_id = args.folder_id or os.environ.get("GDRIVE_FOLDER_ID", "root")
        return GoogleDriveSourceAdapter(access_token=token, folder_id=folder_id)

    elif connector == "jira":
        from knowledge_fabric_adapters.connectors.jira import JiraSourceAdapter
        base_url = args.url or os.environ.get("JIRA_URL", "")
        if not base_url:
            raise ValueError("--url or JIRA_URL environment variable is required")
        jql = args.jql or os.environ.get("JIRA_JQL", "statusCategory = Done ORDER BY updated DESC")
        email = args.email or os.environ.get("JIRA_EMAIL", "")
        token = args.token or os.environ.get("JIRA_TOKEN", "")
        return JiraSourceAdapter(base_url=base_url, jql=jql, email=email, api_token=token)

    else:
        raise ValueError(f"Unknown connector: {connector}. Supported: confluence, notion, gdrive, jira")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync SaaS connectors into Knowledge Fabric.")
    parser.add_argument("--connector", required=True, choices=["confluence", "notion", "gdrive", "google_drive", "jira"], help="Target connector name.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings.yaml.")
    parser.add_argument("--tenant", default="default", help="Tenant ID for partition isolation.")
    parser.add_argument("--url", help="Base URL for target SaaS platform.")
    parser.add_argument("--token", help="API token or integration secret.")
    parser.add_argument("--email", help="Account email if required for authentication.")
    parser.add_argument("--space", help="Confluence space keys (comma-separated).")
    parser.add_argument("--database-id", help="Notion database IDs (comma-separated).")
    parser.add_argument("--folder-id", help="Google Drive folder ID.")
    parser.add_argument("--jql", help="Jira JQL query string.")
    parser.add_argument("--embed", action="store_true", help="Compute and store vector embeddings.")
    parser.add_argument("--max-chunk-chars", type=int, default=1200, help="Max chunk size.")
    parser.add_argument("--chunk-overlap-chars", type=int, default=150, help="Overlap size.")

    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    connection_factory = create_postgres_connection_factory(settings.database)
    repository = KnowledgeRepository(connection_factory=connection_factory)
    chunking = DocumentChunkingService(max_chars=args.max_chunk_chars, overlap_chars=args.chunk_overlap_chars)

    embedding_provider = None
    if args.embed:
        embedding_provider = build_embedding_provider(
            provider_name=settings.embeddings.provider,
            dimension=settings.embeddings.dimension,
        )

    adapter = _build_adapter(args)
    resources = adapter.list_resources()

    summary = {
        "connector": args.connector,
        "tenant_id": args.tenant,
        "resources_discovered": len(resources),
        "documents_ingested": 0,
        "chunks_written": 0,
    }

    for res in resources:
        try:
            item = adapter.fetch_resource(res.resource_id)
        except Exception:
            continue

        raw_content = str(item.get("content", ""))
        doc = Document(
            source_uri=str(item.get("path", f"{args.connector}://{res.resource_id}")),
            source_format=SourceFormat.MARKDOWN,
            content_text=raw_content,
            metadata=item.get("metadata", {}),  # type: ignore
            title=res.name,
        )

        chunks = chunking.chunk_document(doc)
        embeddings = (
            embedding_provider.embed_texts([chunk.chunk_text for chunk in chunks])
            if embedding_provider
            else None
        )

        doc_id = repository.upsert_document(doc, tenant_id=args.tenant)
        count = repository.replace_chunks(
            document_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            tenant_id=args.tenant,
        )
        summary["documents_ingested"] += 1
        summary["chunks_written"] += count

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
