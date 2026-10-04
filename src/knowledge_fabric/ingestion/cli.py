# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""CLI for persistent document ingestion and chunk storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_fabric.chunking import DocumentChunkingService
from knowledge_fabric.config import load_settings
from knowledge_fabric.db import KnowledgeRepository, create_postgres_connection_factory
from knowledge_fabric.embeddings import build_embedding_provider
from knowledge_fabric.ingestion import DocumentIngestionService, TikaClient

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx", ".pptx"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    connection_factory = create_postgres_connection_factory(settings.database)
    repository = KnowledgeRepository(connection_factory=connection_factory)
    tenant_id = args.tenant

    # --- Observability, Diagnostics & Operations Handlers ---
    if args.index_status:
        status_info = repository.get_index_status(tenant_id=tenant_id if tenant_id != "default" else None)
        print(json.dumps(status_info, indent=2))
        return 0

    if args.check_consistency:
        report = repository.check_consistency(tenant_id=tenant_id if tenant_id != "default" else None)
        print(json.dumps(report, indent=2))
        return 0 if report.get("is_healthy", False) else 1

    if args.reembed:
        embedding_provider = build_embedding_provider(
            provider_name=settings.embeddings.provider,
            dimension=settings.embeddings.dimension,
        )
        reembed_summary = repository.reembed_chunks(
            embedding_provider=embedding_provider,
            tenant_id=tenant_id if tenant_id != "default" else None,
        )
        print(json.dumps(reembed_summary, indent=2))
        return 0

    # --- Deletion / Purge Lifecycle Handlers ---
    if args.purge_tenant:
        if not args.confirm:
            print(f"Error: Must specify --confirm to permanently purge tenant '{args.purge_tenant}'.")
            return 1
        summary = repository.purge_tenant(args.purge_tenant)
        print(json.dumps(summary, indent=2))
        return 0

    if args.delete_document_id is not None or args.delete_source_uri is not None:
        deleted = repository.delete_document(
            document_id=args.delete_document_id,
            source_uri=args.delete_source_uri,
            tenant_id=tenant_id,
        )
        print(json.dumps({
            "deleted": deleted,
            "document_id": args.delete_document_id,
            "source_uri": args.delete_source_uri,
            "tenant_id": tenant_id,
        }, indent=2))
        return 0

    if args.delete_source_type:
        count = repository.delete_by_source(
            source_type=args.delete_source_type,
            tenant_id=tenant_id,
        )
        print(json.dumps({
            "deleted_count": count,
            "source_type": args.delete_source_type,
            "tenant_id": tenant_id,
        }, indent=2))
        return 0

    # --- Standard Ingestion Path ---
    if not args.path:
        print("Error: --path is required when not executing diagnostic, re-embed, deletion, or purge operations.")
        return 1

    import inspect
    max_file_size_bytes = (args.max_file_size_mb * 1024 * 1024) if getattr(args, "max_file_size_mb", None) else (25 * 1024 * 1024)
    ingestion_kwargs = {
        "tika_client": TikaClient(
            endpoint=settings.tika.endpoint,
            timeout_seconds=settings.tika.request_timeout_seconds,
        )
    }
    sig = inspect.signature(DocumentIngestionService.__init__)
    if "max_file_size_bytes" in sig.parameters:
        ingestion_kwargs["max_file_size_bytes"] = max_file_size_bytes

    ingestion = DocumentIngestionService(**ingestion_kwargs)
    chunking = DocumentChunkingService(
        max_chars=args.max_chunk_chars,
        overlap_chars=args.chunk_overlap_chars,
    )

    embedding_provider = None
    if args.embed:
        embedding_provider = build_embedding_provider(
            provider_name=settings.embeddings.provider,
            dimension=settings.embeddings.dimension,
        )

    input_paths = _resolve_input_files(args.path, recursive=args.recursive)
    if not input_paths:
        print("No supported files found for ingestion.")
        return 0

    summary = {
        "files_discovered": len(input_paths),
        "documents_ingested": 0,
        "chunks_written": 0,
        "tenant_id": tenant_id,
    }

    for file_path in input_paths:
        try:
            document = ingestion.ingest_file(
                file_path,
                source_metadata={"ingestion_source": "cli", "tenant_id": tenant_id},
            )
            chunks = chunking.chunk_document(document)
            embeddings = embedding_provider.embed_texts([chunk.chunk_text for chunk in chunks]) if embedding_provider else None

            document_id = repository.upsert_document(document, tenant_id=tenant_id)
            inserted_count = repository.replace_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
                tenant_id=tenant_id,
            )
            summary["documents_ingested"] += 1
            summary["chunks_written"] += inserted_count
        except Exception as exc:
            errors = summary.setdefault("errors", [])
            errors.append({
                "file": str(file_path),
                "error": str(exc),
            })

    print(json.dumps(summary, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest, persist, and manage source files for Knowledge Fabric.")
    parser.add_argument("--path", "--source-path", dest="path", default=None, help="Source file or directory path to ingest.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings.yaml.")
    parser.add_argument("--tenant", "--collection", dest="tenant", default="default", help="Tenant ID for multi-tenant isolation.")
    parser.add_argument("--source-type", dest="source_type", default=None, help="Optional source format hint (e.g. markdown, pdf).")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories.")
    parser.add_argument("--embed", action="store_true", help="Compute and store embeddings for chunks using configured provider.")
    parser.add_argument("--max-chunk-chars", type=int, default=1200, help="Maximum chunk size in characters.")
    parser.add_argument("--chunk-overlap-chars", type=int, default=150, help="Overlap size in characters.")
    parser.add_argument("--max-file-size-mb", type=int, default=25, help="Maximum allowed file size in MB for ingestion (default: 25).")

    # Observability & Operations options
    ops = parser.add_argument_group("Observability & Operations")
    ops.add_argument("--index-status", action="store_true", help="Display index statistics, document/chunk counts, and source distribution.")
    ops.add_argument("--check-consistency", action="store_true", help="Audit database consistency for orphaned chunks, empty docs, or null tenants.")
    ops.add_argument("--reembed", action="store_true", help="Recompute embeddings for all chunks in-place using configured provider.")

    # Data Lifecycle & Safe Deletion options
    lifecycle = parser.add_argument_group("Data Lifecycle & Deletion")
    lifecycle.add_argument("--delete-document-id", type=int, default=None, help="Delete a specific document by database ID.")
    lifecycle.add_argument("--delete-source-uri", default=None, help="Delete a specific document by its source URI.")
    lifecycle.add_argument("--delete-source-type", default=None, help="Delete all documents matching source-type within tenant.")
    lifecycle.add_argument("--purge-tenant", default=None, help="Purge all documents, chunks, and metadata for a tenant.")
    lifecycle.add_argument("--confirm", action="store_true", help="Confirm destructive purge operation.")
    return parser



def _resolve_input_files(path_value: str, *, recursive: bool) -> list[Path]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        return [path] if path.suffix.lower() in _SUPPORTED_EXTENSIONS else []

    if recursive:
        files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    else:
        files = sorted(candidate for candidate in path.iterdir() if candidate.is_file())
    return [candidate for candidate in files if candidate.suffix.lower() in _SUPPORTED_EXTENSIONS]


if __name__ == "__main__":
    raise SystemExit(main())
