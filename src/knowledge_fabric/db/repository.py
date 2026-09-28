"""Persistence helpers for documents and chunks."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from knowledge_fabric.ingestion.models import Chunk, Document

ConnectionFactory = Callable[[], Any]


class KnowledgeRepository:
    """Repository for storing ingested documents and chunks in PostgreSQL."""

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._connection_factory = connection_factory

    def upsert_document(self, document: Document, *, tenant_id: str = "default") -> int:
        """Insert or update a document and return its database id."""
        connection = self._connection_factory()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (source_uri, source_type, title, metadata, content_text, tenant_id)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (source_uri)
                DO UPDATE SET
                  source_type = EXCLUDED.source_type,
                  title = EXCLUDED.title,
                  metadata = EXCLUDED.metadata,
                  content_text = EXCLUDED.content_text,
                  tenant_id = EXCLUDED.tenant_id,
                  updated_at = NOW()
                RETURNING id
                """,
                [
                    document.source_uri,
                    document.source_format.value,
                    document.title,
                    _to_json(document.metadata),
                    document.content_text,
                    tenant_id,
                ],
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("Failed to upsert document")
            if hasattr(connection, "commit"):
                connection.commit()
            return int(row[0])

    def replace_chunks(
        self,
        *,
        document_id: int,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
        tenant_id: str = "default",
    ) -> int:
        """Replace chunks for one document and return inserted chunk count."""
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embeddings length must match chunks length")

        connection = self._connection_factory()
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chunks WHERE document_id = %s", [document_id])

            for index, chunk in enumerate(chunks):
                embedding_literal = None
                if embeddings is not None:
                    embedding_literal = _vector_literal(embeddings[index])

                cursor.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, chunk_text, metadata, embedding, tenant_id)
                    VALUES (%s, %s, %s, %s::jsonb, %s::vector, %s)
                    """,
                    [
                        document_id,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        _to_json(chunk.metadata),
                        embedding_literal,
                        tenant_id,
                    ],
                )

            if hasattr(connection, "commit"):
                connection.commit()
        return len(chunks)

    def delete_document(
        self,
        *,
        document_id: int | None = None,
        source_uri: str | None = None,
        tenant_id: str | None = None,
    ) -> bool:
        """Delete one document by id or source URI, strictly scoped to tenant_id.

        PostgreSQL foreign key constraint (ON DELETE CASCADE) automatically
        removes all associated chunks and embeddings.
        Returns True if a document was deleted, False otherwise.
        """
        if document_id is None and source_uri is None:
            raise ValueError("document_id or source_uri must be provided")

        from knowledge_fabric.retrieval.pipeline import validate_tenant_id

        effective_tenant = validate_tenant_id(tenant_id)
        connection = self._connection_factory()
        with connection.cursor() as cursor:
            if effective_tenant is not None:
                cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])

            if document_id is not None:
                if effective_tenant is not None:
                    cursor.execute(
                        "DELETE FROM documents WHERE id = %s AND tenant_id = %s",
                        [document_id, effective_tenant],
                    )
                else:
                    cursor.execute("DELETE FROM documents WHERE id = %s", [document_id])
            else:
                if effective_tenant is not None:
                    cursor.execute(
                        "DELETE FROM documents WHERE source_uri = %s AND tenant_id = %s",
                        [source_uri, effective_tenant],
                    )
                else:
                    cursor.execute("DELETE FROM documents WHERE source_uri = %s", [source_uri])

            deleted_count = cursor.rowcount if hasattr(cursor, "rowcount") else 0
            if hasattr(connection, "commit"):
                connection.commit()
            return bool(deleted_count and deleted_count > 0)

    def delete_by_source(self, *, source_type: str, tenant_id: str) -> int:
        """Delete all documents and cascading chunks of a given source_type for a tenant."""
        from knowledge_fabric.retrieval.pipeline import validate_tenant_id

        effective_tenant = validate_tenant_id(tenant_id)
        if not effective_tenant:
            raise ValueError("tenant_id is required for delete_by_source")

        connection = self._connection_factory()
        with connection.cursor() as cursor:
            cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])

            cursor.execute(
                "DELETE FROM documents WHERE source_type = %s AND tenant_id = %s",
                [source_type, effective_tenant],
            )
            deleted_count = cursor.rowcount if hasattr(cursor, "rowcount") else 0
            if hasattr(connection, "commit"):
                connection.commit()
            return max(0, deleted_count if deleted_count is not None else 0)

    def purge_tenant(self, tenant_id: str) -> dict[str, Any]:
        """Completely purge all documents, chunks, retrieval runs, and audit events for a tenant.

        Requires an explicit, non-empty tenant identifier.
        """
        from knowledge_fabric.retrieval.pipeline import validate_tenant_id

        effective_tenant = validate_tenant_id(tenant_id)
        if not effective_tenant:
            raise ValueError("tenant_id must be non-empty for purge_tenant")

        connection = self._connection_factory()
        with connection.cursor() as cursor:
            cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])

            # Deleting documents cascades to all chunks
            cursor.execute("DELETE FROM documents WHERE tenant_id = %s", [effective_tenant])
            doc_count = cursor.rowcount if hasattr(cursor, "rowcount") else 0

            cursor.execute("DELETE FROM retrieval_runs WHERE tenant_id = %s", [effective_tenant])
            run_count = cursor.rowcount if hasattr(cursor, "rowcount") else 0

            cursor.execute("DELETE FROM audit_events WHERE tenant_id = %s", [effective_tenant])
            audit_count = cursor.rowcount if hasattr(cursor, "rowcount") else 0

            if hasattr(connection, "commit"):
                connection.commit()

            return {
                "tenant_id": effective_tenant,
                "documents_deleted": max(0, doc_count if doc_count is not None else 0),
                "retrieval_runs_deleted": max(0, run_count if run_count is not None else 0),
                "audit_events_deleted": max(0, audit_count if audit_count is not None else 0),
            }

    def get_chunk(self, chunk_id: int, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        """Fetch a single chunk by ID with its document metadata, scoped to tenant."""
        from knowledge_fabric.retrieval.pipeline import validate_tenant_id

        effective_tenant = validate_tenant_id(tenant_id)
        connection = self._connection_factory()
        with connection.cursor() as cursor:
            if effective_tenant is not None:
                cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])

            where_clause = "WHERE c.id = %s"
            params: list[Any] = [chunk_id]
            if effective_tenant is not None:
                where_clause += " AND (c.tenant_id = %s OR d.tenant_id = %s)"
                params.extend([effective_tenant, effective_tenant])

            sql = f"""
                SELECT
                  c.id,
                  c.document_id,
                  d.source_uri,
                  d.source_type,
                  d.title,
                  c.chunk_index,
                  c.chunk_text,
                  c.metadata,
                  c.tenant_id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                {where_clause}
                LIMIT 1
            """
            cursor.execute(sql, params)
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "chunk_id": int(row[0]),
                "document_id": int(row[1]),
                "document_uri": str(row[2]),
                "source_type": str(row[3]),
                "title": str(row[4]) if row[4] is not None else "",
                "chunk_index": int(row[5]),
                "chunk_text": str(row[6]),
                "metadata": dict(row[7] or {}),
                "tenant_id": str(row[8]) if row[8] is not None else None,
            }

    def get_index_status(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        """Query index health, document counts, and source distribution."""
        from knowledge_fabric.retrieval.pipeline import validate_tenant_id

        effective_tenant = validate_tenant_id(tenant_id)
        connection = self._connection_factory()
        with connection.cursor() as cursor:
            if effective_tenant is not None:
                cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])

            where_doc = ""
            where_chunk = ""
            params_doc: list[Any] = []
            params_chunk: list[Any] = []
            if effective_tenant is not None:
                where_doc = "WHERE tenant_id = %s"
                where_chunk = "WHERE tenant_id = %s"
                params_doc = [effective_tenant]
                params_chunk = [effective_tenant]

            # Total documents
            cursor.execute(f"SELECT COUNT(*) FROM documents {where_doc}", params_doc)
            row_docs = cursor.fetchone()
            total_documents = int(row_docs[0]) if row_docs else 0

            # Total chunks
            cursor.execute(f"SELECT COUNT(*) FROM chunks {where_chunk}", params_chunk)
            row_chunks = cursor.fetchone()
            total_chunks = int(row_chunks[0]) if row_chunks else 0

            # Breakdown by source
            sql_sources = f"""
                SELECT
                  d.source_type,
                  COUNT(DISTINCT d.id) AS doc_count,
                  COUNT(c.id) AS chunk_count,
                  MAX(d.updated_at) AS last_indexed_at
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                {where_doc}
                GROUP BY d.source_type
                ORDER BY d.source_type ASC
            """
            cursor.execute(sql_sources, params_doc)
            sources = [
                {
                    "source_type": str(r[0]),
                    "document_count": int(r[1]),
                    "chunk_count": int(r[2]),
                    "last_indexed_at": str(r[3]) if r[3] else None,
                }
                for r in cursor.fetchall()
            ]

            return {
                "tenant_id": effective_tenant or "all",
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "sources": sources,
                "storage_backend": "postgres",
            }

    def check_consistency(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        """Validate database relational invariants and detect orphaned records."""
        from knowledge_fabric.retrieval.pipeline import validate_tenant_id

        effective_tenant = validate_tenant_id(tenant_id)
        connection = self._connection_factory()
        with connection.cursor() as cursor:
            if effective_tenant is not None:
                cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])

            where_clause = ""
            params: list[Any] = []
            if effective_tenant is not None:
                where_clause = "WHERE tenant_id = %s"
                params = [effective_tenant]

            # Orphaned chunks (document_id does not exist in documents)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM chunks c
                LEFT JOIN documents d ON d.id = c.document_id
                WHERE d.id IS NULL
                """
            )
            row_orphans = cursor.fetchone()
            orphaned_chunks = int(row_orphans[0]) if row_orphans else 0

            # Empty documents (document with 0 chunks)
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE c.id IS NULL
                {"AND d.tenant_id = %s" if effective_tenant else ""}
                """,
                params,
            )
            row_empty = cursor.fetchone()
            empty_documents = int(row_empty[0]) if row_empty else 0

            # Documents with NULL tenant_id
            cursor.execute("SELECT COUNT(*) FROM documents WHERE tenant_id IS NULL")
            row_null_tenants = cursor.fetchone()
            null_tenant_documents = int(row_null_tenants[0]) if row_null_tenants else 0

            # Chunks without embedding
            cursor.execute(
                f"SELECT COUNT(*) FROM chunks WHERE embedding IS NULL {'AND tenant_id = %s' if effective_tenant else ''}",
                params,
            )
            row_null_embeddings = cursor.fetchone()
            null_embedding_chunks = int(row_null_embeddings[0]) if row_null_embeddings else 0

            is_healthy = orphaned_chunks == 0 and null_tenant_documents == 0

            return {
                "tenant_id": effective_tenant or "all",
                "is_healthy": is_healthy,
                "orphaned_chunks": orphaned_chunks,
                "empty_documents": empty_documents,
                "null_tenant_documents": null_tenant_documents,
                "null_embedding_chunks": null_embedding_chunks,
            }

    def reembed_chunks(
        self,
        *,
        embedding_provider: Any,
        tenant_id: str | None = None,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Recompute vector embeddings in-place using configured embedding provider."""
        from knowledge_fabric.retrieval.pipeline import validate_tenant_id

        effective_tenant = validate_tenant_id(tenant_id)
        connection = self._connection_factory()
        with connection.cursor() as cursor:
            if effective_tenant is not None:
                cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])

            where_clause = ""
            params: list[Any] = []
            if effective_tenant is not None:
                where_clause = "WHERE tenant_id = %s"
                params = [effective_tenant]

            cursor.execute(f"SELECT id, chunk_text FROM chunks {where_clause} ORDER BY id", params)
            rows = cursor.fetchall()
            total_reembedded = 0

            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                chunk_ids = [r[0] for r in batch]
                texts = [r[1] for r in batch]
                vectors = embedding_provider.embed_texts(texts)

                for cid, vec in zip(chunk_ids, vectors):
                    cursor.execute(
                        "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                        [_vector_literal(vec), cid],
                    )
                    total_reembedded += 1

            if hasattr(connection, "commit"):
                connection.commit()

            return {
                "tenant_id": effective_tenant or "all",
                "chunks_reembedded": total_reembedded,
                "provider": type(embedding_provider).__name__,
                "dimension": embedding_provider.dimension,
            }



def _to_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _vector_literal(values: list[float]) -> str:
    if not values:
        raise ValueError("embedding vector must not be empty")
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"
