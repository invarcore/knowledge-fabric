from __future__ import annotations

from typing import Any

import pytest

from knowledge_fabric.db.repository import KnowledgeRepository
from knowledge_fabric.ingestion.models import Chunk, Document, SourceFormat


class _FakeCursor:
    def __init__(self, fetch_rows: list[tuple[Any, ...]] | None = None, rowcount: int = 1) -> None:
        self.fetch_rows = fetch_rows or [(101,)]
        self.executions: list[tuple[str, list[Any]]] = []
        self.rowcount = rowcount

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        self.executions.append((sql, params or []))

    def fetchone(self) -> tuple[Any, ...] | None:
        if not self.fetch_rows:
            return None
        return self.fetch_rows.pop(0)



class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.commit_calls = 0

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_calls += 1


def test_upsert_document_returns_id() -> None:
    cursor = _FakeCursor(fetch_rows=[(77,)])
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    document = Document(
        source_uri="memory://doc",
        source_format=SourceFormat.TXT,
        content_text="hello",
        title="doc",
        metadata={"k": "v"},
    )
    document_id = repository.upsert_document(document)

    assert document_id == 77
    assert any("INSERT INTO documents" in e[0] for e in cursor.executions)
    assert any("set_config('app.tenant_id'" in e[0] for e in cursor.executions)
    assert connection.commit_calls == 1


def test_replace_chunks_writes_all_chunks() -> None:
    cursor = _FakeCursor()
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    chunks = [
        Chunk(document_uri="memory://doc", chunk_index=0, chunk_text="a", metadata={}),
        Chunk(document_uri="memory://doc", chunk_index=1, chunk_text="b", metadata={}),
    ]
    inserted = repository.replace_chunks(document_id=1, chunks=chunks)

    assert inserted == 2
    assert any("DELETE FROM chunks" in e[0] for e in cursor.executions)
    assert any("set_config('app.tenant_id'" in e[0] for e in cursor.executions)
    assert len(cursor.executions) == 4  # set_config + DELETE + 2 INSERTs
    assert connection.commit_calls == 1


def test_replace_chunks_validates_embedding_count() -> None:
    cursor = _FakeCursor()
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)
    chunks = [Chunk(document_uri="memory://doc", chunk_index=0, chunk_text="a", metadata={})]

    with pytest.raises(ValueError, match="embeddings length must match chunks length"):
        repository.replace_chunks(document_id=1, chunks=chunks, embeddings=[[0.1], [0.2]])


def test_upsert_document_with_tenant_id() -> None:
    cursor = _FakeCursor(fetch_rows=[(88,)])
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    document = Document(
        source_uri="memory://tenant_doc",
        source_format=SourceFormat.TXT,
        content_text="tenant data",
        title="tenant doc",
        metadata={},
    )
    doc_id = repository.upsert_document(document, tenant_id="tenant-alpha")
    assert doc_id == 88
    # Last param in INSERT is tenant_id
    insert_exec = [e for e in cursor.executions if "INSERT INTO documents" in e[0]][0]
    assert insert_exec[1][-1] == "tenant-alpha"
    set_config_exec = [e for e in cursor.executions if "set_config" in e[0]][0]
    assert set_config_exec[1] == ["tenant-alpha"]


def test_replace_chunks_with_tenant_id() -> None:
    cursor = _FakeCursor()
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    chunks = [Chunk(document_uri="memory://doc", chunk_index=0, chunk_text="a", metadata={})]
    repository.replace_chunks(document_id=1, chunks=chunks, tenant_id="tenant-beta")

    chunk_inserts = [e for e in cursor.executions if "INSERT INTO chunks" in e[0]]
    assert chunk_inserts[0][1][-1] == "tenant-beta"
    set_config_exec = [e for e in cursor.executions if "set_config" in e[0]][0]
    assert set_config_exec[1] == ["tenant-beta"]


def test_delete_document_by_id_scoped_to_tenant() -> None:
    cursor = _FakeCursor(rowcount=1)
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    deleted = repository.delete_document(document_id=42, tenant_id="tenant-alpha")
    assert deleted is True
    assert connection.commit_calls == 1

    delete_stmt = [e for e in cursor.executions if "DELETE FROM documents" in e[0]][0]
    assert "WHERE id = %s AND tenant_id = %s" in delete_stmt[0]
    assert delete_stmt[1] == [42, "tenant-alpha"]


def test_delete_document_by_source_uri_scoped_to_tenant() -> None:
    cursor = _FakeCursor(rowcount=1)
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    deleted = repository.delete_document(source_uri="file:///policy.md", tenant_id="tenant-alpha")
    assert deleted is True
    assert connection.commit_calls == 1

    delete_stmt = [e for e in cursor.executions if "DELETE FROM documents" in e[0]][0]
    assert "WHERE source_uri = %s AND tenant_id = %s" in delete_stmt[0]
    assert delete_stmt[1] == ["file:///policy.md", "tenant-alpha"]


def test_delete_document_not_found_returns_false() -> None:
    cursor = _FakeCursor(rowcount=0)
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    deleted = repository.delete_document(document_id=999, tenant_id="tenant-alpha")
    assert deleted is False


def test_delete_by_source_for_tenant() -> None:
    cursor = _FakeCursor(rowcount=5)
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    deleted_count = repository.delete_by_source(source_type="confluence", tenant_id="tenant-alpha")
    assert deleted_count == 5
    assert connection.commit_calls == 1

    delete_stmt = [e for e in cursor.executions if "DELETE FROM documents" in e[0]][0]
    assert "WHERE source_type = %s AND tenant_id = %s" in delete_stmt[0]
    assert delete_stmt[1] == ["confluence", "tenant-alpha"]


def test_purge_tenant_removes_all_tenant_entities() -> None:
    cursor = _FakeCursor(rowcount=7)
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    summary = repository.purge_tenant(tenant_id="tenant-gamma")
    assert summary["tenant_id"] == "tenant-gamma"
    assert summary["documents_deleted"] == 7
    assert summary["retrieval_runs_deleted"] == 7
    assert summary["audit_events_deleted"] == 7
    assert connection.commit_calls == 1

    deletions = [e[0] for e in cursor.executions if "DELETE FROM" in e[0]]
    assert len(deletions) == 3
    assert any("FROM documents WHERE tenant_id = %s" in s for s in deletions)
    assert any("FROM retrieval_runs WHERE tenant_id = %s" in s for s in deletions)
    assert any("FROM audit_events WHERE tenant_id = %s" in s for s in deletions)


def test_delete_and_purge_reject_malformed_tenant_traversal() -> None:
    cursor = _FakeCursor()
    connection = _FakeConnection(cursor)
    repository = KnowledgeRepository(connection_factory=lambda: connection)

    with pytest.raises(ValueError, match="Invalid tenant_id format"):
        repository.delete_document(document_id=1, tenant_id="../../etc/passwd")

    with pytest.raises(ValueError, match="Invalid tenant_id format"):
        repository.delete_by_source(source_type="markdown", tenant_id="../malicious")

    with pytest.raises(ValueError, match="Invalid tenant_id format"):
        repository.purge_tenant(tenant_id="bad tenant with space")


