from __future__ import annotations

from knowledge_fabric.embeddings import MockEmbeddingProvider
from knowledge_fabric.retrieval import RetrievalHit
from knowledge_fabric.retrieval.pipeline import RetrievalPipeline


class _FakeStore:
    def lexical_search(
        self,
        query_text: str,
        top_k: int = 10,
        source_type: str | None = None,
        tenant_id: str | None = None,
        **kwargs: object,
    ) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id=1,
                document_id=10,
                document_uri="memory://doc-10",
                chunk_index=0,
                chunk_text="lexical result",
                score=0.8,
                source="lexical",
                metadata={"kind": "lexical"},
            )
        ]

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_type: str | None = None,
        tenant_id: str | None = None,
        **kwargs: object,
    ) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id=2,
                document_id=11,
                document_uri="memory://doc-11",
                chunk_index=1,
                chunk_text="vector result",
                score=0.7,
                source="vector",
                metadata={"kind": "vector"},
            )
        ]


class _FakeAuditLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def log_retrieval(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def test_pipeline_retrieves_and_logs() -> None:
    audit = _FakeAuditLogger()
    pipeline = RetrievalPipeline(
        retrieval_store=_FakeStore(),  # type: ignore[arg-type]
        embedding_provider=MockEmbeddingProvider(_dimension=8),
        audit_logger=audit,  # type: ignore[arg-type]
    )

    package, trace = pipeline.retrieve_with_trace(
        query_text="test query", top_k=5, trace_id="trace-1", tenant_id="tenant-alpha"
    )

    assert package.query_text == "test query"
    assert len(package.items) == 2
    assert trace.fused_count == 2
    assert len(audit.calls) == 1
    assert audit.calls[0]["trace_id"] == "trace-1"
    assert audit.calls[0]["tenant_id"] == "tenant-alpha"


def test_pipeline_explain_retrieval() -> None:
    audit = _FakeAuditLogger()
    pipeline = RetrievalPipeline(
        retrieval_store=_FakeStore(),  # type: ignore[arg-type]
        embedding_provider=MockEmbeddingProvider(_dimension=8),
        audit_logger=audit,  # type: ignore[arg-type]
    )

    details = pipeline.explain_retrieval(query_text="security", top_k=3, tenant_id="tenant-beta")

    assert details["query_text"] == "security"
    assert details["top_k"] == 3
    assert details["trace"]["lexical_count"] == 1
    assert set(details["sources"]) == {"lexical", "vector"}
    assert details["legs"]["lexical"]["status"] == "ok"
    assert details["legs"]["vector"]["status"] == "ok"
    assert len(audit.calls) == 0  # explain_retrieval must skip audit logging


def test_pipeline_lexical_only_mode_bypasses_embeddings() -> None:
    class _StrictNoEmbeddingProvider:
        dimension = 8

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("embed_texts must not be called in lexical-only mode")

    pipeline = RetrievalPipeline(
        retrieval_store=_FakeStore(),  # type: ignore[arg-type]
        embedding_provider=_StrictNoEmbeddingProvider(),  # type: ignore[arg-type]
    )

    package, trace = pipeline.retrieve_with_trace(query_text="exact match", mode="lexical")

    assert package.retrieval_summary["requested_mode"] == "lexical"
    assert package.retrieval_summary["strategy"] == "lexical"
    assert package.retrieval_summary["is_degraded"] is False
    assert package.retrieval_summary["legs"]["vector"]["status"] == "skipped"
    assert package.retrieval_summary["legs"]["lexical"]["status"] == "ok"
    assert trace.vector_status == "skipped"
    assert trace.lexical_status == "ok"
    assert len(package.items) == 1
    assert package.items[0].snippet == "lexical result"


def test_pipeline_vector_only_mode_bypasses_lexical() -> None:
    class _StrictNoLexicalStore:
        def lexical_search(self, *args, **kwargs):
            raise AssertionError("lexical_search must not be called in vector-only mode")

        def vector_search(self, *args, **kwargs):
            return _FakeStore().vector_search(*args, **kwargs)

    pipeline = RetrievalPipeline(
        retrieval_store=_StrictNoLexicalStore(),  # type: ignore[arg-type]
        embedding_provider=MockEmbeddingProvider(_dimension=8),
    )

    package, trace = pipeline.retrieve_with_trace(query_text="semantic meaning", mode="vector")

    assert package.retrieval_summary["requested_mode"] == "vector"
    assert package.retrieval_summary["strategy"] == "vector"
    assert package.retrieval_summary["is_degraded"] is False
    assert package.retrieval_summary["legs"]["lexical"]["status"] == "skipped"
    assert package.retrieval_summary["legs"]["vector"]["status"] == "ok"
    assert trace.lexical_status == "skipped"
    assert trace.vector_status == "ok"
    assert len(package.items) == 1
    assert package.items[0].snippet == "vector result"


def test_pipeline_hybrid_degraded_when_vector_leg_fails() -> None:
    class _FailingVectorStore:
        def lexical_search(self, *args, **kwargs):
            return _FakeStore().lexical_search(*args, **kwargs)

        def vector_search(self, *args, **kwargs):
            raise RuntimeError("pgvector connection dropped")

    pipeline = RetrievalPipeline(
        retrieval_store=_FailingVectorStore(),  # type: ignore[arg-type]
        embedding_provider=MockEmbeddingProvider(_dimension=8),
    )

    package, trace = pipeline.retrieve_with_trace(
        query_text="robust query", mode="hybrid", fail_closed=False
    )

    assert package.retrieval_summary["requested_mode"] == "hybrid"
    assert package.retrieval_summary["strategy"] == "degraded_lexical"
    assert package.retrieval_summary["is_degraded"] is True
    assert package.retrieval_summary["legs"]["vector"]["status"] == "error"
    assert package.retrieval_summary["legs"]["lexical"]["status"] == "ok"
    assert any("vector_leg_failed" in w for w in package.retrieval_summary["warnings"])
    assert trace.is_degraded is True
    assert len(package.items) == 1
    assert package.items[0].snippet == "lexical result"


def test_pipeline_hybrid_fails_closed_when_flagged() -> None:
    import pytest

    class _FailingVectorStore:
        def lexical_search(self, *args, **kwargs):
            return _FakeStore().lexical_search(*args, **kwargs)

        def vector_search(self, *args, **kwargs):
            raise RuntimeError("Fatal vector backend failure")

    pipeline = RetrievalPipeline(
        retrieval_store=_FailingVectorStore(),  # type: ignore[arg-type]
        embedding_provider=MockEmbeddingProvider(_dimension=8),
    )

    with pytest.raises(RuntimeError, match="Fatal vector backend failure"):
        pipeline.retrieve_with_trace(query_text="strict query", mode="hybrid", fail_closed=True)


def test_pipeline_rejects_invalid_mode() -> None:
    import pytest

    pipeline = RetrievalPipeline(
        retrieval_store=_FakeStore(),  # type: ignore[arg-type]
        embedding_provider=MockEmbeddingProvider(_dimension=8),
    )

    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        pipeline.retrieve_with_trace(query_text="query", mode="unsupported_quantum")

