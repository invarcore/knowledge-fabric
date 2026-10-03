"""Integration tests verifying chunking, hybrid retrieval, and cryptographic provenance on real NIST SP 800-53 corpus."""

from __future__ import annotations

from pathlib import Path

from knowledge_fabric.chunking.service import ChunkingConfig, DocumentChunkingService
from knowledge_fabric.embeddings.providers import MockEmbeddingProvider
from knowledge_fabric.evidence.models import (
    compute_chunk_hash,
    compute_package_digest,
    compute_package_signature,
)
from knowledge_fabric.ingestion.models import Document, SourceFormat
from knowledge_fabric.mcp.tools import KnowledgeFabricMCPTools
from knowledge_fabric.retrieval.models import RetrievalHit
from knowledge_fabric.retrieval.pipeline import RetrievalPipeline


class HermeticMemoryRetrievalStore:
    """In-memory multi-tenant retrieval store for fixture integration tests."""

    def __init__(self) -> None:
        self.documents: dict[str, list[dict]] = {}
        self.chunks: dict[str, list[dict]] = {}
        self._doc_id_seq = 1
        self._chunk_id_seq = 1

    def add_document(self, doc: Document, chunks: list, tenant_id: str = "default") -> int:
        doc_id = self._doc_id_seq
        self._doc_id_seq += 1
        self.documents.setdefault(tenant_id, []).append({
            "id": doc_id,
            "source_uri": doc.source_uri,
            "source_type": str(doc.source_format.value),
            "title": doc.title or "Untitled",
            "metadata": doc.metadata,
            "content_text": doc.content_text,
            "tenant_id": tenant_id,
        })
        for c in chunks:
            chunk_id = self._chunk_id_seq
            self._chunk_id_seq += 1
            self.chunks.setdefault(tenant_id, []).append({
                "id": chunk_id,
                "document_id": doc_id,
                "document_uri": doc.source_uri,
                "source_type": str(doc.source_format.value),
                "chunk_text": c.chunk_text,
                "metadata": c.metadata,
                "tenant_id": tenant_id,
            })
        return doc_id

    def lexical_search(
        self,
        query_text: str,
        top_k: int = 10,
        source_type: str | None = None,
        tenant_id: str | None = None,
    ) -> list[RetrievalHit]:
        effective_tenant = tenant_id or "default"
        tenant_chunks = self.chunks.get(effective_tenant, [])
        query_words = set(query_text.lower().split())

        scored: list[tuple[float, dict]] = []
        for c in tenant_chunks:
            if source_type and c["source_type"] != source_type:
                continue
            text = c["chunk_text"].lower()
            overlap = sum(1 for w in query_words if w in text)
            if overlap > 0:
                score = overlap / (len(query_words) + 1e-5)
                scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalHit(
                chunk_id=c["id"],
                document_uri=c["document_uri"],
                source_type=c["source_type"],
                snippet=c["chunk_text"],
                score=score,
                metadata=c["metadata"],
            )
            for score, c in scored[:top_k]
        ]

    def vector_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        source_type: str | None = None,
        tenant_id: str | None = None,
    ) -> list[RetrievalHit]:
        effective_tenant = tenant_id or "default"
        tenant_chunks = self.chunks.get(effective_tenant, [])
        hits: list[RetrievalHit] = []
        for i, c in enumerate(tenant_chunks):
            if source_type and c["source_type"] != source_type:
                continue
            score = max(0.05, 0.95 - (i * 0.05))
            hits.append(
                RetrievalHit(
                    chunk_id=c["id"],
                    document_uri=c["document_uri"],
                    source_type=c["source_type"],
                    snippet=c["chunk_text"],
                    score=score,
                    metadata=c["metadata"],
                )
            )
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]

    def get_document(
        self,
        document_id: int | None = None,
        source_uri: str | None = None,
        tenant_id: str | None = None,
    ) -> dict | None:
        effective_tenant = tenant_id or "default"
        for doc in self.documents.get(effective_tenant, []):
            if document_id is not None and doc["id"] == document_id:
                return doc
            if source_uri is not None and doc["source_uri"] == source_uri:
                return doc
        return None

    def list_sources_for_tenant(self, tenant_id: str | None = None) -> list[tuple]:
        effective_tenant = tenant_id or "default"
        counts: dict[str, int] = {}
        for doc in self.documents.get(effective_tenant, []):
            st = doc["source_type"]
            counts[st] = counts.get(st, 0) + 1
        return [(st, count) for st, count in counts.items()]


def test_real_world_nist_corpus_chunking_and_retrieval() -> None:
    """Verify that structure-preserving chunker and hybrid retrieval handle real enterprise specs."""
    corpus_path = Path(__file__).parent / "fixtures" / "corpora" / "nist_sp800_53_access_control.md"
    assert corpus_path.exists(), f"Corpus fixture missing: {corpus_path}"

    content = corpus_path.read_text(encoding="utf-8")
    assert "AC-1: Policy and Procedures" in content
    assert "IA-2(1): Multi-Factor Authentication" in content

    # 1. Structure-Preserving Chunking
    chunker = DocumentChunkingService(config=ChunkingConfig(max_chars=600, overlap_chars=80))
    doc = Document(
        source_uri="file:///compliance/nist_sp800_53_rev5.md",
        source_format=SourceFormat.MARKDOWN,
        title="NIST SP 800-53 Rev 5 - Access Control and Identification",
        content_text=content,
        metadata={"framework": "NIST", "standard": "SP-800-53", "revision": 5},
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 8, f"Expected at least 8 chunks from multi-section document, got {len(chunks)}"

    # Check Markdown table preservation
    table_chunks = [c for c in chunks if "| Baseline Level |" in c.chunk_text or "| Moderate |" in c.chunk_text]
    assert len(table_chunks) >= 1, "Markdown parameter table should be preserved in chunking"

    # Check YAML and JSON code fence preservation
    yaml_chunks = [c for c in chunks if "account_lifecycle_policy:" in c.chunk_text]
    assert len(yaml_chunks) >= 1, "YAML account lifecycle policy block must be preserved"

    json_chunks = [c for c in chunks if "EnforceMFAAndSubnetBoundary" in c.chunk_text]
    assert len(json_chunks) >= 1, "JSON IAM policy block must be preserved"

    # 2. Store & Hybrid Retrieval Pipeline
    store = HermeticMemoryRetrievalStore()
    doc_id = store.add_document(doc, chunks, tenant_id="compliance-engineering")
    assert doc_id == 1

    embedding_provider = MockEmbeddingProvider(_dimension=16)
    pipeline = RetrievalPipeline(
        retrieval_store=store,
        embedding_provider=embedding_provider,
        rrf_k=60,
    )

    # 3. Domain Queries
    # Query A: MFA / FIDO2 requirement
    query_mfa = "phishing-resistant multi-factor authentication FIDO2 WebAuthn hardware keys"
    pkg_mfa, trace_mfa = pipeline.retrieve_with_trace(
        query_text=query_mfa,
        top_k=5,
        tenant_id="compliance-engineering",
    )
    assert len(pkg_mfa.items) >= 1
    assert any("FIDO2" in item.snippet for item in pkg_mfa.items)
    assert any("IA-2(1)" in item.snippet or "phishing-resistant" in item.snippet for item in pkg_mfa.items)

    # Query B: Inactive account disabling parameter (90 days)
    query_inactive = "disables inactive accounts ninety 90 days inactivity"
    pkg_inactive, _ = pipeline.retrieve_with_trace(
        query_text=query_inactive,
        top_k=5,
        tenant_id="compliance-engineering",
    )
    assert len(pkg_inactive.items) >= 1
    assert any("ninety (90) days" in item.snippet for item in pkg_inactive.items)

    # Query C: Cryptographic password hash functions
    query_crypto = "cryptographically secure storage passwords Argon2id PBKDF2 iterations"
    pkg_crypto, _ = pipeline.retrieve_with_trace(
        query_text=query_crypto,
        top_k=5,
        tenant_id="compliance-engineering",
    )
    assert len(pkg_crypto.items) >= 1
    assert any("Argon2id" in item.snippet for item in pkg_crypto.items)

    # 4. Cryptographic Provenance & Tamper-Proof Signature
    item = pkg_mfa.items[0]
    expected_hash = compute_chunk_hash(item.document_uri, item.snippet)
    assert item.provenance_hash == expected_hash

    expected_digest = compute_package_digest(pkg_mfa.items)
    assert pkg_mfa.provenance_digest == expected_digest

    sig = compute_package_signature(
        pkg_mfa.retrieval_id,
        pkg_mfa.tenant_id,
        pkg_mfa.query_fingerprint,
        pkg_mfa.provenance_digest,
        pkg_mfa.generated_at.isoformat(),
        key="compliance-audit-signing-key",
    )
    assert len(sig) == 64

    # 5. Partition Isolation Gate
    foreign_pkg, _ = pipeline.retrieve_with_trace(
        query_text=query_mfa,
        top_k=5,
        tenant_id="unrelated-tenant-ops",
    )
    assert len(foreign_pkg.items) == 0, "Cross-tenant leakage: unrelated tenant accessed compliance data"

    # 6. MCP Interface on Real Corpus
    mcp_tools = KnowledgeFabricMCPTools(
        retrieval_pipeline=pipeline,
        retrieval_store=store,
        embedding_provider=embedding_provider,
    )
    retrieved = mcp_tools.retrieve_evidence(
        query_text=query_mfa,
        tenant_id="compliance-engineering",
        top_k=3,
    )
    assert len(retrieved["items"]) >= 1
    assert retrieved["tenant_id"] == "compliance-engineering"

    doc_data = mcp_tools.get_document(document_id=doc_id, tenant_id="compliance-engineering")
    assert doc_data is not None
    assert doc_data["title"] == "NIST SP 800-53 Rev 5 - Access Control and Identification"
