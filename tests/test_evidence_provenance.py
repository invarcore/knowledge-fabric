"""Evidence provenance contract tests for Knowledge Fabric.

Verifies SHA-256 chunk hashing, composite package digests, query fingerprinting,
and backward compatibility with existing tests.
"""

import hashlib
import json

from knowledge_fabric.evidence.models import (
    EvidenceItem,
    EvidencePackage,
    build_evidence_package,
    compute_chunk_hash,
    compute_package_digest,
    compute_query_fingerprint,
)
from knowledge_fabric.fusion.rrf import HybridHit
from knowledge_fabric.retrieval.models import RetrievalHit


# ── Hash Computation Tests ─────────────────────────────────────────

def test_compute_chunk_hash_deterministic():
    """Same inputs always produce the same SHA-256 hash."""
    h1 = compute_chunk_hash("docs/policy.md", "Access requires MFA.")
    h2 = compute_chunk_hash("docs/policy.md", "Access requires MFA.")
    assert h1 == h2
    assert len(h1) == 64


def test_compute_chunk_hash_changes_with_uri():
    """Different document_uri produces a different hash."""
    h1 = compute_chunk_hash("docs/a.md", "same content")
    h2 = compute_chunk_hash("docs/b.md", "same content")
    assert h1 != h2


def test_compute_chunk_hash_changes_with_content():
    """Different snippet produces a different hash."""
    h1 = compute_chunk_hash("docs/a.md", "content A")
    h2 = compute_chunk_hash("docs/a.md", "content B")
    assert h1 != h2


def test_compute_chunk_hash_matches_manual_sha256():
    """Hash matches manually computed SHA-256 of canonical JSON [uri, snippet]."""
    uri, snippet = "docs/policy.md", "MFA required"
    expected = hashlib.sha256(
        json.dumps([uri, snippet], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert compute_chunk_hash(uri, snippet) == expected


def test_compute_package_digest_deterministic():
    """Composite digest is stable across calls."""
    items = [
        EvidenceItem(
            chunk_id=1,
            document_id=1,
            document_uri="a.md",
            chunk_index=0,
            snippet="text",
            score=0.9,
            provenance_hash="a" * 64,
        ),
        EvidenceItem(
            chunk_id=2,
            document_id=2,
            document_uri="b.md",
            chunk_index=0,
            snippet="text2",
            score=0.8,
            provenance_hash="b" * 64,
        ),
    ]
    d1 = compute_package_digest(items)
    d2 = compute_package_digest(items)
    assert d1 == d2
    assert len(d1) == 64


def test_compute_package_digest_empty_list():
    """Empty items list produces empty string digest."""
    assert compute_package_digest([]) == ""


def test_compute_query_fingerprint_tenant_isolation():
    """Same query on different tenants produces different fingerprints."""
    f1 = compute_query_fingerprint("admin grants", "tenant_a", "hybrid")
    f2 = compute_query_fingerprint("admin grants", "tenant_b", "hybrid")
    assert f1 != f2


def test_compute_query_fingerprint_mode_sensitivity():
    """Same query with different mode produces different fingerprint."""
    f1 = compute_query_fingerprint("query", "tenant", "hybrid")
    f2 = compute_query_fingerprint("query", "tenant", "lexical")
    assert f1 != f2


# ── build_evidence_package Integration Tests ────────────────────────

def test_build_evidence_package_populates_provenance_hash():
    """Each item in the package has a non-empty provenance_hash."""
    hits = [_make_hybrid_hit(1, "doc.md", "evidence text")]
    package = build_evidence_package("query", hits, tenant_id="t1")
    assert package.items[0].provenance_hash != ""
    assert len(package.items[0].provenance_hash) == 64


def test_build_evidence_package_populates_package_digest():
    """Package provenance_digest is computed from chunk hashes."""
    hits = [_make_hybrid_hit(1, "a.md", "text A"), _make_hybrid_hit(2, "b.md", "text B")]
    package = build_evidence_package("query", hits, tenant_id="t1")
    assert package.provenance_digest != ""
    expected = compute_package_digest(package.items)
    assert package.provenance_digest == expected


def test_build_evidence_package_populates_query_fingerprint():
    """Package query_fingerprint is non-empty when tenant_id is provided."""
    hits = [_make_hybrid_hit(1, "doc.md", "text")]
    package = build_evidence_package("query", hits, tenant_id="t1", mode="hybrid")
    assert package.query_fingerprint != ""
    assert len(package.query_fingerprint) == 64


def test_build_evidence_package_backward_compatible():
    """Calling without new parameters still works (backward compat)."""
    hits = [_make_hybrid_hit(1, "doc.md", "text")]
    package = build_evidence_package("query", hits)
    assert package.query_text == "query"
    assert len(package.items) == 1
    assert package.items[0].provenance_hash != ""


def test_evidence_package_to_dict_emits_task1_canonical_schema():
    """EvidencePackage.to_dict() emits Task 1 canonical keys and chunk schema."""
    hits = [_make_hybrid_hit(1, "doc.md", "evidence text")]
    package = build_evidence_package("query", hits, tenant_id="tenant-123")
    d = package.to_dict()

    # Task 1 top-level keys
    assert "retrieval_id" in d and len(d["retrieval_id"]) > 0
    assert d["tenant_id"] == "tenant-123"
    assert "timestamp_utc" in d
    assert "provenance_digest" in d
    assert "chunks" in d

    # Task 1 chunk keys
    chunk = d["chunks"][0]
    assert chunk["chunk_id"] == "1"
    assert chunk["source_uri"] == "doc.md"
    assert chunk["document_uri"] == "doc.md"
    assert chunk["content"] == "evidence text"
    assert chunk["provenance_hash"] == compute_chunk_hash("doc.md", "evidence text")
    assert chunk["score"] == 0.5


# ── Helper ──────────────────────────────────────────────────────────

def _make_hybrid_hit(chunk_id: int, uri: str, text: str) -> HybridHit:
    return HybridHit(
        hit=RetrievalHit(
            chunk_id=chunk_id,
            document_id=chunk_id,
            document_uri=uri,
            chunk_index=0,
            chunk_text=text,
            score=0.9,
            source="lexical",
        ),
        fused_score=0.5,
        sources=["lexical"],
    )
