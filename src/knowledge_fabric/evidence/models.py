# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Evidence package contracts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from knowledge_fabric.fusion.rrf import HybridHit


def compute_chunk_hash(document_uri: str, snippet: str) -> str:
    """Compute deterministic SHA-256 provenance hash using RFC 8785 canonical JSON serialization.

    Formula: SHA-256(json.dumps([document_uri, snippet], separators=(',', ':'), ensure_ascii=False))
    """
    canonical_payload = json.dumps(
        [document_uri, snippet],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def compute_package_digest(items: list[EvidenceItem]) -> str:
    """Compute composite SHA-256 digest over ordered chunk provenance hashes.

    Formula: SHA-256(hash_1 + ":" + hash_2 + ":" + ... + hash_N)
    Returns empty string if items list is empty.
    """
    if not items:
        return ""
    combined = ":".join(item.provenance_hash for item in items).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def compute_package_signature(
    retrieval_id: str,
    tenant_id: str | None,
    query_fingerprint: str,
    provenance_digest: str,
    timestamp_utc: str,
    key: str | bytes | None = None,
) -> str:
    """Compute HMAC-SHA256 signature for evidence package provenance using canonical JSON.

    Formula: HMAC-SHA256(key, canonical_json([retrieval_id, tenant_id, query_fingerprint, provenance_digest, timestamp_utc]))
    Returns empty string if key is not configured.
    """
    secret = key if key is not None else os.environ.get("KF_EVIDENCE_HMAC_KEY", "")
    if not secret:
        return ""
    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    else:
        secret_bytes = secret
    tenant = tenant_id or ""
    msg = json.dumps(
        [retrieval_id, tenant, query_fingerprint, provenance_digest, timestamp_utc],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hmac.new(secret_bytes, msg, hashlib.sha256).hexdigest()


def compute_query_fingerprint(
    query_text: str,
    tenant_id: str | None = None,
    mode: str = "hybrid",
) -> str:
    """Compute deterministic SHA-256 fingerprint for a retrieval request.

    Formula: SHA-256(canonical_json([tenant_id, query_text, mode]))
    """
    tenant = tenant_id if tenant_id is not None else ""
    msg = json.dumps([tenant, query_text, mode], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(msg).hexdigest()


@dataclass(slots=True)
class EvidenceItem:
    """One evidence entry returned to callers."""

    chunk_id: int
    document_id: int
    document_uri: str
    chunk_index: int
    snippet: str
    score: float
    retrieval_sources: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    provenance_hash: str = ""

    @property
    def citation_source(self) -> str:
        """Return human-readable citation source string."""
        path = self.metadata.get("heading_path")
        if path:
            return f"{self.document_uri}#{path}"
        return self.document_uri


@dataclass(slots=True)
class EvidencePackage:
    """Top-level evidence response."""

    query_text: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    items: list[EvidenceItem] = field(default_factory=list)
    retrieval_summary: dict[str, object] = field(default_factory=dict)
    provenance_digest: str = ""
    query_fingerprint: str = ""
    retrieval_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str | None = None
    package_signature: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval_id": self.retrieval_id,
            "tenant_id": self.tenant_id,
            "timestamp_utc": self.generated_at.isoformat(),
            "query_text": self.query_text,
            "generated_at": self.generated_at.isoformat(),
            "items": [asdict(item) for item in self.items],
            "chunks": [
                {
                    "chunk_id": str(item.chunk_id),
                    "content": item.snippet,
                    "source_uri": item.document_uri,
                    "provenance_hash": item.provenance_hash,
                    "score": item.score,
                }
                for item in self.items
            ],
            "retrieval_summary": self.retrieval_summary,
            "provenance_digest": self.provenance_digest,
            "query_fingerprint": self.query_fingerprint,
            "package_signature": self.package_signature,
        }

    def format_citations(self, style: str = "markdown") -> str:
        """Format evidence items into structured text citations."""
        if not self.items:
            return "No citations available."
        lines: list[str] = []
        for i, item in enumerate(self.items, 1):
            heading_str = f" ({item.metadata.get('heading_path')})" if item.metadata.get("heading_path") else ""
            lines.append(f"[{i}] {item.document_uri}{heading_str} (score: {item.score:.4f})")
            lines.append(f"    \"{item.snippet}\"")
        return "\n".join(lines)


def build_evidence_package(
    query_text: str,
    hybrid_hits: list[HybridHit],
    *,
    summary_extra: dict[str, object] | None = None,
    tenant_id: str | None = None,
    mode: str = "hybrid",
    retrieval_id: str | None = None,
    hmac_key: str | bytes | None = None,
) -> EvidencePackage:
    items = [
        EvidenceItem(
            chunk_id=hybrid.hit.chunk_id,
            document_id=hybrid.hit.document_id,
            document_uri=hybrid.hit.document_uri,
            chunk_index=hybrid.hit.chunk_index,
            snippet=hybrid.hit.chunk_text,
            score=hybrid.fused_score,
            retrieval_sources=hybrid.sources,
            metadata=hybrid.hit.metadata,
            provenance_hash=compute_chunk_hash(
                hybrid.hit.document_uri,
                hybrid.hit.chunk_text,
            ),
        )
        for hybrid in hybrid_hits
    ]
    summary: dict[str, object] = {
        "total_items": len(items),
        "sources": sorted({source for item in items for source in item.retrieval_sources}),
    }
    if summary_extra:
        summary.update(summary_extra)

    pkg_retrieval_id = retrieval_id or uuid.uuid4().hex
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    digest = compute_package_digest(items)
    fingerprint = compute_query_fingerprint(query_text, tenant_id=tenant_id, mode=mode)
    signature = compute_package_signature(
        retrieval_id=pkg_retrieval_id,
        tenant_id=tenant_id,
        query_fingerprint=fingerprint,
        provenance_digest=digest,
        timestamp_utc=now_iso,
        key=hmac_key,
    )

    return EvidencePackage(
        query_text=query_text,
        generated_at=now,
        items=items,
        retrieval_summary=summary,
        provenance_digest=digest,
        query_fingerprint=fingerprint,
        retrieval_id=pkg_retrieval_id,
        tenant_id=tenant_id,
        package_signature=signature,
    )


