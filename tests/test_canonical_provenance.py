# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive regression & multi-byte test suite for canonical RFC 8785 provenance in Knowledge Fabric."""

import pytest
from knowledge_fabric.evidence.provenance import (
    ProvenanceSigningError,
    compute_canonical_provenance_bytes,
    create_provenance_signature,
)


def test_kf_accepts_both_source_uri_and_document_uri():
    c_source = [{"chunk_id": "c1", "source_uri": "s3://bucket/doc.md", "content_hash": "a" * 64}]
    c_doc = [{"chunk_id": "c1", "document_uri": "s3://bucket/doc.md", "content_hash": "a" * 64}]

    b_source = compute_canonical_provenance_bytes("t1", c_source)
    b_doc = compute_canonical_provenance_bytes("t1", c_doc)
    assert b_source == b_doc


def test_kf_rejects_unkeyed_or_whitespace_signing_key():
    chunks = [{"chunk_id": "c1", "source_uri": "s3://b/d.md", "content_hash": "b" * 64}]
    for bad_key in ["", "   "]:
        with pytest.raises(ProvenanceSigningError, match="Signing key is mandatory"):
            create_provenance_signature(chunks, secret_key=bad_key, tenant_id="t1")


def test_kf_prevents_delimiter_collision_and_handles_multibyte_utf8():
    c1 = [
        {"chunk_id": "1", "source_uri": "s3://t/caf\u00e9.md", "content_hash": "ab:c"},
        {"chunk_id": "2", "source_uri": "s3://t/caf\u00e9.md", "content_hash": "d"},
    ]
    c2 = [
        {"chunk_id": "1", "source_uri": "s3://t/caf\u00e9.md", "content_hash": "ab"},
        {"chunk_id": "2", "source_uri": "s3://t/caf\u00e9.md", "content_hash": "c:d"},
    ]
    b1 = compute_canonical_provenance_bytes("tenant_1", c1)
    b2 = compute_canonical_provenance_bytes("tenant_1", c2)
    assert b1 != b2
    assert b'"15:s3://t/caf\xc3\xa9.md"' in b1


def test_kf_missing_field_raises():
    bad_chunks = [{"chunk_id": "c1", "source_uri": "s3://b/d.md"}]  # missing content_hash
    with pytest.raises(ProvenanceSigningError, match="Missing required provenance field: content_hash"):
        compute_canonical_provenance_bytes("t1", bad_chunks)


def test_kf_valid_signature_emission():
    chunks = [{"chunk_id": "c1", "source_uri": "s3://b/d.md", "content_hash": "a" * 64}]
    sig = create_provenance_signature(chunks, secret_key="my-key", tenant_id="t1")
    assert isinstance(sig, str)
    assert len(sig) == 64


def test_kf_extract_field_from_object():
    class ChunkObj:
        def __init__(self, chunk_id: str, document_uri: str, content_hash: str):
            self.chunk_id = chunk_id
            self.document_uri = document_uri
            self.content_hash = content_hash

    chunks = [ChunkObj("c1", "s3://b/d.md", "a" * 64)]
    b = compute_canonical_provenance_bytes("t1", chunks)
    assert b'"chunk_id":"2:c1"' in b
