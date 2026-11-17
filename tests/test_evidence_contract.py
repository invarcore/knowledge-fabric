# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for standalone RFC 8785 evidence contract verifier in Knowledge Fabric."""

import time
from datetime import datetime, timezone

import pytest

from knowledge_fabric.evidence_contract import (
    EvidenceContractError,
    _normalize_jcs,
    canonical_json_bytes,
    compute_chunk_hash,
    compute_package_digest,
    length_prefixed_utf8,
    sign_evidence_contract_rfc8785,
    verify_evidence_contract_rfc8785,
)

SECRET_KEY = "test-secret-key-32bytes-long-strict"


def test_evidence_contract_pattern1_chunks_lifecycle() -> None:
    uri = "docs/security_policy.md"
    snippet = "All admin logins require FIDO2 hardware tokens."
    chash = compute_chunk_hash(uri, snippet)

    payload = {
        "tenant_id": "tenant-corp",
        "chunks": [
            {
                "chunk_id": "chk-001",
                "source_uri": uri,
                "snippet": snippet,
                "content_hash": chash,
            }
        ],
    }

    sig = sign_evidence_contract_rfc8785(payload, SECRET_KEY)
    assert len(sig) == 64

    # Verification passes
    assert verify_evidence_contract_rfc8785(payload, sig, SECRET_KEY) is True

    # Tampered signature fails closed
    bad_sig = "0" * 64
    with pytest.raises(EvidenceContractError, match="signature verification failed"):
        verify_evidence_contract_rfc8785(payload, bad_sig, SECRET_KEY)

    # Tampered snippet fails closed
    tampered_payload = dict(payload)
    tampered_payload["chunks"] = [
        {
            "chunk_id": "chk-001",
            "source_uri": uri,
            "snippet": "TAMPERED: All logins require nothing.",
            "content_hash": chash,
        }
    ]
    with pytest.raises(EvidenceContractError, match="Provenance hash mismatch"):
        verify_evidence_contract_rfc8785(tampered_payload, sig, SECRET_KEY)


def test_evidence_contract_pattern2_envelope_lifecycle() -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    uri1 = "docs/a.txt"
    snip1 = "Content A"
    h1 = compute_chunk_hash(uri1, snip1)

    digest = compute_package_digest([h1])
    payload = {
        "retrieval_id": "ret-1234",
        "tenant_id": "tenant-corp",
        "query_fingerprint": "fp-abcdef",
        "provenance_digest": digest,
        "timestamp_utc": now_iso,
        "items": [
            {
                "chunk_id": 1,
                "document_uri": uri1,
                "snippet": snip1,
                "provenance_hash": h1,
            }
        ],
    }

    sig = sign_evidence_contract_rfc8785(payload, SECRET_KEY)
    assert verify_evidence_contract_rfc8785(payload, sig, SECRET_KEY, max_age_seconds=60.0) is True

    # Expiry verification
    with pytest.raises(EvidenceContractError, match="Evidence payload expired"):
        verify_evidence_contract_rfc8785(payload, sig, SECRET_KEY, max_age_seconds=0.00001)
        time.sleep(0.01)


def test_evidence_contract_validation_errors() -> None:
    payload = {"retrieval_id": "r1"}

    # Missing signature
    with pytest.raises(EvidenceContractError, match="signature is mandatory"):
        verify_evidence_contract_rfc8785(payload, "", SECRET_KEY)

    with pytest.raises(EvidenceContractError, match="signature is mandatory"):
        verify_evidence_contract_rfc8785(payload, "   ", SECRET_KEY)

    # Missing secret key
    with pytest.raises(EvidenceContractError, match="secret_key is mandatory"):
        verify_evidence_contract_rfc8785(payload, "abc", "")

    with pytest.raises(EvidenceContractError, match="secret_key is mandatory"):
        verify_evidence_contract_rfc8785(payload, "abc", "   ")

    # Missing tenant id in Pattern 1
    with pytest.raises(EvidenceContractError, match="Tenant ID is mandatory"):
        sign_evidence_contract_rfc8785({"chunks": []}, SECRET_KEY)

    # Malformed timestamp
    bad_ts_payload = {
        "retrieval_id": "r1",
        "timestamp_utc": "invalid-timestamp",
    }
    with pytest.raises(EvidenceContractError, match="Malformed evidence timestamp"):
        verify_evidence_contract_rfc8785(bad_ts_payload, "abc", SECRET_KEY, max_age_seconds=10.0)


def test_evidence_contract_jcs_and_digest_edge_cases() -> None:
    # Package digest empty
    assert compute_package_digest([]) == ""

    # Length prefixed UTF-8 multibyte
    prefixed = length_prefixed_utf8("café")
    assert prefixed == "5:café"

    # Canonical JCS JSON bytes
    raw = {"b": 2, "a": 1, "float_val": 4.0, "zero": 0.0, "flag": True, "none_val": None, "list": [1, "x"]}
    b = canonical_json_bytes(raw)
    assert b == b'{"a":1,"b":2,"flag":true,"float_val":4,"list":[1,"x"],"none_val":null,"zero":0}'

    # JCS float edge cases
    assert _normalize_jcs(3.14) == 3.14
    with pytest.raises(EvidenceContractError, match="forbids NaN"):
        _normalize_jcs(float("nan"))
    with pytest.raises(EvidenceContractError, match="forbids NaN"):
        _normalize_jcs(float("inf"))

    # Bytes secret key & naive timestamp without tzinfo
    naive_ts_payload = {
        "retrieval_id": "r2",
        "timestamp_utc": "2026-11-17T12:00:00",
    }
    b_sig = sign_evidence_contract_rfc8785(naive_ts_payload, b"secret-bytes-12345678901234567890")
    assert verify_evidence_contract_rfc8785(naive_ts_payload, b_sig, b"secret-bytes-12345678901234567890") is True

    # Empty secret key in signing
    with pytest.raises(EvidenceContractError, match="Signing key is mandatory"):
        sign_evidence_contract_rfc8785(naive_ts_payload, "")
