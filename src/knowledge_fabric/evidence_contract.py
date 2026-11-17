# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Standalone Zero-Dependency Evidence Contract Verifier for Knowledge Fabric.

Implements RFC 8785 Canonical JSON hashing, constant-time HMAC-SHA256 signature
verification, chunk provenance integrity checking, and temporal freshness validation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Union


class EvidenceContractError(ValueError):
    """Raised when evidence contract verification fails closed due to tampering or policy violation."""


def _normalize_jcs(obj: Any) -> Any:
    """Normalize strings (NFC), numbers, mappings, and sequences per RFC 8785 JCS rules."""
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                raise EvidenceContractError("RFC 8785 JSON canonicalization forbids NaN and Infinity.")
            if obj == 0.0:
                return 0
            if obj.is_integer():
                return int(obj)
        return obj
    if isinstance(obj, Mapping):
        return {unicodedata.normalize("NFC", str(k)): _normalize_jcs(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_jcs(x) for x in obj]
    return obj


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize payload into RFC 8785 (JCS) canonical UTF-8 JSON bytes."""
    normalized = _normalize_jcs(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def length_prefixed_utf8(value: str) -> str:
    """Encode string with UTF-8 byte length prefix: '{byte_length}:{nfc_normalized_str}'."""
    norm = unicodedata.normalize("NFC", value)
    utf8_bytes = norm.encode("utf-8")
    return f"{len(utf8_bytes)}:{norm}"


def compute_chunk_hash(source_uri: str, content: str) -> str:
    """Compute deterministic SHA-256 chunk hash using RFC 8785 canonical JSON array."""
    canonical_payload = json.dumps(
        [unicodedata.normalize("NFC", source_uri), unicodedata.normalize("NFC", content)],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def compute_package_digest(chunk_hashes: Sequence[str]) -> str:
    """Compute composite SHA-256 digest over ordered chunk hashes using length-prefixed encoding."""
    if not chunk_hashes:
        return ""
    parts: list[bytes] = []
    for h in chunk_hashes:
        h_bytes = unicodedata.normalize("NFC", h).encode("utf-8")
        parts.append(struct.pack(">I", len(h_bytes)))
        parts.append(h_bytes)
    return hashlib.sha256(b"".join(parts)).hexdigest()


def compute_canonical_contract_bytes(
    payload: Mapping[str, Any],
    tenant_id: str | None = None,
) -> bytes:
    """Emit deterministic RFC 8785 canonical bytes for evidence contract signing and verification."""
    # Pattern 1: Raw chunks structure (provenance.py / INT-01 style)
    if "chunks" in payload:
        raw_chunks = payload.get("chunks", [])
        tid = tenant_id or payload.get("tenant_id", "")
        if not tid or not str(tid).strip():
            raise EvidenceContractError("Tenant ID is mandatory for contract provenance calculation.")
        norm_tenant = unicodedata.normalize("NFC", str(tid).strip())
        canonical_items = []
        for c in raw_chunks:
            cid = str(c.get("chunk_id", ""))
            chash = str(c.get("content_hash", "") or c.get("provenance_hash", ""))
            suri = str(c.get("source_uri", "") or c.get("document_uri", ""))
            canonical_items.append({
                "chunk_id": length_prefixed_utf8(cid),
                "content_hash": length_prefixed_utf8(chash),
                "source_uri": length_prefixed_utf8(suri),
            })
        contract_body = {
            "chunks": canonical_items,
            "tenant_id": length_prefixed_utf8(norm_tenant),
            "v": "rfc8785-v1",
        }
        return json.dumps(
            contract_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    # Pattern 2: EvidencePackage envelope (models.py / INT-02 style)
    retrieval_id = str(payload.get("retrieval_id", ""))
    tid = str(tenant_id if tenant_id is not None else payload.get("tenant_id", "") or "")
    q_fp = str(payload.get("query_fingerprint", ""))
    prov_digest = str(payload.get("provenance_digest", ""))
    ts = str(payload.get("timestamp_utc", "") or payload.get("generated_at", ""))

    envelope = [retrieval_id, tid, q_fp, prov_digest, ts]
    return json.dumps(
        envelope,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sign_evidence_contract_rfc8785(
    payload: Mapping[str, Any],
    secret_key: Union[str, bytes],
    tenant_id: str | None = None,
) -> str:
    """Generate keyed HMAC-SHA256 signature for evidence contract payload."""
    if not secret_key or (isinstance(secret_key, str) and not secret_key.strip()):
        raise EvidenceContractError("Signing key is mandatory; unkeyed fallback is forbidden.")
    key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    canonical_bytes = compute_canonical_contract_bytes(payload, tenant_id=tenant_id)
    return hmac.new(key_bytes, canonical_bytes, hashlib.sha256).hexdigest()


def verify_evidence_contract_rfc8785(
    payload: Mapping[str, Any],
    signature: str,
    secret_key: Union[str, bytes],
    tenant_id: str | None = None,
    max_age_seconds: float | None = None,
) -> bool:
    """Verify evidence contract signature and integrity fail-closed per RFC 8785.

    Validates:
    1. Signature presence and format.
    2. Secret key non-emptiness.
    3. Temporal freshness if timestamp is present and max_age_seconds is specified.
    4. Chunk provenance integrity against reported snippets/URIs.
    5. Constant-time HMAC-SHA256 signature verification over canonical bytes.
    """
    if not signature or not isinstance(signature, str) or not signature.strip():
        raise EvidenceContractError("Evidence signature is mandatory.")

    if not secret_key or (isinstance(secret_key, str) and not secret_key.strip()):
        raise EvidenceContractError("Verification secret_key is mandatory.")

    # Temporal freshness check
    ts_str = payload.get("timestamp_utc") or payload.get("generated_at")
    if max_age_seconds is not None and ts_str:
        try:
            ts_clean = str(ts_str).replace("Z", "+00:00")
            parsed_dt = datetime.fromisoformat(ts_clean)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age = (now - parsed_dt).total_seconds()
            if age > max_age_seconds:
                raise EvidenceContractError(
                    f"Evidence payload expired: age {age:.1f}s exceeds limit of {max_age_seconds:.1f}s."
                )
        except (ValueError, TypeError) as err:
            raise EvidenceContractError(f"Malformed evidence timestamp '{ts_str}': {err}") from err

    # Chunk-level provenance integrity validation
    items = payload.get("items") or payload.get("chunks")
    if items and isinstance(items, Sequence):
        for item in items:
            if isinstance(item, Mapping):
                snippet = item.get("snippet") or item.get("content")
                uri = item.get("document_uri") or item.get("source_uri")
                reported_hash = item.get("provenance_hash") or item.get("content_hash")
                if snippet is not None and uri is not None and reported_hash is not None:
                    expected_hash = compute_chunk_hash(str(uri), str(snippet))
                    if not hmac.compare_digest(str(reported_hash).lower(), expected_hash.lower()):
                        cid = item.get("chunk_id", "unknown")
                        raise EvidenceContractError(
                            f"Provenance hash mismatch on chunk {cid}: expected {expected_hash[:12]}, got {reported_hash[:12]}."
                        )

    expected_sig = sign_evidence_contract_rfc8785(
        payload=payload,
        secret_key=secret_key,
        tenant_id=tenant_id,
    )

    if not hmac.compare_digest(expected_sig.lower(), signature.strip().lower()):
        raise EvidenceContractError("Evidence contract HMAC signature verification failed.")

    return True
