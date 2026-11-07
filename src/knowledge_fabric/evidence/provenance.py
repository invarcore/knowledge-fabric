# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Evidence provenance calculation and signing for Knowledge Fabric.

Enforces RFC 8785 byte-length provenance and mandatory HMAC signing (INT-01/02).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Union


class ProvenanceSigningError(ValueError):
    """Raised when signing key or tenant context is missing during provenance emission."""


def _length_prefixed_utf8(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    utf8_bytes = normalized.encode("utf-8")
    return f"{len(utf8_bytes)}:{normalized}"


def _extract_field(item: Union[Mapping[str, Any], Any], primary: str, fallback: str = "") -> str:
    if isinstance(item, Mapping):
        val = item.get(primary) or (item.get(fallback) if fallback else None)
    else:
        val = getattr(item, primary, None) or (getattr(item, fallback, None) if fallback else None)
    if val is None or not str(val).strip():
        raise ProvenanceSigningError(f"Missing required provenance field: {primary}")
    return str(val)


def compute_canonical_provenance_bytes(
    tenant_id: str, chunks: Sequence[Union[Mapping[str, Any], Any]]
) -> bytes:
    """Emit byte-identical RFC 8785 provenance payload matching intent-fabric consumer."""
    if not tenant_id or not tenant_id.strip():
        raise ProvenanceSigningError("Tenant ID is mandatory for provenance emission.")
    norm_tenant = unicodedata.normalize("NFC", tenant_id.strip())
    canonical_items = [
        {
            "chunk_id": _length_prefixed_utf8(_extract_field(c, "chunk_id")),
            "content_hash": _length_prefixed_utf8(_extract_field(c, "content_hash")),
            "source_uri": _length_prefixed_utf8(_extract_field(c, "source_uri", "document_uri")),
        }
        for c in chunks
    ]
    payload = {
        "chunks": canonical_items,
        "tenant_id": _length_prefixed_utf8(norm_tenant),
        "v": "rfc8785-v1",
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def create_provenance_signature(
    chunks: Sequence[Union[Mapping[str, Any], Any]],
    secret_key: str,
    tenant_id: str,
) -> str:
    """Generate mandatory keyed HMAC-SHA256 provenance signature in Knowledge Fabric."""
    if not secret_key or not secret_key.strip():
        raise ProvenanceSigningError(
            "Signing key is mandatory; unkeyed digest fallback is forbidden."
        )
    canonical_bytes = compute_canonical_provenance_bytes(tenant_id, chunks)
    return hmac.new(
        secret_key.encode("utf-8"), canonical_bytes, hashlib.sha256
    ).hexdigest()
