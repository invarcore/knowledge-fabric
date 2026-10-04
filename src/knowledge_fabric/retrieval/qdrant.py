# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Qdrant remote vector database retrieval store adapter.

Enables Knowledge Fabric to query Qdrant clusters over REST API for 50M+ scale,
with zero external dependencies (urllib only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from knowledge_fabric.retrieval.models import RetrievalHit


@dataclass
class QdrantRetrievalStore:
    """RetrievalStore implementation querying remote Qdrant collections.

    Attributes:
        url: Qdrant cluster REST URL (e.g. "http://localhost:6333" or cloud endpoint)
        collection: Collection name (default: "knowledge_fabric")
        api_key: Optional API key for Qdrant Cloud
    """

    url: str = "http://localhost:6333"
    collection: str = "knowledge_fabric"
    api_key: str = ""

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "knowledge-fabric-qdrant-store/1.0",
        }
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    def _request(self, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        req_url = f"{self.url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload else None
        req = Request(req_url, headers=self._headers(), data=data, method=method)
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def vector_search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        tenant_id: str = "default",
    ) -> list[RetrievalHit]:
        """Query Qdrant points with vector similarity and tenant isolation filter."""
        must_conditions: list[dict[str, Any]] = [
            {"key": "tenant_id", "match": {"value": tenant_id}}
        ]

        if filters:
            for k, v in filters.items():
                must_conditions.append({"key": f"metadata.{k}", "match": {"value": v}})

        search_payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
            "filter": {"must": must_conditions},
        }

        path = f"/collections/{quote(self.collection)}/points/search"
        try:
            res = self._request(path, method="POST", payload=search_payload)
        except Exception:
            return []

        hits: list[RetrievalHit] = []
        for point in res.get("result", []):
            payload = point.get("payload", {})
            hits.append(
                RetrievalHit(
                    chunk_id=int(point.get("id", 0)),
                    document_uri=payload.get("document_uri", ""),
                    source_type=payload.get("source_type", "qdrant"),
                    snippet=payload.get("chunk_text", ""),
                    score=float(point.get("score", 0.0)),
                    metadata=payload.get("metadata", {}),
                )
            )
        return hits

    def lexical_search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        tenant_id: str = "default",
    ) -> list[RetrievalHit]:
        """Perform text payload query against Qdrant scroll API."""
        must_conditions: list[dict[str, Any]] = [
            {"key": "tenant_id", "match": {"value": tenant_id}},
            {"key": "chunk_text", "match": {"text": query_text}},
        ]

        if filters:
            for k, v in filters.items():
                must_conditions.append({"key": f"metadata.{k}", "match": {"value": v}})

        scroll_payload = {
            "limit": top_k,
            "with_payload": True,
            "filter": {"must": must_conditions},
        }

        path = f"/collections/{quote(self.collection)}/points/scroll"
        try:
            res = self._request(path, method="POST", payload=scroll_payload)
        except Exception:
            return []

        hits: list[RetrievalHit] = []
        for point in res.get("result", {}).get("points", []):
            payload = point.get("payload", {})
            hits.append(
                RetrievalHit(
                    chunk_id=int(point.get("id", 0)),
                    document_uri=payload.get("document_uri", ""),
                    source_type=payload.get("source_type", "qdrant"),
                    snippet=payload.get("chunk_text", ""),
                    score=1.0,  # Match indicator
                    metadata=payload.get("metadata", {}),
                )
            )
        return hits

    def health_check(self) -> dict[str, object]:
        """Verify Qdrant cluster reachability and collection status."""
        try:
            path = f"/collections/{quote(self.collection)}"
            res = self._request(path, method="GET")
            status = res.get("result", {}).get("status", "unknown")
            vectors_count = res.get("result", {}).get("vectors_count", 0)
            return {
                "backend": "qdrant",
                "status": "connected",
                "collection_status": status,
                "vectors_count": vectors_count,
            }
        except Exception as exc:
            return {
                "backend": "qdrant",
                "status": "error",
                "error": str(exc),
            }
