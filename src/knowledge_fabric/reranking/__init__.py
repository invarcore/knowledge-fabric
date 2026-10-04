# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Reranking module — pluggable post-retrieval relevance scoring.

Three implementations are provided:

    PassthroughReranker   — no-op (default, preserves RRF ordering, zero deps)
    CrossEncoderReranker  — local sentence-transformers cross-encoder (recommended default)
    CohereReranker        — Cohere Rerank API (best quality, requires API key)

Configuration via environment variables:
    RERANKER=none           → PassthroughReranker (no change to RRF results)
    RERANKER=cross_encoder  → CrossEncoderReranker (local, ~500MB model, no API key)
    RERANKER=cohere         → CohereReranker (API, free tier, requires COHERE_API_KEY)

Quick-start (cross-encoder):
    pip install sentence-transformers   # or add to pyproject.toml extras
    export RERANKER=cross_encoder

Quick-start (Cohere):
    export RERANKER=cohere COHERE_API_KEY=your-key

If RERANKER is not set, PassthroughReranker is used — retrieval still works
via RRF, just without neural reranking.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from knowledge_fabric.retrieval.models import RetrievalHit


class Reranker(Protocol):
    """Interface for post-retrieval rerankers."""

    def rerank(self, query_text: str, hits: list[RetrievalHit], top_n: int) -> list[RetrievalHit]:  # pragma: no cover
        """Return top_n hits re-ordered by relevance to query_text."""


@dataclass(slots=True)
class PassthroughReranker:
    """No-op reranker — returns hits in their original RRF order.

    Use this when you want the reranker interface wired in but don't want
    to add model dependencies yet.
    """

    def rerank(self, query_text: str, hits: list[RetrievalHit], top_n: int) -> list[RetrievalHit]:
        return hits[:top_n]


@dataclass
class CrossEncoderReranker:
    """Local neural reranker using sentence-transformers cross-encoders.

    Runs entirely on-device — no API key, no network calls, fully private.
    Adds ~500MB model download on first use.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
        Fast, lightweight (22M params), strong MRR on MS MARCO.

    Prerequisites:
        pip install sentence-transformers   # or add to pyproject.toml extras

    Environment variables:
        CROSS_ENCODER_MODEL  default: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    _model_name: str = field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    _model: object = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_env(cls) -> CrossEncoderReranker:
        model_name = os.environ.get(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        return cls(_model_name=model_name)

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker.\n"
                    "Install it with: pip install sentence-transformers\n"
                    "Or switch to the Cohere reranker: RERANKER=cohere"
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query_text: str, hits: list[RetrievalHit], top_n: int) -> list[RetrievalHit]:
        if not hits:
            return []
        model = self._get_model()
        pairs = [(query_text, hit.chunk_text) for hit in hits]
        scores: list[float] = model.predict(pairs).tolist()  # type: ignore[union-attr]
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [hit for hit, _ in ranked[:top_n]]


@dataclass(slots=True)
class CohereReranker:
    """Reranker using the Cohere Rerank API.

    Best quality reranking available via API. Free tier: 1000 rerank calls/month.
    No model download required — all inference runs on Cohere's infrastructure.

    Environment variables:
        COHERE_API_KEY      (required) — https://dashboard.cohere.com/api-keys
        COHERE_RERANK_MODEL default: rerank-english-v3.0
    """

    _api_key: str
    _model: str

    @classmethod
    def from_env(cls) -> CohereReranker:
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise OSError("COHERE_API_KEY is required for CohereReranker")
        model = os.environ.get("COHERE_RERANK_MODEL", "rerank-english-v3.0")
        return cls(_api_key=api_key, _model=model)

    def rerank(self, query_text: str, hits: list[RetrievalHit], top_n: int) -> list[RetrievalHit]:
        if not hits:
            return []
        documents = [hit.chunk_text for hit in hits]
        body = json.dumps({
            "model": self._model,
            "query": query_text,
            "documents": documents,
            "top_n": top_n,
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.cohere.com/v2/rerank",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "X-Client-Name": "knowledge-fabric",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

        reranked: list[RetrievalHit] = []
        for item in result["results"]:
            original_hit = hits[item["index"]]
            reranked.append(RetrievalHit(
                chunk_id=original_hit.chunk_id,
                document_id=original_hit.document_id,
                document_uri=original_hit.document_uri,
                chunk_index=original_hit.chunk_index,
                chunk_text=original_hit.chunk_text,
                score=float(item["relevance_score"]),
                source=f"reranked_cohere:{original_hit.source}",
                metadata=original_hit.metadata,
            ))
        return reranked


def build_reranker(reranker_name: str = "") -> Reranker:
    """Factory that reads RERANKER env var and returns the right reranker.

    RERANKER=cross_encoder  → CrossEncoderReranker (local, recommended default)
    RERANKER=cohere         → CohereReranker (API, best quality)
    RERANKER=none or unset  → PassthroughReranker (no-op)
    """
    name = (reranker_name or os.environ.get("RERANKER", "none")).lower().strip()
    if name == "cross_encoder":
        return CrossEncoderReranker.from_env()
    if name == "cohere":
        return CohereReranker.from_env()
    return PassthroughReranker()
