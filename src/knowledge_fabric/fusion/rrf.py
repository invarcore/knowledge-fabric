# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Reciprocal rank fusion for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_fabric.retrieval.models import RetrievalHit


@dataclass(slots=True)
class HybridHit:
    """Fused retrieval hit with trace metadata."""

    hit: RetrievalHit
    fused_score: float
    sources: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    lexical_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
    *,
    k: int = 60,
    lexical_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[HybridHit]:
    """Fuse lexical and vector results by chunk id."""
    combined: dict[int, HybridHit] = {}

    _accumulate(combined, lexical_hits, "lexical", k, lexical_weight)
    _accumulate(combined, vector_hits, "vector", k, vector_weight)

    return sorted(
        combined.values(),
        key=lambda item: (item.fused_score, item.hit.score),
        reverse=True,
    )


def _accumulate(
    target: dict[int, HybridHit],
    hits: list[RetrievalHit],
    source_name: str,
    k: int,
    weight: float,
) -> None:
    for rank, hit in enumerate(hits, start=1):
        increment = weight * (1.0 / (k + rank))
        if hit.chunk_id not in target:
            target[hit.chunk_id] = HybridHit(hit=hit, fused_score=0.0, sources=[])

        target_hit = target[hit.chunk_id]
        target_hit.fused_score += increment
        if source_name not in target_hit.sources:
            target_hit.sources.append(source_name)
