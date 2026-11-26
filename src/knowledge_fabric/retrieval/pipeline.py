# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""End-to-end retrieval orchestration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, is_dataclass, replace
import os
import re
from time import perf_counter
from typing import Any, Mapping

from knowledge_fabric.chunking.case_guard import rerank_chunks_for_case_bench
from knowledge_fabric.db.audit import AuditLogger
from knowledge_fabric.embeddings import EmbeddingProvider
from knowledge_fabric.evidence.models import EvidencePackage, build_evidence_package
from knowledge_fabric.fusion.rrf import HybridHit, reciprocal_rank_fusion
from knowledge_fabric.reranking import PassthroughReranker, Reranker
from knowledge_fabric.retrieval.models import RetrievalHit
from knowledge_fabric.retrieval.store import RetrievalStore


@dataclass(slots=True)
class RetrievalTrace:
    """Debug and observability details for one retrieval call."""

    query_text: str
    top_k: int
    lexical_count: int
    vector_count: int
    fused_count: int
    latency_ms: int
    mode: str = "hybrid"
    strategy: str = "hybrid"
    lexical_latency_ms: int = 0
    vector_latency_ms: int = 0
    lexical_status: str = "ok"
    vector_status: str = "ok"
    is_degraded: bool = False
    warnings: list[str] = field(default_factory=list)


_DEFAULT_MAX_QUERY_LENGTH = int(os.environ.get("KNOWLEDGE_MAX_QUERY_LENGTH", "4000"))
_DEFAULT_MAX_TOP_K = int(os.environ.get("KNOWLEDGE_MAX_TOP_K", "100"))
_DEFAULT_MAX_RERANK_CANDIDATES = int(os.environ.get("KNOWLEDGE_MAX_RERANK_CANDIDATES", "100"))

# Supports standard enterprise formats: alphanumeric, hyphens, underscores, dots, colons, and @ (emails, domains, URNs, UUIDs)
_TENANT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_.:@-]{1,128}$")


def validate_tenant_id(tenant_id: str | None) -> str | None:
    """Validate that tenant_id adheres to safe enterprise namespace format.

    Supports UUIDs, domain names, emails, and namespaced IDs while preventing
    SQL injection, path traversal (e.g. '../'), and control character injection.
    Can be bypassed if KNOWLEDGE_STRICT_TENANT_CHECK=false.
    """
    if tenant_id is None:
        return None
    cleaned = tenant_id.strip()
    if not cleaned:
        return None

    # Allow disabling strict check via environment if custom enterprise naming schemes are used
    if os.environ.get("KNOWLEDGE_STRICT_TENANT_CHECK", "true").lower() in ("false", "0", "off"):
        # Still sanitize against null bytes and control chars
        return "".join(ch for ch in cleaned if ord(ch) >= 32 and ch != "\x7f")[:128]

    if not _TENANT_ID_REGEX.match(cleaned) or ".." in cleaned:
        raise ValueError(
            f"Invalid tenant_id format: '{tenant_id!r}'. Must match ^[a-zA-Z0-9_.:@-]{{1,128}}$ with no path traversal."
        )
    return cleaned


class RetrievalPipeline:
    """Coordinates lexical/vector retrieval, fusion, packaging, and audit logging."""

    def __init__(
        self,
        *,
        retrieval_store: RetrievalStore,
        embedding_provider: EmbeddingProvider,
        audit_logger: AuditLogger | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        vector_weight: float = 1.0,
        max_query_length: int | None = None,
        max_top_k: int | None = None,
        max_rerank_candidates: int | None = None,
    ) -> None:
        self._retrieval_store = retrieval_store
        self._embedding_provider = embedding_provider
        self._audit_logger = audit_logger
        self._reranker: Reranker = reranker if reranker is not None else PassthroughReranker()
        self._rrf_k = rrf_k
        self._lexical_weight = lexical_weight
        self._vector_weight = vector_weight
        self._max_query_length = (
            max_query_length if max_query_length is not None else _DEFAULT_MAX_QUERY_LENGTH
        )
        self._max_top_k = max_top_k if max_top_k is not None else _DEFAULT_MAX_TOP_K
        self._max_rerank_candidates = (
            max_rerank_candidates
            if max_rerank_candidates is not None
            else _DEFAULT_MAX_RERANK_CANDIDATES
        )

    def retrieve_evidence(
        self,
        *,
        query_text: str,
        top_k: int = 10,
        source_type: str | None = None,
        trace_id: str | None = None,
        tenant_id: str | None = None,
        mode: str = "hybrid",
        fail_closed: bool = False,
    ) -> EvidencePackage:
        package, _ = self.retrieve_with_trace(
            query_text=query_text,
            top_k=top_k,
            source_type=source_type,
            trace_id=trace_id,
            tenant_id=tenant_id,
            mode=mode,
            fail_closed=fail_closed,
        )
        return package

    def retrieve_with_trace(
        self,
        *,
        query_text: str,
        top_k: int = 10,
        source_type: str | None = None,
        trace_id: str | None = None,
        tenant_id: str | None = None,
        mode: str = "hybrid",
        fail_closed: bool = False,
        skip_audit: bool = False,
    ) -> tuple[EvidencePackage, RetrievalTrace]:
        # Security hardening: Configurable input bounding against DoS & Injection
        safe_query = (query_text or "").strip()[: self._max_query_length]
        safe_top_k = max(1, min(top_k, self._max_top_k))
        safe_tenant = validate_tenant_id(tenant_id)

        normalized_mode = (mode or "hybrid").lower().strip()
        if normalized_mode not in ("hybrid", "lexical", "vector"):
            raise ValueError(
                f"Invalid retrieval mode: {mode!r}. Must be 'hybrid', 'lexical', or 'vector'."
            )

        start = perf_counter()
        lexical_hits: list[RetrievalHit] = []
        vector_hits: list[RetrievalHit] = []
        lexical_status = "skipped"
        vector_status = "skipped"
        lexical_latency_ms = 0
        vector_latency_ms = 0
        warnings: list[str] = []
        is_degraded = False

        # --- Lexical Leg ---
        if normalized_mode in ("lexical", "hybrid"):
            t0 = perf_counter()
            try:
                lexical_hits = self._retrieval_store.lexical_search(
                    safe_query, top_k=safe_top_k, source_type=source_type, tenant_id=safe_tenant
                )
                lexical_status = "ok"
            except Exception as exc:
                lexical_status = "error"
                warnings.append(f"lexical_leg_failed: {exc}")
                if fail_closed or normalized_mode == "lexical":
                    raise
            finally:
                lexical_latency_ms = int((perf_counter() - t0) * 1000)

        # --- Vector Leg ---
        if normalized_mode in ("vector", "hybrid"):
            t0 = perf_counter()
            try:
                vector_query = self._embedding_provider.embed_texts([safe_query])[0]
                vector_hits = self._retrieval_store.vector_search(
                    vector_query, top_k=safe_top_k, source_type=source_type, tenant_id=safe_tenant
                )
                vector_status = "ok"
            except Exception as exc:
                vector_status = "error"
                warnings.append(f"vector_leg_failed: {exc}")
                if fail_closed or normalized_mode == "vector":
                    raise
            finally:
                vector_latency_ms = int((perf_counter() - t0) * 1000)

        # --- Fusion / Strategy Determination ---
        if normalized_mode == "hybrid":
            if lexical_status == "ok" and vector_status == "ok":
                effective_strategy = "hybrid"
                fused = reciprocal_rank_fusion(
                    lexical_hits=lexical_hits,
                    vector_hits=vector_hits,
                    k=self._rrf_k,
                    lexical_weight=self._lexical_weight,
                    vector_weight=self._vector_weight,
                )
            elif lexical_status == "ok" and vector_status == "error":
                is_degraded = True
                effective_strategy = "degraded_lexical"
                fused = [
                    HybridHit(hit=h, fused_score=h.score, sources=["lexical"]) for h in lexical_hits
                ]
            elif lexical_status == "error" and vector_status == "ok":
                is_degraded = True
                effective_strategy = "degraded_vector"
                fused = [
                    HybridHit(hit=h, fused_score=h.score, sources=["vector"]) for h in vector_hits
                ]
            else:
                is_degraded = True
                effective_strategy = "failed"
                fused = []
        elif normalized_mode == "lexical":
            effective_strategy = "lexical"
            fused = [
                HybridHit(hit=h, fused_score=h.score, sources=["lexical"]) for h in lexical_hits
            ]
        elif normalized_mode == "vector":
            effective_strategy = "vector"
            fused = [HybridHit(hit=h, fused_score=h.score, sources=["vector"]) for h in vector_hits]
        else:
            effective_strategy = normalized_mode
            fused = []

        if not fused and not warnings:
            warnings.append("empty_results")

        # Cap candidate pool for reranking to protect CPU/GPU from compute explosion
        candidates_to_rerank = fused[: self._max_rerank_candidates]
        # Apply reranker (no-op if PassthroughReranker)
        reranked = self._reranker.rerank(safe_query, candidates_to_rerank, top_n=safe_top_k)

        latency_ms = int((perf_counter() - start) * 1000)

        summary_extra: dict[str, object] = {
            "strategy": effective_strategy,
            "requested_mode": normalized_mode,
            "is_degraded": is_degraded,
            "warnings": warnings,
            "legs": {
                "lexical": {
                    "status": lexical_status,
                    "count": len(lexical_hits),
                    "latency_ms": lexical_latency_ms,
                },
                "vector": {
                    "status": vector_status,
                    "count": len(vector_hits),
                    "latency_ms": vector_latency_ms,
                },
            },
        }
        package = build_evidence_package(
            safe_query,
            reranked,
            summary_extra=summary_extra,
            tenant_id=safe_tenant,
            mode=normalized_mode,
        )

        trace = RetrievalTrace(
            query_text=safe_query,
            top_k=safe_top_k,
            lexical_count=len(lexical_hits),
            vector_count=len(vector_hits),
            fused_count=len(reranked),
            latency_ms=latency_ms,
            mode=normalized_mode,
            strategy=effective_strategy,
            lexical_latency_ms=lexical_latency_ms,
            vector_latency_ms=vector_latency_ms,
            lexical_status=lexical_status,
            vector_status=vector_status,
            is_degraded=is_degraded,
            warnings=warnings,
        )

        if self._audit_logger is not None and not skip_audit:
            try:
                self._audit_logger.log_retrieval(
                    query_text=query_text,
                    top_k=top_k,
                    retrieval_mode=effective_strategy,
                    result_count=len(reranked),
                    latency_ms=latency_ms,
                    trace_id=trace_id,
                    tenant_id=safe_tenant,
                    details=self._trace_to_dict(trace),
                )
            except Exception as audit_exc:
                # Do not allow telemetry or audit logging failures to crash degraded retrieval
                warnings.append(f"audit_logging_failed: {audit_exc}")
        return package, trace

    def explain_retrieval(
        self,
        *,
        query_text: str,
        top_k: int = 10,
        source_type: str | None = None,
        tenant_id: str | None = None,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        package, trace = self.retrieve_with_trace(
            query_text=query_text,
            top_k=top_k,
            source_type=source_type,
            tenant_id=tenant_id,
            mode=mode,
            skip_audit=True,
        )
        return {
            "query_text": query_text,
            "top_k": top_k,
            "mode": trace.mode,
            "strategy": trace.strategy,
            "is_degraded": trace.is_degraded,
            "warnings": trace.warnings,
            "trace": self._trace_to_dict(trace),
            "result_chunk_ids": [item.chunk_id for item in package.items],
            "sources": package.retrieval_summary.get("sources", []),
            "legs": package.retrieval_summary.get("legs", {}),
        }

    @staticmethod
    def _trace_to_dict(trace: RetrievalTrace) -> dict[str, Any]:
        return {
            "query_text": trace.query_text,
            "top_k": trace.top_k,
            "mode": trace.mode,
            "strategy": trace.strategy,
            "lexical_count": trace.lexical_count,
            "vector_count": trace.vector_count,
            "fused_count": trace.fused_count,
            "latency_ms": trace.latency_ms,
            "lexical_latency_ms": trace.lexical_latency_ms,
            "vector_latency_ms": trace.vector_latency_ms,
            "lexical_status": trace.lexical_status,
            "vector_status": trace.vector_status,
            "is_degraded": trace.is_degraded,
            "warnings": trace.warnings,
        }

    def _execute_hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_type: str | None = None,
        tenant_id: str | None = None,
        mode: str = "hybrid",
        fail_closed: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute hybrid/lexical/vector search and return candidate hits."""
        package, _ = self.retrieve_with_trace(
            query_text=query,
            top_k=top_k,
            source_type=source_type,
            tenant_id=tenant_id,
            mode=mode,
            fail_closed=fail_closed,
            skip_audit=True,
        )
        return list(package.items)

    def retrieve(
        self,
        query: str = "",
        *,
        query_text: str | None = None,
        top_k: int = 10,
        enable_case_guard: bool = False,
        quarantine_insecure: bool = False,
        quarantine_tests: bool | None = None,
        award_safeguard_boost: bool = False,
        safeguard_boost: float = 1.35,
        insecure_penalty: float | None = None,
        source_type: str | None = None,
        tenant_id: str | None = None,
        mode: str = "hybrid",
        fail_closed: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        """Retrieve candidates and optionally apply CASE-Guard reranking/quarantining.

        Invariants:
        1. Hits are mapped strictly by 'source_index' emitted by case_guard (P1),
           never by 'chunk_id', so hits with chunk_id=None or duplicate IDs are preserved.
        2. quarantine_insecure defaults to False per-query so audit queries are not blinded.
        3. Existing hit.metadata keys are strictly preserved when attaching metadata["case_guard"].
        """
        effective_query = query_text if query_text is not None else query
        try:
            raw_hits = self._execute_hybrid_search(
                effective_query,
                top_k=top_k,
                source_type=source_type,
                tenant_id=tenant_id,
                mode=mode,
                fail_closed=fail_closed,
                **kwargs,
            )
        except TypeError:
            raw_hits = self._execute_hybrid_search(effective_query, top_k=top_k)

        if not enable_case_guard:
            return list(raw_hits[:top_k])

        reranked_chunks = rerank_chunks_for_case_bench(
            raw_hits,
            quarantine_insecure=quarantine_insecure,
            quarantine_tests=quarantine_tests,
            award_safeguard_boost=award_safeguard_boost,
            safeguard_boost=safeguard_boost,
            insecure_penalty=insecure_penalty,
        )

        results: list[Any] = []
        for chunk in reranked_chunks:
            src_idx = chunk.get("source_index")
            if src_idx is None or src_idx < 0 or src_idx >= len(raw_hits):
                continue
            orig_hit = raw_hits[src_idx]

            raw_meta = getattr(orig_hit, "metadata", None)
            if isinstance(raw_meta, Mapping):
                meta = dict(raw_meta)
            elif isinstance(orig_hit, Mapping) and isinstance(orig_hit.get("metadata"), Mapping):
                meta = dict(orig_hit["metadata"])
            else:
                meta = {}

            case_guard_dict = {
                "is_security_safeguard": bool(chunk.get("is_security_safeguard", False)),
                "has_insecure_exemplar": bool(chunk.get("has_insecure_exemplar", False)),
                "safeguard_signals": list(chunk.get("safeguard_signals", ())),
                "insecure_signals": list(chunk.get("insecure_signals", ())),
                "adjusted_score": float(chunk.get("adjusted_score", 0.0)),
            }
            meta["case_guard"] = case_guard_dict

            adjusted_score = float(chunk.get("adjusted_score", 0.0))

            if is_dataclass(orig_hit) and not isinstance(orig_hit, type):
                fields_set = {f.name for f in orig_hit.__dataclass_fields__.values()}
                kwargs_to_replace: dict[str, Any] = {}
                if "metadata" in fields_set:
                    kwargs_to_replace["metadata"] = meta
                if "score" in fields_set:
                    kwargs_to_replace["score"] = adjusted_score
                updated_hit = replace(orig_hit, **kwargs_to_replace)
            elif isinstance(orig_hit, Mapping):
                updated_hit = dict(orig_hit)
                updated_hit["metadata"] = meta
                updated_hit["score"] = adjusted_score
            else:
                updated_hit = copy.copy(orig_hit)
                if hasattr(updated_hit, "metadata"):
                    updated_hit.metadata = meta
                if hasattr(updated_hit, "score"):
                    updated_hit.score = adjusted_score

            results.append(updated_hit)

        return results[:top_k]
