"""Evaluation runner for lexical, vector, and hybrid retrieval.

Computes IR benchmark metrics (Recall@K, Precision@K, MRR, NDCG@K, No-result accuracy)
across categorized evaluation query suites (exact, paraphrase, disagreement, metadata, unanswerable).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from knowledge_fabric.retrieval.models import RetrievalHit

RetrieverResult = list[int] | list[RetrievalHit] | Any
RetrieverFn = Callable[[str, str, int], RetrieverResult]


@dataclass(slots=True)
class EvaluationQuery:
    """One evaluation query specification."""

    id: str
    text: str
    relevant_chunk_ids: list[int]
    category: str = "general"
    expected_document_uris: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    graded_relevance: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class ModeMetrics:
    """Metrics for one retrieval mode (overall and broken down by category)."""

    mode: str
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_10: float
    no_result_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    query_count: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)


class EvaluationReport(dict[str, ModeMetrics]):
    """Container for benchmark evaluation results supporting dict mapping and reporting."""

    def __init__(
        self,
        *,
        dataset_id: str = "custom",
        dataset_version: str = "1.0.0",
        generated_at: str | None = None,
        modes: dict[str, ModeMetrics] | None = None,
        per_query_results: list[dict[str, Any]] | None = None,
        system_info: dict[str, Any] | None = None,
        quality_gate_passed: bool = True,
        quality_gate_failures: list[str] | None = None,
    ) -> None:
        super().__init__(modes or {})
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.generated_at = generated_at or datetime.now(UTC).isoformat()
        self.modes = modes or {}
        self.per_query_results = per_query_results or []
        self.system_info = system_info or {}
        self.quality_gate_passed = quality_gate_passed
        self.quality_gate_failures = quality_gate_failures or []

    def check_quality_gates(
        self,
        *,
        min_mrr: float | None = None,
        min_recall: float | None = None,
        min_ndcg: float | None = None,
        target_mode: str = "hybrid",
    ) -> bool:
        """Check whether the evaluation results satisfy quality gate thresholds."""
        failures: list[str] = []
        metrics = self.modes.get(target_mode)
        if not metrics:
            failures.append(f"Target mode '{target_mode}' was not evaluated.")
        else:
            if min_mrr is not None and metrics.mrr < min_mrr:
                failures.append(
                    f"MRR gate failed for {target_mode}: {metrics.mrr:.3f} < {min_mrr:.3f}"
                )
            if min_recall is not None and metrics.recall_at_10 < min_recall:
                failures.append(
                    f"Recall@10 gate failed for {target_mode}: {metrics.recall_at_10:.3f} < {min_recall:.3f}"
                )
            if min_ndcg is not None and metrics.ndcg_at_10 < min_ndcg:
                failures.append(
                    f"NDCG@10 gate failed for {target_mode}: {metrics.ndcg_at_10:.3f} < {min_ndcg:.3f}"
                )

        self.quality_gate_passed = len(failures) == 0
        self.quality_gate_failures = failures
        return self.quality_gate_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "generated_at": self.generated_at,
            "quality_gate_passed": self.quality_gate_passed,
            "quality_gate_failures": self.quality_gate_failures,
            "system_info": self.system_info,
            "modes": {m: asdict(met) for m, met in self.modes.items()},
            "per_query_results": self.per_query_results,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# Retrieval Evaluation Report: `{self.dataset_id}` (v{self.dataset_version})",
            "",
            f"- **Generated At:** {self.generated_at}",
            f"- **Quality Gate:** {'PASSED' if self.quality_gate_passed else 'FAILED'}",
        ]
        if self.quality_gate_failures:
            lines.append("  - **Failures:**")
            for failure in self.quality_gate_failures:
                lines.append(f"    - ⚠️ {failure}")

        if self.system_info:
            lines.append("- **System Info:**")
            for k, v in self.system_info.items():
                lines.append(f"  - `{k}`: {v}")

        lines.extend([
            "",
            "## Summary Metrics Across Modes",
            "",
            "| Mode | Recall@10 | Precision@5 | MRR | NDCG@10 | No-Result Acc | Avg Latency (ms) | P95 Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for mode_name, metrics in self.modes.items():
            lines.append(
                f"| `{mode_name}` | {metrics.recall_at_10:.3f} | {metrics.precision_at_5:.3f} | "
                f"{metrics.mrr:.3f} | {metrics.ndcg_at_10:.3f} | {metrics.no_result_accuracy:.3f} | "
                f"{metrics.avg_latency_ms:.1f} | {metrics.p95_latency_ms:.1f} |"
            )

        # Per-Category Breakdown
        lines.extend([
            "",
            "## Category Breakdown",
            "",
            "| Mode | Category | Queries | Recall@10 | Precision@5 | MRR | NDCG@10 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for mode_name, metrics in self.modes.items():
            for cat, cat_metrics in metrics.by_category.items():
                lines.append(
                    f"| `{mode_name}` | `{cat}` | {int(cat_metrics.get('count', 0))} | "
                    f"{cat_metrics.get('recall_at_10', 0.0):.3f} | {cat_metrics.get('precision_at_5', 0.0):.3f} | "
                    f"{cat_metrics.get('mrr', 0.0):.3f} | {cat_metrics.get('ndcg_at_10', 0.0):.3f} |"
                )

        lines.append("")
        return "\n".join(lines)


class RetrievalEvaluationRunner:
    """Computes retrieval metrics for lexical, vector, and hybrid modes."""

    def __init__(
        self,
        retriever_fn: RetrieverFn,
        dataset_id: str = "custom",
        dataset_version: str = "1.0.0",
        system_info: dict[str, Any] | None = None,
    ) -> None:
        self._retriever_fn = retriever_fn
        self._dataset_id = dataset_id
        self._dataset_version = dataset_version
        self._system_info = system_info or {}

    def evaluate(
        self,
        *,
        queries: list[EvaluationQuery],
        modes: tuple[str, ...] = ("lexical", "vector", "hybrid"),
        top_k: int = 10,
    ) -> EvaluationReport:
        by_mode: dict[str, ModeMetrics] = {}
        all_query_logs: list[dict[str, Any]] = []

        for mode in modes:
            recall_values: list[float] = []
            precision_values: list[float] = []
            mrr_values: list[float] = []
            ndcg_values: list[float] = []
            no_result_values: list[float] = []
            latencies: list[float] = []

            # Category tracking
            cat_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

            for query in queries:
                t0 = perf_counter()
                raw_result = self._retriever_fn(mode, query.text, top_k)
                elapsed_ms = (perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)

                ranked = _extract_ranked_ids(raw_result)
                relevant = set(query.relevant_chunk_ids)

                is_answerable = len(relevant) > 0

                rec = recall_at_k(ranked, relevant, k=top_k) if is_answerable else 0.0
                prec = precision_at_k(ranked, relevant, k=min(5, top_k)) if is_answerable else 0.0
                mrr = mean_reciprocal_rank(ranked, relevant) if is_answerable else 0.0
                ndcg = ndcg_at_k(
                    ranked,
                    relevant,
                    k=top_k,
                    graded_relevance=query.graded_relevance if query.graded_relevance else None,
                ) if is_answerable else 0.0
                no_res = no_result_accuracy(ranked, relevant)

                if is_answerable:
                    recall_values.append(rec)
                    precision_values.append(prec)
                    mrr_values.append(mrr)
                    ndcg_values.append(ndcg)
                else:
                    no_result_values.append(no_res)

                # Record per-category
                cat = query.category or "general"
                cat_metrics[cat]["count"].append(1.0)
                if is_answerable:
                    cat_metrics[cat]["recall_at_10"].append(rec)
                    cat_metrics[cat]["precision_at_5"].append(prec)
                    cat_metrics[cat]["mrr"].append(mrr)
                    cat_metrics[cat]["ndcg_at_10"].append(ndcg)
                else:
                    cat_metrics[cat]["no_result_accuracy"].append(no_res)

                all_query_logs.append({
                    "mode": mode,
                    "query_id": query.id,
                    "category": cat,
                    "is_answerable": is_answerable,
                    "ranked_chunk_ids": ranked[:top_k],
                    "relevant_chunk_ids": query.relevant_chunk_ids,
                    "recall": rec,
                    "precision": prec,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "no_result_accuracy": no_res,
                    "latency_ms": elapsed_ms,
                })

            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95) if sorted_latencies else 0
            p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0

            # Summarize categories
            by_category_summary: dict[str, dict[str, float]] = {}
            for cat_name, metrics_dict in cat_metrics.items():
                by_category_summary[cat_name] = {
                    "count": float(len(metrics_dict["count"])),
                    "recall_at_10": _avg(metrics_dict.get("recall_at_10", [])),
                    "precision_at_5": _avg(metrics_dict.get("precision_at_5", [])),
                    "mrr": _avg(metrics_dict.get("mrr", [])),
                    "ndcg_at_10": _avg(metrics_dict.get("ndcg_at_10", [])),
                    "no_result_accuracy": _avg(metrics_dict.get("no_result_accuracy", [])),
                }

            by_mode[mode] = ModeMetrics(
                mode=mode,
                recall_at_10=_avg(recall_values),
                precision_at_5=_avg(precision_values),
                mrr=_avg(mrr_values),
                ndcg_at_10=_avg(ndcg_values),
                no_result_accuracy=_avg(no_result_values) if no_result_values else 0.0,
                avg_latency_ms=_avg(latencies),
                p95_latency_ms=p95_latency,
                query_count=len(queries),
                answerable_count=len(recall_values),
                unanswerable_count=len(no_result_values),
                by_category=by_category_summary,
            )

        return EvaluationReport(
            dataset_id=self._dataset_id,
            dataset_version=self._dataset_version,
            modes=by_mode,
            per_query_results=all_query_logs,
            system_info=self._system_info,
        )


def _extract_ranked_ids(result: Any) -> list[int]:
    """Extract list of chunk IDs from various retriever return types."""
    if not result:
        return []
    if hasattr(result, "items"):
        # EvidencePackage
        return [int(item.chunk_id) for item in result.items]
    if isinstance(result, list):
        ids: list[int] = []
        for item in result:
            if isinstance(item, int):
                ids.append(item)
            elif hasattr(item, "chunk_id"):
                ids.append(int(item.chunk_id))
            elif hasattr(item, "id") and str(item.id).isdigit():
                ids.append(int(item.id))
        return ids
    return []


def load_dataset(path: str | Path) -> tuple[dict[str, Any], list[EvaluationQuery]]:
    """Load evaluation dataset metadata and queries from YAML or JSON."""
    target_path = Path(path)
    if not target_path.exists() and str(path).endswith("queries.yaml"):
        candidate = Path(__file__).resolve().parent / "queries.yaml"
        if candidate.exists():
            target_path = candidate
    raw = yaml.safe_load(target_path.read_text(encoding="utf-8")) or {}
    metadata = {
        "version": raw.get("version", "1.0.0"),
        "dataset_id": raw.get("dataset_id", "custom"),
        "license": raw.get("license", "unknown"),
        "provenance": raw.get("provenance", ""),
        "corpus": raw.get("corpus", []),
    }
    raw_queries = raw.get("queries", [])
    queries = [
        EvaluationQuery(
            id=str(item["id"]),
            text=str(item["text"]),
            relevant_chunk_ids=[int(c) for c in item.get("relevant_chunk_ids", [])],
            category=str(item.get("category", "general")),
            expected_document_uris=[str(u) for u in item.get("expected_document_uris", [])],
            filters=item.get("filters", {}) or {},
            description=str(item.get("description", "")),
            graded_relevance={int(k): int(v) for k, v in item.get("graded_relevance", {}).items()},
        )
        for item in raw_queries
    ]
    return metadata, queries


def load_queries(path: str | Path) -> list[EvaluationQuery]:
    """Backward-compatible query loader."""
    _, queries = load_dataset(path)
    return queries


def recall_at_k(ranked_ids: list[int], relevant: set[int], *, k: int) -> float:
    """Fraction of relevant items retrieved in top k."""
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    found = sum(1 for chunk_id in top if chunk_id in relevant)
    return found / len(relevant)


def precision_at_k(ranked_ids: list[int], relevant: set[int], *, k: int) -> float:
    """Fraction of top k results that are relevant."""
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    found = sum(1 for chunk_id in top if chunk_id in relevant)
    return found / k


def mean_reciprocal_rank(ranked_ids: list[int], relevant: set[int]) -> float:
    """Reciprocal rank of the first relevant result found."""
    for index, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    ranked_ids: list[int],
    relevant: set[int],
    *,
    k: int,
    graded_relevance: dict[int, int] | None = None,
) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    top = ranked_ids[:k]
    dcg = 0.0
    for index, chunk_id in enumerate(top, start=1):
        if graded_relevance and chunk_id in graded_relevance:
            rel = graded_relevance[chunk_id]
            gain = (2.0**rel - 1.0)
        else:
            gain = 1.0 if chunk_id in relevant else 0.0
        dcg += gain / math.log2(index + 1)

    ideal_hits = min(k, len(relevant))
    if ideal_hits == 0:
        return 0.0

    if graded_relevance:
        ideal_relevances = sorted(
            [graded_relevance.get(cid, 1) for cid in relevant], reverse=True
        )[:ideal_hits]
        idcg = sum((2.0**r - 1.0) / math.log2(idx + 1) for idx, r in enumerate(ideal_relevances, start=1))
    else:
        idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def no_result_accuracy(ranked_ids: list[int], relevant: set[int]) -> float:
    """Evaluates whether the system appropriately refrained from returning false positives for unanswerable queries."""
    if len(relevant) == 0:
        # Expected no matches: 1.0 if empty, 0.0 if false positive results returned
        return 1.0 if len(ranked_ids) == 0 else 0.0
    return 1.0


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Pipeline Adapter & In-Memory Evaluation Store for Offline Benchmarking
# ---------------------------------------------------------------------------


class InMemoryEvaluationStore:
    """Self-contained in-memory search store implementing the RetrievalStore interface.

    Allows offline evaluation without external database dependencies.
    """

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self.corpus = corpus

    def lexical_search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        source_type: str | None = None,
        tenant_id: str | None = None,
        **kwargs: object,
    ) -> list[RetrievalHit]:
        query_terms = set(query_text.lower().replace("?", "").replace(",", "").split())
        hits: list[tuple[float, dict[str, Any]]] = []

        for doc in self.corpus:
            doc_source = doc.get("source_type", "")
            if source_type and doc_source != source_type:
                continue
            if filters and "source_type" in filters and doc_source != filters["source_type"]:
                continue

            text = doc.get("text", "").lower()
            doc_words = set(text.replace("?", "").replace(",", "").split())
            overlap = sum(1.0 for term in query_terms if term in doc_words or term in text)
            if overlap > 0:
                score = overlap / max(1, len(query_terms))
                hits.append((score, doc))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalHit(
                chunk_id=int(doc["id"]),
                document_id=int(doc["id"]),
                document_uri=doc.get("document_uri", f"doc://{doc['id']}"),
                chunk_index=0,
                chunk_text=doc.get("text", ""),
                score=score,
                source="lexical",
                metadata={"source_type": doc.get("source_type", "")},
            )
            for score, doc in hits[:top_k]
        ]

    def vector_search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        source_type: str | None = None,
        tenant_id: str | None = None,
        **kwargs: object,
    ) -> list[RetrievalHit]:
        hits: list[tuple[float, dict[str, Any]]] = []

        for doc in self.corpus:
            doc_source = doc.get("source_type", "")
            if source_type and doc_source != source_type:
                continue
            if filters and "source_type" in filters and doc_source != filters["source_type"]:
                continue

            # Deterministic projection based on hash of text & query vector
            text = doc.get("text", "")
            h = sum(ord(c) for c in text[:30]) % 100
            score = 0.5 + (h / 200.0)
            hits.append((score, doc))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalHit(
                chunk_id=int(doc["id"]),
                document_id=int(doc["id"]),
                document_uri=doc.get("document_uri", f"doc://{doc['id']}"),
                chunk_index=0,
                chunk_text=doc.get("text", ""),
                score=score,
                source="vector",
                metadata={"source_type": doc.get("source_type", "")},
            )
            for score, doc in hits[:top_k]
        ]

    def get_document(self, **kwargs: object) -> dict[str, Any] | None:
        return None

    def delete_document(self, **kwargs: object) -> bool:
        return True

    def purge_tenant(self, tenant_id: str) -> dict[str, Any]:
        return {"deleted_documents": 0, "deleted_chunks": 0}

    def get_chunk(self, *, chunk_id: int, tenant_id: str | None = None) -> dict[str, Any] | None:
        for doc in self.corpus:
            if int(doc.get("id", 0)) == chunk_id:
                return {
                    "chunk_id": chunk_id,
                    "document_id": chunk_id,
                    "document_uri": doc.get("document_uri", ""),
                    "source_type": doc.get("source_type", ""),
                    "title": doc.get("title", ""),
                    "chunk_index": 0,
                    "chunk_text": doc.get("text", ""),
                    "metadata": {"source_type": doc.get("source_type", "")},
                    "tenant_id": tenant_id,
                }
        return None

    def get_index_status(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        return {
            "tenant_id": tenant_id or "all",
            "total_documents": len(self.corpus),
            "total_chunks": len(self.corpus),
            "sources": [{"source_type": "seed", "document_count": len(self.corpus), "chunk_count": len(self.corpus), "last_indexed_at": None}],
            "storage_backend": "in_memory",
        }

    def check_consistency(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        return {
            "tenant_id": tenant_id or "all",
            "is_healthy": True,
            "orphaned_chunks": 0,
            "empty_documents": 0,
            "null_tenant_documents": 0,
            "null_embedding_chunks": 0,
        }


def create_seed_pipeline(corpus_path: str | Path | None = None) -> Any:
    """Create a RetrievalPipeline wired with an in-memory seed corpus."""
    from knowledge_fabric.embeddings import MockEmbeddingProvider
    from knowledge_fabric.retrieval.pipeline import RetrievalPipeline

    path = corpus_path or Path(__file__).parent / "queries.yaml"
    metadata, _ = load_dataset(path)
    corpus = metadata.get("corpus", [])
    store = InMemoryEvaluationStore(corpus)
    return RetrievalPipeline(
        retrieval_store=store,  # type: ignore[arg-type]
        embedding_provider=MockEmbeddingProvider(_dimension=8),
    )


def make_pipeline_retriever(pipeline: Any, tenant_id: str | None = None) -> RetrieverFn:
    """Adapt a RetrievalPipeline instance into a standard RetrieverFn."""

    def _retriever(mode: str, query_text: str, top_k: int) -> list[int]:
        package = pipeline.retrieve_evidence(
            query_text=query_text,
            top_k=top_k,
            mode=mode,
            tenant_id=tenant_id,
        )
        return [item.chunk_id for item in package.items]

    return _retriever


# ---------------------------------------------------------------------------
# CLI Command Entry Point (`knowledge-fabric-eval`)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running retrieval evaluation and checking quality gates."""
    parser = argparse.ArgumentParser(
        prog="knowledge-fabric-eval",
        description="Knowledge Fabric retrieval evaluation runner and quality gate checker.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).parent / "queries.yaml"),
        help="Path to evaluation dataset YAML (default: seed queries.yaml)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="lexical,vector,hybrid",
        help="Comma-separated retrieval modes to evaluate (default: lexical,vector,hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K results per query (default: 10)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON evaluation report",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Path to write Markdown evaluation report",
    )
    parser.add_argument(
        "--threshold-mrr",
        type=float,
        default=None,
        help="Quality gate: minimum required MRR for hybrid retrieval",
    )
    parser.add_argument(
        "--threshold-recall",
        type=float,
        default=None,
        help="Quality gate: minimum required Recall@10 for hybrid retrieval",
    )
    parser.add_argument(
        "--threshold-ndcg",
        type=float,
        default=None,
        help="Quality gate: minimum required NDCG@10 for hybrid retrieval",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console markdown output",
    )

    args = parser.parse_args(argv)

    metadata, queries = load_dataset(args.dataset)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    pipeline = create_seed_pipeline(args.dataset)
    retriever = make_pipeline_retriever(pipeline)

    runner = RetrievalEvaluationRunner(
        retriever_fn=retriever,
        dataset_id=metadata.get("dataset_id", "custom"),
        dataset_version=metadata.get("version", "1.0.0"),
        system_info={
            "dataset_provenance": metadata.get("provenance", ""),
            "pipeline": "RetrievalPipeline (InMemoryEvaluationStore + MockEmbeddingProvider)",
            "modes_evaluated": list(modes),
            "top_k": args.top_k,
        },
    )

    report = runner.evaluate(queries=queries, modes=modes, top_k=args.top_k)

    # Check Quality Gates
    passed = report.check_quality_gates(
        min_mrr=args.threshold_mrr,
        min_recall=args.threshold_recall,
        min_ndcg=args.threshold_ndcg,
        target_mode="hybrid",
    )

    if not args.quiet:
        print(report.to_markdown())

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_json(), encoding="utf-8")
        print(f"Wrote JSON evaluation report to: {out_path}")

    if args.output_markdown:
        out_path = Path(args.output_markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"Wrote Markdown evaluation report to: {out_path}")

    if not passed:
        print("❌ Evaluation quality gate FAILED:", file=sys.stderr)
        for failure in report.quality_gate_failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
