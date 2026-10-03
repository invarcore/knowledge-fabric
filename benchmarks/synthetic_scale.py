#!/usr/bin/env python3
"""Configurable synthetic scale and latency benchmark for Knowledge Fabric.

Implements Roadmap Phase 11 validation track:
- Generates reproducible synthetic document and vector corpora.
- Measures ingestion throughput (chunks/sec).
- Measures cold vs warm query latency percentiles (p50, p95, p99).
- Compares Lexical, Vector, and Hybrid RRF retrieval strategies.
- Adheres to Capability Truth: records exact hardware, dimensions, and limits.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure knowledge_fabric is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from knowledge_fabric.embeddings import MockEmbeddingProvider
from knowledge_fabric.evaluation.runner import InMemoryEvaluationStore
from knowledge_fabric.retrieval.pipeline import RetrievalPipeline


@dataclass(slots=True)
class ScaleBenchmarkResult:
    dataset_size_documents: int
    dataset_size_chunks: int
    vector_dimension: int
    ingestion_duration_seconds: float
    ingestion_throughput_chunks_per_sec: float
    cold_query_latency_ms: float
    warm_p50_latency_ms: float
    warm_p95_latency_ms: float
    warm_p99_latency_ms: float
    lexical_avg_latency_ms: float
    vector_avg_latency_ms: float
    hybrid_avg_latency_ms: float
    hardware_info: dict[str, str]


def generate_synthetic_corpus(
    doc_count: int = 100,
    chunks_per_doc: int = 4,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate deterministic synthetic documents and chunks."""
    rng = random.Random(random_seed)
    tech_vocabulary = [
        "architecture", "pipeline", "reciprocal", "fusion", "embedding",
        "pgvector", "hnsw", "tsvector", "evidence", "provenance",
        "tenant", "namespace", "isolation", "reranking", "mcp",
        "governance", "policy", "token", "audit", "latency",
    ]

    corpus: list[dict[str, Any]] = []
    chunk_id = 1
    for doc_idx in range(1, doc_count + 1):
        doc_uri = f"synthetic://corp/doc_{doc_idx:04d}.md"
        source_type = "markdown" if doc_idx % 2 == 0 else "python"
        for chunk_idx in range(chunks_per_doc):
            terms = rng.sample(tech_vocabulary, k=min(6, len(tech_vocabulary)))
            text = f"Section {chunk_idx} of Document {doc_idx}: " + " ".join(terms) + " " + "".join(
                rng.choices(string.ascii_lowercase + " ", k=80)
            )
            corpus.append({
                "id": chunk_id,
                "document_id": doc_idx,
                "document_uri": doc_uri,
                "chunk_index": chunk_idx,
                "source_type": source_type,
                "text": text,
            })
            chunk_id += 1

    return corpus


def run_scale_benchmark(
    doc_count: int = 200,
    chunks_per_doc: int = 3,
    query_count: int = 50,
    dimension: int = 64,
    top_k: int = 10,
) -> ScaleBenchmarkResult:
    """Execute synthetic benchmark and collect throughput and latency metrics."""
    # 1. Ingestion Phase
    t_ingest_start = perf_counter()
    corpus = generate_synthetic_corpus(doc_count=doc_count, chunks_per_doc=chunks_per_doc)
    store = InMemoryEvaluationStore(corpus)
    provider = MockEmbeddingProvider(_dimension=dimension)
    pipeline = RetrievalPipeline(retrieval_store=store, embedding_provider=provider)
    t_ingest_end = perf_counter()

    ingest_dur = max(0.0001, t_ingest_end - t_ingest_start)
    total_chunks = len(corpus)
    throughput = total_chunks / ingest_dur

    # 2. Query Latency Phase
    sample_queries = [
        "reciprocal rank fusion pipeline latency",
        "pgvector hnsw index performance",
        "evidence provenance and tenant isolation",
        "token audit governance policy",
        "architecture embedding reranking mcp",
    ]

    # Cold Query (first query execution)
    t_cold_0 = perf_counter()
    pipeline.retrieve_evidence(query_text=sample_queries[0], top_k=top_k, mode="hybrid")
    cold_latency_ms = (perf_counter() - t_cold_0) * 1000.0

    # Warm Queries
    latencies: list[float] = []
    lex_latencies: list[float] = []
    vec_latencies: list[float] = []
    hyb_latencies: list[float] = []

    for i in range(query_count):
        q = sample_queries[i % len(sample_queries)]

        # Lexical
        t0 = perf_counter()
        pipeline.retrieve_evidence(query_text=q, top_k=top_k, mode="lexical")
        lex_latencies.append((perf_counter() - t0) * 1000.0)

        # Vector
        t0 = perf_counter()
        pipeline.retrieve_evidence(query_text=q, top_k=top_k, mode="vector")
        vec_latencies.append((perf_counter() - t0) * 1000.0)

        # Hybrid
        t0 = perf_counter()
        pipeline.retrieve_evidence(query_text=q, top_k=top_k, mode="hybrid")
        hyb_ms = (perf_counter() - t0) * 1000.0
        hyb_latencies.append(hyb_ms)
        latencies.append(hyb_ms)

    latencies.sort()
    p50_idx = int(len(latencies) * 0.50)
    p95_idx = int(len(latencies) * 0.95)
    p99_idx = int(len(latencies) * 0.99)

    return ScaleBenchmarkResult(
        dataset_size_documents=doc_count,
        dataset_size_chunks=total_chunks,
        vector_dimension=dimension,
        ingestion_duration_seconds=round(ingest_dur, 4),
        ingestion_throughput_chunks_per_sec=round(throughput, 1),
        cold_query_latency_ms=round(cold_latency_ms, 2),
        warm_p50_latency_ms=round(latencies[p50_idx], 2),
        warm_p95_latency_ms=round(latencies[p95_idx], 2),
        warm_p99_latency_ms=round(latencies[min(p99_idx, len(latencies) - 1)], 2),
        lexical_avg_latency_ms=round(sum(lex_latencies) / len(lex_latencies), 2),
        vector_avg_latency_ms=round(sum(vec_latencies) / len(vec_latencies), 2),
        hybrid_avg_latency_ms=round(sum(hyb_latencies) / len(hyb_latencies), 2),
        hardware_info={
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": platform.processor() or "Unknown CPU",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="synthetic-scale-benchmark",
        description="Run Knowledge Fabric synthetic scale and query latency benchmarks.",
    )
    parser.add_argument("--docs", type=int, default=200, help="Number of synthetic documents (default: 200)")
    parser.add_argument("--chunks-per-doc", type=int, default=3, help="Chunks per synthetic document (default: 3)")
    parser.add_argument("--queries", type=int, default=50, help="Number of benchmark query iterations (default: 50)")
    parser.add_argument("--dimension", type=int, default=64, help="Vector dimension (default: 64)")
    parser.add_argument("--top-k", type=int, default=10, help="Candidate top-K per query (default: 10)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")

    args = parser.parse_args(argv)

    print(f"Executing scale benchmark: {args.docs} docs, {args.docs * args.chunks_per_doc} chunks, {args.queries} queries...")
    res = run_scale_benchmark(
        doc_count=args.docs,
        chunks_per_doc=args.chunks_per_doc,
        query_count=args.queries,
        dimension=args.dimension,
        top_k=args.top_k,
    )

    data = asdict(res)
    print("\n=== Scale Benchmark Results ===")
    print(json.dumps(data, indent=2))

    if args.output_json:
        out_file = Path(args.output_json)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark results to: {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
