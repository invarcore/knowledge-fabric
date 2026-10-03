from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_fabric.evaluation import (
    EvaluationQuery,
    EvaluationReport,
    InMemoryEvaluationStore,
    ModeMetrics,
    RetrievalEvaluationRunner,
    create_seed_pipeline,
    load_dataset,
    load_queries,
    main,
    make_pipeline_retriever,
    mean_reciprocal_rank,
    ndcg_at_k,
    no_result_accuracy,
    precision_at_k,
    recall_at_k,
)

_SEED_QUERIES_PATH = Path(__file__).resolve().parent.parent / "src" / "knowledge_fabric" / "evaluation" / "queries.yaml"


def test_metric_functions() -> None:
    ranked = [3, 1, 9, 7, 5]
    relevant = {1, 5}

    assert recall_at_k(ranked, relevant, k=3) == 0.5
    assert precision_at_k(ranked, relevant, k=5) == 0.4
    assert mean_reciprocal_rank(ranked, relevant) == 0.5
    assert 0.0 <= ndcg_at_k(ranked, relevant, k=5) <= 1.0


def test_no_result_accuracy_metric() -> None:
    # If no relevant documents exist (unanswerable):
    assert no_result_accuracy([], set()) == 1.0  # correctly returned nothing
    assert no_result_accuracy([1, 2], set()) == 0.0  # returned false positives

    # If relevant documents exist (answerable):
    assert no_result_accuracy([1], {1}) == 1.0


def test_ndcg_at_k_graded_relevance() -> None:
    ranked = [101, 102, 103]
    relevant = {101, 102}
    graded = {101: 3, 102: 1}

    score = ndcg_at_k(ranked, relevant, k=3, graded_relevance=graded)
    assert 0.0 < score <= 1.0

    # Empty relevant yields 0.0
    assert ndcg_at_k([101], set(), k=3) == 0.0


def test_load_queries_from_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "queries.yaml"
    yaml_file.write_text(
        """
queries:
  - id: q1
    text: sample
    relevant_chunk_ids: [1, 2]
""".strip(),
        encoding="utf-8",
    )
    queries = load_queries(yaml_file)
    assert len(queries) == 1
    assert queries[0].id == "q1"
    assert queries[0].relevant_chunk_ids == [1, 2]
    assert queries[0].category == "general"


def test_load_dataset_with_metadata_and_corpus(tmp_path: Path) -> None:
    yaml_file = tmp_path / "dataset.yaml"
    yaml_file.write_text(
        """
version: "1.2.0"
dataset_id: "test-eval-v1"
license: "Apache-2.0"
provenance: "Unit test dataset"
corpus:
  - id: 1
    text: "Sample text"
    source_type: "markdown"
queries:
  - id: q-exact-01
    category: "exact"
    text: "sample query"
    relevant_chunk_ids: [1]
    expected_document_uris: ["doc-1"]
    description: "Sample description"
""".strip(),
        encoding="utf-8",
    )
    metadata, queries = load_dataset(yaml_file)
    assert metadata["version"] == "1.2.0"
    assert metadata["dataset_id"] == "test-eval-v1"
    assert len(metadata["corpus"]) == 1
    assert len(queries) == 1
    assert queries[0].id == "q-exact-01"
    assert queries[0].category == "exact"
    assert queries[0].description == "Sample description"


def test_evaluation_runner_modes() -> None:
    queries = load_queries(_SEED_QUERIES_PATH)

    def _retriever(mode: str, query: str, top_k: int) -> list[int]:
        if mode == "lexical":
            return [101, 202, 999]
        if mode == "vector":
            return [202, 888, 777]
        return [101, 202, 301]

    runner = RetrievalEvaluationRunner(_retriever)
    report = runner.evaluate(queries=queries)

    assert set(report.keys()) == {"lexical", "vector", "hybrid"}
    assert all(0.0 <= metrics.recall_at_10 <= 1.0 for metrics in report.values())
    assert all(0.0 <= metrics.mrr <= 1.0 for metrics in report.values())
    assert all(0.0 <= metrics.ndcg_at_10 <= 1.0 for metrics in report.values())


def test_evaluation_runner_category_breakdown() -> None:
    queries = [
        EvaluationQuery(id="q1", text="query 1", relevant_chunk_ids=[101], category="exact"),
        EvaluationQuery(id="q2", text="query 2", relevant_chunk_ids=[202], category="paraphrase"),
        EvaluationQuery(id="q3", text="query 3", relevant_chunk_ids=[], category="unanswerable"),
    ]

    def _retriever(mode: str, query: str, top_k: int) -> list[int]:
        if "query 1" in query:
            return [101]
        if "query 2" in query:
            return [202]
        return []  # unanswerable -> returns empty

    runner = RetrievalEvaluationRunner(_retriever)
    report = runner.evaluate(queries=queries, modes=("hybrid",))

    hybrid = report["hybrid"]
    assert hybrid.query_count == 3
    assert hybrid.answerable_count == 2
    assert hybrid.unanswerable_count == 1
    assert hybrid.recall_at_10 == 1.0
    assert hybrid.precision_at_5 == pytest.approx(1.0 / 5.0)
    assert hybrid.mrr == 1.0
    assert hybrid.no_result_accuracy == 1.0

    assert "exact" in hybrid.by_category
    assert "paraphrase" in hybrid.by_category
    assert "unanswerable" in hybrid.by_category
    assert hybrid.by_category["exact"]["recall_at_10"] == 1.0
    assert hybrid.by_category["unanswerable"]["no_result_accuracy"] == 1.0


def test_evaluation_report_formatting_and_gates() -> None:
    report = EvaluationReport(
        dataset_id="test-dataset",
        dataset_version="1.0.0",
        modes={
            "hybrid": ModeMetrics(
                mode="hybrid",
                recall_at_10=0.85,
                precision_at_5=0.40,
                mrr=0.75,
                ndcg_at_10=0.80,
                no_result_accuracy=1.0,
                avg_latency_ms=12.5,
                p95_latency_ms=25.0,
                query_count=10,
                answerable_count=8,
                unanswerable_count=2,
                by_category={
                    "exact": {"count": 4, "recall_at_10": 0.90, "precision_at_5": 0.50, "mrr": 0.85, "ndcg_at_10": 0.88},
                },
            )
        },
    )

    # Test JSON output
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["dataset_id"] == "test-dataset"
    assert "hybrid" in parsed["modes"]

    # Test Markdown output
    md_str = report.to_markdown()
    assert "# Retrieval Evaluation Report: `test-dataset`" in md_str
    assert "| `hybrid` |" in md_str
    assert "| `exact` |" in md_str

    # Test quality gates
    assert report.check_quality_gates(min_mrr=0.70, min_recall=0.80) is True
    assert report.quality_gate_passed is True

    assert report.check_quality_gates(min_mrr=0.90) is False
    assert report.quality_gate_passed is False
    assert len(report.quality_gate_failures) == 1


def test_seed_pipeline_and_pipeline_retriever() -> None:
    pipeline = create_seed_pipeline()
    retriever = make_pipeline_retriever(pipeline)

    # Test retrieval across all three modes
    lexical_hits = retriever("lexical", "reciprocal rank fusion", 5)
    assert len(lexical_hits) > 0
    assert 101 in lexical_hits or 102 in lexical_hits

    vector_hits = retriever("vector", "nearest neighbor chunk embeddings", 5)
    assert len(vector_hits) > 0

    hybrid_hits = retriever("hybrid", "evidence package response model", 5)
    assert len(hybrid_hits) > 0

    # Run runner over seed dataset using seed pipeline
    metadata, queries = load_dataset(_SEED_QUERIES_PATH)
    runner = RetrievalEvaluationRunner(retriever, dataset_id=metadata["dataset_id"])
    report = runner.evaluate(queries=queries, modes=("lexical", "vector", "hybrid"))

    assert set(report.keys()) == {"lexical", "vector", "hybrid"}
    for mode in ("lexical", "vector", "hybrid"):
        metrics = report[mode]
        assert metrics.query_count == len(queries)
        assert metrics.avg_latency_ms >= 0.0


def test_cli_main_execution(tmp_path: Path) -> None:
    out_json = tmp_path / "eval_report.json"
    out_md = tmp_path / "eval_report.md"

    exit_code = main([
        "--modes", "hybrid,lexical",
        "--output-json", str(out_json),
        "--output-markdown", str(out_md),
        "--threshold-mrr", "0.0",
        "--quiet",
    ])
    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()

    # Test failing quality gate
    fail_exit_code = main([
        "--threshold-mrr", "1.1",  # impossible MRR
        "--quiet",
    ])
    assert fail_exit_code == 1
