# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Retrieval evaluation helpers and quality gate runner."""

from knowledge_fabric.evaluation.runner import (
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

__all__ = [
    "EvaluationQuery",
    "EvaluationReport",
    "InMemoryEvaluationStore",
    "ModeMetrics",
    "RetrievalEvaluationRunner",
    "create_seed_pipeline",
    "load_dataset",
    "load_queries",
    "main",
    "make_pipeline_retriever",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "no_result_accuracy",
    "precision_at_k",
    "recall_at_k",
]
