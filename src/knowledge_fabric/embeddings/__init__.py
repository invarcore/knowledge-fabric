# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Embedding provider abstraction."""

from knowledge_fabric.embeddings.providers import (
    CohereEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingProviderRegistry,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_embedding_provider,
)

__all__ = [
    "CohereEmbeddingProvider",
    "EmbeddingProvider",
    "EmbeddingProviderRegistry",
    "GeminiEmbeddingProvider",
    "MockEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OpenRouterEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "build_embedding_provider",
]
