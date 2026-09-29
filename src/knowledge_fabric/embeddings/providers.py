"""Embedding provider interfaces and real implementations."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class EmbeddingProvider(Protocol):
    """Interface for text-to-vector providers."""

    @property
    def dimension(self) -> int:  # pragma: no cover
        """Return embedding vector dimension."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        """Generate embeddings for input texts."""


@dataclass(slots=True)
class MockEmbeddingProvider:
    """Deterministic provider for local development and tests. NOT for production.

    Vector search results are meaningless — vectors are SHA-256 hashes of text,
    not semantic representations. Lexical search still works correctly.

    Switch to a real provider:
        EMBEDDING_PROVIDER=ollama   (free, local — requires: ollama pull nomic-embed-text)
        EMBEDDING_PROVIDER=openai   (requires OPENAI_API_KEY)
        EMBEDDING_PROVIDER=cohere   (requires COHERE_API_KEY, free tier available)
    """

    _dimension: int = 1536
    _warn: bool = True

    def __post_init__(self) -> None:
        if self._warn:
            warnings.warn(
                "\n\n\u26a0\ufe0f  MockEmbeddingProvider is active.\n"
                "   Vector search is non-functional — results are hash-based, not semantic.\n"
                "   Set EMBEDDING_PROVIDER=ollama (free) or openai or cohere for real retrieval.\n",
                stacklevel=3,
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self._dimension):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2.0 - 1.0)
        return values


@dataclass(slots=True)
class OllamaEmbeddingProvider:
    """Calls a local Ollama server for real embeddings.

    Prerequisites:
        brew install ollama   # or download from ollama.com
        ollama serve
        ollama pull nomic-embed-text   # 768-dim, fast, free

    Environment variables:
        OLLAMA_BASE_URL   default: http://localhost:11434
        OLLAMA_EMBED_MODEL  default: nomic-embed-text
    """

    _model: str
    _base_url: str
    _dimension: int

    @classmethod
    def from_env(cls) -> OllamaEmbeddingProvider:
        model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        # nomic-embed-text = 768d; mxbai-embed-large = 1024d; all-minilm = 384d
        dim_by_model: dict[str, int] = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
        }
        dimension = dim_by_model.get(model, 768)
        return cls(_model=model, _base_url=base_url.rstrip("/"), _dimension=dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"model": self._model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["embedding"]


@dataclass(slots=True)
class OpenAIEmbeddingProvider:
    """Calls the OpenAI embeddings API.

    Prerequisites:
        pip install openai   # or add to pyproject.toml dependencies

    Environment variables (required):
        OPENAI_API_KEY
        OPENAI_EMBED_MODEL  default: text-embedding-3-small  (1536d)
    """

    _model: str
    _api_key: str
    _dimension: int

    @classmethod
    def from_env(cls) -> OpenAIEmbeddingProvider:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise OSError("OPENAI_API_KEY environment variable is required")
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        dim_by_model: dict[str, int] = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        dimension = dim_by_model.get(model, 1536)
        return cls(_model=model, _api_key=api_key, _dimension=dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return [item["embedding"] for item in result["data"]]


@dataclass(slots=True)
class CohereEmbeddingProvider:
    """Calls the Cohere Embed v2 API. Free tier available — no credit card required.

    Recommended models:
        embed-english-v3.0        1024d, best quality/cost for English corpora
        embed-multilingual-v3.0   1024d, cross-lingual retrieval
        embed-english-light-v3.0  384d,  lower latency / storage

    Environment variables:
        COHERE_API_KEY        (required) — https://dashboard.cohere.com/api-keys
        COHERE_EMBED_MODEL    default: embed-english-v3.0
        COHERE_INPUT_TYPE     default: search_document
                              Use search_query when embedding queries at retrieval time.
    """

    _model: str
    _api_key: str
    _dimension: int
    _input_type: str

    @classmethod
    def from_env(cls) -> CohereEmbeddingProvider:
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise OSError("COHERE_API_KEY environment variable is required")
        model = os.environ.get("COHERE_EMBED_MODEL", "embed-english-v3.0")
        input_type = os.environ.get("COHERE_INPUT_TYPE", "search_document")
        dim_by_model: dict[str, int] = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-v2.0": 4096,
        }
        dimension = dim_by_model.get(model, 1024)
        return cls(_model=model, _api_key=api_key, _dimension=dimension, _input_type=input_type)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({
            "model": self._model,
            "texts": texts,
            "input_type": self._input_type,
            "embedding_types": ["float"],
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.cohere.com/v2/embed",
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
        return result["embeddings"]["float"]


@dataclass(slots=True)
class GeminiEmbeddingProvider:
    """Calls the Google Gemini embeddings API.

    Prerequisites:
        Set GEMINI_API_KEY (or GOOGLE_API_KEY).
        Get a key from https://aistudio.google.com/apikey

    Default model:
        text-embedding-004 (768 dimensions)

    Environment variables:
        GEMINI_API_KEY / GOOGLE_API_KEY
        GEMINI_EMBED_MODEL (default: text-embedding-004)
    """

    _model: str
    _api_key: str
    _dimension: int = 768

    @classmethod
    def from_env(cls) -> GeminiEmbeddingProvider:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise OSError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
        model = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")
        dim_by_model: dict[str, int] = {
            "text-embedding-004": 768,
            "embedding-001": 768,
            "gemini-embedding-exp-03-07": 768,
        }
        dimension = dim_by_model.get(model, 768)
        return cls(_model=model, _api_key=api_key, _dimension=dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model_name = self._model.removeprefix("models/")
        requests_payload = [
            {
                "model": f"models/{model_name}",
                "content": {"parts": [{"text": text}]},
            }
            for text in texts
        ]
        body = json.dumps({"requests": requests_payload}).encode("utf-8")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:batchEmbedContents"
            f"?key={self._api_key}"
        )
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return [item["values"] for item in result["embeddings"]]


@dataclass
class SentenceTransformerEmbeddingProvider:
    """Local SentenceTransformer embedding provider via Hugging Face.

    Prerequisites:
        pip install sentence-transformers

    Recommended models:
        all-MiniLM-L6-v2       384d, fast, lightweight
        BAAI/bge-small-en-v1.5 384d, high quality
        BAAI/bge-base-en-v1.5  768d, enterprise quality

    Environment variables:
        SENTENCE_TRANSFORMER_MODEL  default: all-MiniLM-L6-v2
    """

    _model_name: str
    _dimension: int
    _model: Any = None

    @classmethod
    def from_env(cls) -> SentenceTransformerEmbeddingProvider:
        model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        dim_by_model: dict[str, int] = {
            "all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
        }
        dimension = dim_by_model.get(model_name, 384)
        return cls(_model_name=model_name, _dimension=dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except (ImportError, ModuleNotFoundError) as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
            if hasattr(self._model, "get_sentence_embedding_dimension"):
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]


class EmbeddingProviderRegistry:
    """Extensible registry for embedding providers."""

    _factories: dict[str, Callable[..., EmbeddingProvider]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., EmbeddingProvider]) -> None:
        """Register an embedding provider factory."""
        cls._factories[name.lower().strip()] = factory

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> EmbeddingProvider:
        """Resolve and instantiate an embedding provider."""
        key = name.lower().strip()
        if key not in cls._factories:
            raise KeyError(
                f"Unknown embedding provider '{name}'. "
                f"Available providers: {sorted(cls._factories.keys())}"
            )
        factory = cls._factories[key]
        if callable(factory):
            try:
                return factory(**kwargs)
            except TypeError:
                return factory()
        return factory

    @classmethod
    def list_providers(cls) -> list[str]:
        return sorted(list(cls._factories.keys()))

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name.lower().strip() in cls._factories


EmbeddingProviderRegistry.register(
    "mock",
    lambda **kwargs: MockEmbeddingProvider(
        _dimension=kwargs.get("dimension", 1536),
        _warn=kwargs.get("warn", True),
    ),
)
EmbeddingProviderRegistry.register("ollama", OllamaEmbeddingProvider.from_env)
EmbeddingProviderRegistry.register("openai", OpenAIEmbeddingProvider.from_env)
EmbeddingProviderRegistry.register("cohere", CohereEmbeddingProvider.from_env)
EmbeddingProviderRegistry.register("gemini", GeminiEmbeddingProvider.from_env)
EmbeddingProviderRegistry.register("sentence_transformers", SentenceTransformerEmbeddingProvider.from_env)
EmbeddingProviderRegistry.register("sentence-transformers", SentenceTransformerEmbeddingProvider.from_env)
EmbeddingProviderRegistry.register("local", SentenceTransformerEmbeddingProvider.from_env)


def build_embedding_provider(provider_name: str = "", dimension: int = 768) -> EmbeddingProvider:
    """Factory that reads EMBEDDING_PROVIDER env var and returns the right provider.

    EMBEDDING_PROVIDER=mock     → MockEmbeddingProvider (default, dev only — emits warning)
    EMBEDDING_PROVIDER=ollama   → OllamaEmbeddingProvider.from_env()
    EMBEDDING_PROVIDER=gemini   → GeminiEmbeddingProvider.from_env()
    EMBEDDING_PROVIDER=openai   → OpenAIEmbeddingProvider.from_env()
    EMBEDDING_PROVIDER=cohere   → CohereEmbeddingProvider.from_env()
    EMBEDDING_PROVIDER=local    → SentenceTransformerEmbeddingProvider.from_env()

    Recommended for local dev:  ollama (free, local) or local (sentence-transformers)
    Recommended for production: gemini (free tier/enterprise), cohere, or openai
    """
    name = (provider_name or os.environ.get("EMBEDDING_PROVIDER", "mock")).lower().strip()
    if EmbeddingProviderRegistry.is_registered(name):
        return EmbeddingProviderRegistry.get(name, dimension=dimension)
    return MockEmbeddingProvider(_dimension=dimension)
