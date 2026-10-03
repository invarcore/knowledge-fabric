import json
import pytest

from knowledge_fabric.embeddings import (
    EmbeddingProviderRegistry,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_embedding_provider,
)


def test_mock_embeddings_are_deterministic() -> None:
    provider = MockEmbeddingProvider(_dimension=8)
    first = provider.embed_texts(["hello world"])[0]
    second = provider.embed_texts(["hello world"])[0]
    other = provider.embed_texts(["other"])[0]

    assert first == second
    assert first != other
    assert len(first) == 8


def test_mock_provider_dimension_property() -> None:
    provider = MockEmbeddingProvider(_dimension=32)
    assert provider.dimension == 32


def test_embedding_provider_registry_builtins() -> None:
    providers = EmbeddingProviderRegistry.list_providers()
    assert "mock" in providers
    assert "gemini" in providers
    assert "ollama" in providers
    assert "openai" in providers
    assert "cohere" in providers
    assert "sentence_transformers" in providers
    assert "local" in providers


def test_embedding_provider_registry_custom() -> None:
    class DummyProvider:
        @property
        def dimension(self) -> int:
            return 42

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.5] * 42 for _ in texts]

    EmbeddingProviderRegistry.register("custom_dummy", lambda **_kw: DummyProvider())
    assert EmbeddingProviderRegistry.is_registered("custom_dummy")
    instance = EmbeddingProviderRegistry.get("custom_dummy")
    assert instance.dimension == 42
    assert len(instance.embed_texts(["hi"])[0]) == 42


def test_gemini_embedding_provider_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    provider = GeminiEmbeddingProvider.from_env()
    assert provider.dimension == 768

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "embeddings": [
                    {"values": [0.1] * 768},
                    {"values": [0.2] * 768},
                ]
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _req, timeout=30: _FakeResponse(),
    )

    vecs = provider.embed_texts(["first", "second"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 768
    assert vecs[0][0] == 0.1
    assert vecs[1][0] == 0.2


def test_sentence_transformer_missing_dep() -> None:
    provider = SentenceTransformerEmbeddingProvider(
        _model_name="all-MiniLM-L6-v2",
        _dimension=384,
    )
    assert provider.dimension == 384
    # If sentence_transformers is not in test env, it should cleanly raise ImportError
    # If it is, embed_texts should return vector of 384 floats.
    try:
        vecs = provider.embed_texts(["test sentence"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 384
    except ImportError as exc:
        assert "sentence-transformers is not installed" in str(exc)


def test_openrouter_embedding_provider_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OSError, match="OPENROUTER_API_KEY environment variable is required"):
        build_embedding_provider("openrouter")


def test_openrouter_embedding_provider_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    provider = build_embedding_provider("openrouter")
    assert provider.dimension == 1536

    # Test custom model dimension
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "baai/bge-m3")
    bge_provider = build_embedding_provider("openrouter")
    assert bge_provider.dimension == 1024

    # Test empty texts
    assert provider.embed_texts([]) == []

    class _FakeOpenRouterResponse:
        def __enter__(self) -> "_FakeOpenRouterResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "object": "list",
                "data": [
                    {"index": 1, "embedding": [0.42] * 1536},
                    {"index": 0, "embedding": [0.11] * 1536},
                ],
                "model": "openai/text-embedding-3-small",
            }
            return json.dumps(payload).encode("utf-8")

    captured_headers: dict[str, str] = {}

    def _fake_urlopen(req: Any, timeout: int = 30) -> _FakeOpenRouterResponse:
        captured_headers.update(req.headers)
        return _FakeOpenRouterResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    vecs = provider.embed_texts(["doc1", "doc2"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1536
    # Verified ordering by index (index 0 was returned second in mock payload)
    assert vecs[0][0] == 0.11
    assert vecs[1][0] == 0.42
    assert "Bearer test-openrouter-key" in captured_headers.get("Authorization", "")

