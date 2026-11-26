# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knowledge_fabric.retrieval.pipeline import RetrievalPipeline


@dataclass
class _MockRetrievalHit:
    chunk_id: str | None
    chunk_text: str
    document_uri: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class _StubRetrievalPipeline(RetrievalPipeline):
    def __init__(self, stub_hits: list[_MockRetrievalHit]) -> None:
        self._stub_hits = stub_hits

    def _execute_hybrid_search(self, query: str, *, top_k: int = 10) -> list[Any]:
        return list(self._stub_hits[:top_k])


def test_retrieve_preserves_multiple_hits_with_none_chunk_id_and_metadata() -> None:
    hits = [
        _MockRetrievalHit(
            chunk_id=None,
            chunk_text="def helper_a(x):\n    return x + 1\n",
            document_uri="src/a.py",
            score=0.60,
            metadata={"source": "repo-a"},
        ),
        _MockRetrievalHit(
            chunk_id=None,
            chunk_text=(
                "import hmac\n"
                "def validate_token(a: str, b: str) -> bool:\n"
                "    if not hmac.compare_digest(a, b):\n"
                "        raise PermissionError('bad')\n"
                "    return True\n"
            ),
            document_uri="src/auth.py",
            score=0.55,
            metadata={"source": "repo-b"},
        ),
    ]

    pipeline = _StubRetrievalPipeline(hits)
    results = pipeline.retrieve(
        "validate token", top_k=5, enable_case_guard=True, award_safeguard_boost=True
    )

    assert len(results) == 2
    assert results[0].document_uri == "src/auth.py"
    assert results[0].metadata["source"] == "repo-b"
    assert results[0].metadata["case_guard"]["is_security_safeguard"] is True
    assert results[1].document_uri == "src/a.py"
    assert results[1].metadata["source"] == "repo-a"


def test_retrieve_quarantine_insecure_quarantines_unverified_tests_by_default() -> None:
    hits = [
        _MockRetrievalHit(
            chunk_id="test-exploit",
            chunk_text="import os\nos.system(cmd)\n",
            document_uri="tests/test_exploit.py",
            score=0.95,
            metadata={"origin_verified": False},
        ),
        _MockRetrievalHit(
            chunk_id="safe-prod",
            chunk_text="def add(a, b):\n    return a + b\n",
            document_uri="src/math.py",
            score=0.50,
        ),
        _MockRetrievalHit(
            chunk_id="verified-test",
            chunk_text="import os\nos.system(cmd)\n",
            document_uri="tests/test_verified.py",
            score=0.40,
            metadata={"origin_verified": True},
        ),
    ]

    pipeline = _StubRetrievalPipeline(hits)

    quarantined = pipeline.retrieve(
        "math", top_k=5, enable_case_guard=True, quarantine_insecure=True
    )
    assert [h.chunk_id for h in quarantined] == ["safe-prod", "verified-test"]

    kept = pipeline.retrieve(
        "math",
        top_k=5,
        enable_case_guard=True,
        quarantine_insecure=True,
        quarantine_tests=False,
    )
    assert [h.chunk_id for h in kept] == ["safe-prod", "verified-test", "test-exploit"]


def test_retrieve_disabled_case_guard_leaves_hits_untouched() -> None:
    hits = [
        _MockRetrievalHit(
            chunk_id="insecure-1",
            chunk_text="import os\nos.system(cmd)\n",
            document_uri="src/runner.py",
            score=0.90,
        ),
    ]
    pipeline = _StubRetrievalPipeline(hits)
    results = pipeline.retrieve("audit os.system", top_k=5, enable_case_guard=False)

    assert len(results) == 1
    assert results[0].score == 0.90
    assert "case_guard" not in results[0].metadata
