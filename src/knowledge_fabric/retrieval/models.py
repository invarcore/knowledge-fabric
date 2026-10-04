# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Models for lexical/vector retrieval outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalHit:
    """One retrieval result row."""

    chunk_id: int
    document_id: int = 0
    document_uri: str = ""
    chunk_index: int = 0
    chunk_text: str = ""
    score: float = 0.0
    source: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __init__(
        self,
        chunk_id: int,
        document_id: int = 0,
        document_uri: str = "",
        chunk_index: int = 0,
        chunk_text: str = "",
        score: float = 0.0,
        source: str = "",
        metadata: dict[str, object] | None = None,
        *,
        snippet: str | None = None,
        source_type: str | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_uri = document_uri
        self.chunk_index = chunk_index
        self.chunk_text = snippet if snippet is not None else chunk_text
        self.score = score
        self.source = source_type if source_type is not None else source
        self.metadata = metadata if metadata is not None else {}

    @property
    def snippet(self) -> str:
        return self.chunk_text

    @snippet.setter
    def snippet(self, value: str) -> None:
        self.chunk_text = value

    @property
    def source_type(self) -> str:
        return self.source

    @source_type.setter
    def source_type(self, value: str) -> None:
        self.source = value

