# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Data models for ingestion and chunking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class SourceFormat(StrEnum):
    """Supported source formats for ingestion."""

    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"


class IngestionStatus(StrEnum):
    """Execution status of an ingested document."""

    ADDED = "added"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(slots=True)
class Document:
    """Canonical ingested document."""

    source_uri: str
    source_format: SourceFormat
    content_text: str
    metadata: dict[str, object] = field(default_factory=dict)
    title: str | None = None
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = ""
    content_length: int = 0
    headings: list[str] = field(default_factory=list)
    page_count: int | None = None

    def __post_init__(self) -> None:
        if not self.content_hash and self.content_text:
            object.__setattr__(
                self,
                "content_hash",
                hashlib.sha256(self.content_text.encode("utf-8")).hexdigest(),
            )
        if not self.content_length and self.content_text:
            object.__setattr__(self, "content_length", len(self.content_text))
        if self.content_hash and "content_hash" not in self.metadata:
            self.metadata["content_hash"] = self.content_hash
        if self.content_length and "content_length" not in self.metadata:
            self.metadata["content_length"] = self.content_length
        if self.headings and "headings" not in self.metadata:
            self.metadata["headings"] = self.headings


@dataclass(slots=True)
class Chunk:
    """Canonical chunk representation."""

    document_uri: str
    chunk_index: int
    chunk_text: str
    metadata: dict[str, object] = field(default_factory=dict)
    heading_path: str = ""
    heading_level: int | None = None
    content_hash: str = ""
    start_offset: int = 0
    end_offset: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash and self.chunk_text:
            object.__setattr__(
                self,
                "content_hash",
                hashlib.sha256(self.chunk_text.encode("utf-8")).hexdigest(),
            )
        # Ensure metadata contains these fields for downstream citation extraction
        if self.heading_path and "heading_path" not in self.metadata:
            self.metadata["heading_path"] = self.heading_path
        if self.heading_level is not None and "heading_level" not in self.metadata:
            self.metadata["heading_level"] = self.heading_level
        if self.content_hash and "content_hash" not in self.metadata:
            self.metadata["content_hash"] = self.content_hash
        if self.start_offset is not None and "start_offset" not in self.metadata:
            self.metadata["start_offset"] = self.start_offset
        if self.end_offset is not None and "end_offset" not in self.metadata:
            self.metadata["end_offset"] = self.end_offset


@dataclass(slots=True)
class IngestionRecord:
    """Structured record for a single document ingestion outcome."""

    source_path: str
    status: IngestionStatus
    document_id: int | None = None
    chunks_count: int = 0
    error_message: str | None = None
    duration_ms: float = 0.0
