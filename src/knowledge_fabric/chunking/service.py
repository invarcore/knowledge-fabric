# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Structure-aware chunking service for Markdown, HTML, PDF/pages, and fallback text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge_fabric.ingestion.models import Chunk, Document, SourceFormat

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CODE_BLOCK_PATTERN = re.compile(r"(```[\s\S]*?```)")
_TABLE_PATTERN = re.compile(r"((?:^\|.*?\|\s*$\n?)+)", re.MULTILINE)
_PAGE_MARKER_PATTERN = re.compile(r"(?:\f|---\s*Page\s+(\d+)\s*---)", re.IGNORECASE)


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration parameters for document chunking."""

    max_chars: int = 1200
    overlap_chars: int = 150
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    min_chunk_chars: int = 30


class DocumentChunkingService:
    """Split ingested documents into structure-preserving chunks."""

    def __init__(
        self,
        max_chars: int = 1200,
        overlap_chars: int = 150,
        config: ChunkingConfig | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            if max_chars <= 0:
                raise ValueError("max_chars must be greater than 0")
            if overlap_chars < 0:
                raise ValueError("overlap_chars must be >= 0")
            if overlap_chars >= max_chars:
                raise ValueError("overlap_chars must be smaller than max_chars")
            self._config = ChunkingConfig(max_chars=max_chars, overlap_chars=overlap_chars)

    @property
    def max_chars(self) -> int:
        return self._config.max_chars

    @property
    def overlap_chars(self) -> int:
        return self._config.overlap_chars

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk document content with structure-aware logic according to format."""
        if document.source_format is SourceFormat.MARKDOWN:
            chunks = self._chunk_markdown_structure(document)
            if chunks:
                return chunks

        # Page-aware chunking for PDFs and presentations
        if document.source_format in (SourceFormat.PDF, SourceFormat.PPTX):
            page_chunks = self._chunk_page_aware(document)
            if page_chunks:
                return page_chunks

        return self._chunk_fallback(document)

    def _chunk_markdown_structure(self, document: Document) -> list[Chunk]:
        text = document.content_text.strip()
        matches = list(_HEADING_PATTERN.finditer(text))
        if not matches:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        heading_stack: list[tuple[int, str]] = []

        for pos, match in enumerate(matches):
            level = len(match.group(1))
            heading_title = match.group(2).strip()

            # Update heading hierarchy stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_title))
            heading_path = " > ".join(title for _, title in heading_stack)

            start = match.start()
            end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if len(section_text) <= self._config.max_chars:
                offset_start = text.find(section_text, start)
                offset_end = offset_start + len(section_text) if offset_start >= 0 else end
                chunk = Chunk(
                    document_uri=document.source_uri,
                    chunk_index=chunk_index,
                    chunk_text=section_text,
                    heading_path=heading_path,
                    heading_level=level,
                    start_offset=max(0, offset_start),
                    end_offset=offset_end,
                    metadata={
                        "heading": heading_title,
                        "heading_level": level,
                        "heading_path": heading_path,
                        "start_offset": max(0, offset_start),
                        "end_offset": offset_end,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1
                continue

            # Split large section into atomic blocks (headings, code blocks, tables, paragraphs)
            blocks = self._split_section_into_blocks(section_text)
            accumulated_text = ""
            accumulated_start = start

            for block in blocks:
                if not accumulated_text:
                    accumulated_text = block
                elif len(accumulated_text) + 1 + len(block) <= self._config.max_chars:
                    accumulated_text = f"{accumulated_text}\n\n{block}"
                else:
                    # Emit current accumulated chunk
                    offset_start = text.find(accumulated_text, accumulated_start)
                    offset_end = offset_start + len(accumulated_text) if offset_start >= 0 else accumulated_start + len(accumulated_text)
                    chunks.append(
                        Chunk(
                            document_uri=document.source_uri,
                            chunk_index=chunk_index,
                            chunk_text=accumulated_text,
                            heading_path=heading_path,
                            heading_level=level,
                            start_offset=max(0, offset_start),
                            end_offset=offset_end,
                            metadata={
                                "heading": heading_title,
                                "heading_level": level,
                                "heading_path": heading_path,
                                "split_from_heading_section": True,
                                "start_offset": max(0, offset_start),
                                "end_offset": offset_end,
                            },
                        )
                    )
                    chunk_index += 1
                    accumulated_start = max(0, offset_end - self._config.overlap_chars)
                    accumulated_text = block

            if accumulated_text:
                offset_start = text.find(accumulated_text, accumulated_start)
                offset_end = offset_start + len(accumulated_text) if offset_start >= 0 else accumulated_start + len(accumulated_text)
                chunks.append(
                    Chunk(
                        document_uri=document.source_uri,
                        chunk_index=chunk_index,
                        chunk_text=accumulated_text,
                        heading_path=heading_path,
                        heading_level=level,
                        start_offset=max(0, offset_start),
                        end_offset=offset_end,
                        metadata={
                            "heading": heading_title,
                            "heading_level": level,
                            "heading_path": heading_path,
                            "split_from_heading_section": True,
                            "start_offset": max(0, offset_start),
                            "end_offset": offset_end,
                        },
                    )
                )
                chunk_index += 1

        return chunks

    def _split_section_into_blocks(self, section_text: str) -> list[str]:
        """Decompose a markdown section into discrete structural blocks (paragraphs, tables, code blocks)."""
        # Split on double newline while keeping code blocks intact
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        blocks: list[str] = []
        inside_code_fence = False
        current_fence_block: list[str] = []

        for p in raw_paragraphs:
            fence_count = p.count("```")
            if fence_count % 2 != 0:
                inside_code_fence = not inside_code_fence

            if inside_code_fence or fence_count > 0:
                current_fence_block.append(p)
                if not inside_code_fence:
                    blocks.append("\n\n".join(current_fence_block))
                    current_fence_block = []
            else:
                if current_fence_block:
                    current_fence_block.append(p)
                    blocks.append("\n\n".join(current_fence_block))
                    current_fence_block = []
                else:
                    blocks.append(p)

        if current_fence_block:
            blocks.append("\n\n".join(current_fence_block))

        # Further split any single block that exceeds max_chars
        refined: list[str] = []
        for b in blocks:
            if len(b) <= self._config.max_chars:
                refined.append(b)
            else:
                refined.extend(self._split_with_overlap(b))
        return refined

    def _chunk_page_aware(self, document: Document) -> list[Chunk]:
        """Detect form-feed or slide delimiters and emit page-attributed chunks."""
        text = document.content_text.strip()
        pages = _PAGE_MARKER_PATTERN.split(text)
        if len(pages) <= 1:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        page_num = 1

        for part in pages:
            if not part:
                continue
            if part.isdigit():
                page_num = int(part)
                continue

            page_text = part.strip()
            if not page_text:
                continue

            splits = self._split_with_overlap(page_text)
            for split in splits:
                chunks.append(
                    Chunk(
                        document_uri=document.source_uri,
                        chunk_index=chunk_index,
                        chunk_text=split,
                        metadata={
                            "page_number": page_num,
                            "strategy": "page_aware",
                        },
                    )
                )
                chunk_index += 1
            page_num += 1

        return chunks

    def _chunk_fallback(self, document: Document) -> list[Chunk]:
        """Fallback character window chunking for unstructured plain text."""
        text = document.content_text.strip()
        splits = self._split_with_overlap(text)
        chunks: list[Chunk] = []
        current_pos = 0

        for index, chunk_text in enumerate(splits):
            start = text.find(chunk_text, current_pos)
            end = start + len(chunk_text) if start >= 0 else current_pos + len(chunk_text)
            current_pos = max(0, end - self._config.overlap_chars)
            chunks.append(
                Chunk(
                    document_uri=document.source_uri,
                    chunk_index=index,
                    chunk_text=chunk_text,
                    start_offset=max(0, start),
                    end_offset=end,
                    metadata={
                        "strategy": "fallback_window",
                        "start_offset": max(0, start),
                        "end_offset": end,
                    },
                )
            )
        return chunks

    def _split_with_overlap(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self._config.max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self._config.max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - self._config.overlap_chars)
        return chunks
