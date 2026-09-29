"""Document ingestion service."""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from knowledge_fabric.ingestion.models import Document, SourceFormat
from knowledge_fabric.ingestion.tika_client import TikaClient

_TEXT_EXTENSIONS: dict[str, SourceFormat] = {
    ".md": SourceFormat.MARKDOWN,
    ".markdown": SourceFormat.MARKDOWN,
    ".txt": SourceFormat.TXT,
    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,
}

_BINARY_EXTENSIONS: dict[str, SourceFormat] = {
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
    ".pptx": SourceFormat.PPTX,
}

_DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB
_MD_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_HTML_TITLE_PATTERN = re.compile(r"(?i)<title\b[^>]*>(.*?)</title>")
_HTML_HEADING_PATTERN = re.compile(r"(?i)<h[1-6]\b[^>]*>(.*?)</h[1-6]>")


class DocumentIngestionService:
    """Ingest source files into canonical `Document` models with security bounds."""

    def __init__(
        self,
        tika_client: TikaClient,
        max_file_size_bytes: int = _DEFAULT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        self._tika_client = tika_client
        self._max_file_size_bytes = max_file_size_bytes

    def ingest_file(
        self,
        file_path: str | Path,
        source_metadata: dict[str, object] | None = None,
    ) -> Document:
        """Ingest one supported file, enforce size bounds, and preserve source metadata."""
        resolved = Path(file_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")

        file_size = resolved.stat().st_size
        if file_size > self._max_file_size_bytes:
            raise ValueError(
                f"File size {file_size} bytes exceeds maximum configured limit of {self._max_file_size_bytes} bytes: {resolved.name}"
            )

        source_format = self._resolve_source_format(resolved.suffix.lower())
        title = resolved.stem
        headings: list[str] = []

        if source_format in _TEXT_EXTENSIONS.values():
            raw_text = resolved.read_text(encoding="utf-8", errors="replace")
            if source_format is SourceFormat.MARKDOWN:
                content_text = raw_text
                # Extract markdown headings
                extracted_headings = [m.group(2).strip() for m in _MD_HEADING_PATTERN.finditer(raw_text)]
                if extracted_headings:
                    headings = extracted_headings
            elif source_format is SourceFormat.HTML:
                # Extract HTML title and headings
                title_match = _HTML_TITLE_PATTERN.search(raw_text)
                if title_match:
                    extracted_title = unescape(title_match.group(1)).strip()
                    if extracted_title:
                        title = extracted_title
                headings = [unescape(m.group(1)).strip() for m in _HTML_HEADING_PATTERN.finditer(raw_text)]
                content_text = _strip_html(raw_text)
            else:
                content_text = raw_text
        else:
            content_text = self._tika_client.extract_text(resolved)

        metadata = {
            "source_path": str(resolved),
            "file_name": resolved.name,
            "file_extension": resolved.suffix.lower(),
            "file_size_bytes": file_size,
        }
        if source_metadata:
            metadata.update(source_metadata)

        return Document(
            source_uri=str(resolved),
            source_format=source_format,
            content_text=content_text.strip(),
            title=title,
            headings=headings,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_source_format(extension: str) -> SourceFormat:
        if extension in _TEXT_EXTENSIONS:
            return _TEXT_EXTENSIONS[extension]
        if extension in _BINARY_EXTENSIONS:
            return _BINARY_EXTENSIONS[extension]
        raise ValueError(f"Unsupported file extension: {extension}")


def _strip_html(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
