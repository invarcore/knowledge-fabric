from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from knowledge_fabric.ingestion import DocumentIngestionService, SourceFormat, TikaClient


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return


def test_ingest_markdown_preserves_source_metadata(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Title\n\nBody", encoding="utf-8")

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    document = service.ingest_file(input_file, source_metadata={"origin": "unit-test"})

    assert document.source_format is SourceFormat.MARKDOWN
    assert document.title == "sample"
    assert "Body" in document.content_text
    assert document.metadata["origin"] == "unit-test"
    assert document.metadata["file_name"] == "sample.md"
    assert document.metadata["file_extension"] == ".md"


def test_ingest_html_strips_tags(tmp_path: Path) -> None:
    input_file = tmp_path / "page.html"
    input_file.write_text("<html><body><h1>Heading</h1><p>Paragraph.</p></body></html>", encoding="utf-8")

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    document = service.ingest_file(input_file)

    assert document.source_format is SourceFormat.HTML
    assert "Heading" in document.content_text
    assert "<h1>" not in document.content_text


def test_ingest_txt(tmp_path: Path) -> None:
    input_file = tmp_path / "notes.txt"
    input_file.write_text("plain text", encoding="utf-8")

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    document = service.ingest_file(input_file)

    assert document.source_format is SourceFormat.TXT
    assert document.content_text == "plain text"


@pytest.mark.parametrize(
    ("name", "expected_format"),
    [
        ("report.pdf", SourceFormat.PDF),
        ("document.docx", SourceFormat.DOCX),
        ("slide.pptx", SourceFormat.PPTX),
    ],
)
def test_ingest_binary_uses_tika(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    expected_format: SourceFormat,
) -> None:
    input_file = tmp_path / name
    input_file.write_bytes(b"binary")

    captured: dict[str, Any] = {}

    def _fake_put(url: str, data: Any, headers: dict[str, str], timeout: int) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse("Extracted from binary")

    monkeypatch.setattr("knowledge_fabric.ingestion.tika_client.requests.put", _fake_put)

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998", timeout_seconds=12))
    document = service.ingest_file(input_file)

    assert document.source_format is expected_format
    assert document.content_text == "Extracted from binary"
    assert captured["url"] == "http://localhost:9998/tika"
    assert captured["timeout"] == 12


def test_ingest_rejects_unsupported_extension(tmp_path: Path) -> None:
    input_file = tmp_path / "unknown.csv"
    input_file.write_text("a,b,c", encoding="utf-8")

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    with pytest.raises(ValueError, match="Unsupported file extension"):
        service.ingest_file(input_file)


def test_ingest_computes_content_hash_and_length(tmp_path: Path) -> None:
    input_file = tmp_path / "hashed.md"
    content = "# Title\n\nSome body text for hash testing."
    input_file.write_text(content, encoding="utf-8")

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    document = service.ingest_file(input_file)

    import hashlib
    expected_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
    assert document.content_hash == expected_hash
    assert document.content_length == len(content.strip())
    assert document.metadata["content_hash"] == expected_hash
    assert document.metadata["content_length"] == len(content.strip())


def test_ingest_enforces_max_file_size(tmp_path: Path) -> None:
    input_file = tmp_path / "large.txt"
    input_file.write_text("X" * 1024, encoding="utf-8")

    service = DocumentIngestionService(
        TikaClient(endpoint="http://localhost:9998"),
        max_file_size_bytes=512,
    )
    with pytest.raises(ValueError, match="exceeds maximum configured limit"):
        service.ingest_file(input_file)


def test_ingest_extracts_headings_from_markdown(tmp_path: Path) -> None:
    input_file = tmp_path / "headings.md"
    content = "# Main Topic\nContent\n## Subtopic A\nContent\n### Detail 1\nContent\n## Subtopic B\nContent"
    input_file.write_text(content, encoding="utf-8")

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    document = service.ingest_file(input_file)

    assert document.headings == ["Main Topic", "Subtopic A", "Detail 1", "Subtopic B"]
    assert document.metadata["headings"] == ["Main Topic", "Subtopic A", "Detail 1", "Subtopic B"]


def test_ingest_extracts_headings_from_html(tmp_path: Path) -> None:
    input_file = tmp_path / "page.html"
    content = "<html><head><title>Page Title</title></head><body><h1>Main Title</h1><h2>Section 1</h2><p>Text</p></body></html>"
    input_file.write_text(content, encoding="utf-8")

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    document = service.ingest_file(input_file)

    assert document.headings == ["Main Title", "Section 1"]
    assert document.metadata["headings"] == ["Main Title", "Section 1"]


def test_ingestion_record_and_status_models() -> None:
    from knowledge_fabric.ingestion.models import IngestionRecord, IngestionStatus

    record = IngestionRecord(
        source_path="/path/to/doc.pdf",
        status=IngestionStatus.ADDED,
        document_id=42,
        chunks_count=15,
        duration_ms=123.4,
    )
    assert record.status is IngestionStatus.ADDED
    assert record.status.value == "added"
    assert record.document_id == 42
    assert record.chunks_count == 15
    assert record.error_message is None


def test_tika_client_endpoint_normalization() -> None:
    client1 = TikaClient(endpoint="http://localhost:9998")
    assert client1._endpoint == "http://localhost:9998/tika"

    client2 = TikaClient(endpoint="http://localhost:9998/")
    assert client2._endpoint == "http://localhost:9998/tika"

    client3 = TikaClient(endpoint="http://localhost:9998/tika")
    assert client3._endpoint == "http://localhost:9998/tika"

    client4 = TikaClient(endpoint="http://localhost:9998/tika/")
    assert client4._endpoint == "http://localhost:9998/tika"


def test_ingest_html_extracts_title_and_headings(tmp_path: Path) -> None:
    input_file = tmp_path / "page_with_title.html"
    input_file.write_text(
        "<html><head><title>Enterprise Architecture Guide</title></head>"
        "<body><h1>Chapter 1: Principles</h1><h2>Core Tenets</h2><p>Content.</p></body></html>",
        encoding="utf-8",
    )

    service = DocumentIngestionService(TikaClient(endpoint="http://localhost:9998"))
    document = service.ingest_file(input_file)

    assert document.source_format is SourceFormat.HTML
    assert document.title == "Enterprise Architecture Guide"
    assert document.headings == ["Chapter 1: Principles", "Core Tenets"]
    assert "Chapter 1: Principles" in document.content_text
    assert "Enterprise Architecture Guide" in document.content_text


