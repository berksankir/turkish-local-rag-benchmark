from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from turkish_local_rag.config import DownloaderConfig, ResolvedPaths
from turkish_local_rag.download import (
    ExistingFileChangedError,
    HTTPDownloadError,
    PDFValidationError,
    SourceDocument,
    download_document,
    load_manifest,
)


PDF_BYTES = b"%PDF-1.7\nsmall test fixture\n%%EOF\n"
SOURCE = SourceDocument(
    id="test-document",
    title="Test Belgesi",
    source_page_url="https://example.test/sources",
    pdf_url="https://example.test/document.pdf",
)
SETTINGS = DownloaderConfig(
    timeout_seconds=7,
    chunk_size_bytes=8,
    maximum_pdf_bytes=1024,
    user_agent="test-agent/1.0",
)


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str = "application/pdf",
        status: int = 200,
        content_length: int | None = None,
        final_url: str = SOURCE.pdf_url,
    ) -> None:
        self._body = BytesIO(body)
        self.headers = {"Content-Type": content_type}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        else:
            self.headers["Content-Length"] = str(len(body))
        self.status = status
        self._final_url = final_url
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def close(self) -> None:
        self.closed = True
        self._body.close()

    def geturl(self) -> str:
        return self._final_url


def _paths(tmp_path: Path) -> ResolvedPaths:
    return ResolvedPaths(
        project_root=tmp_path,
        source_manifest=tmp_path / "manifest.json",
        pdf_directory=tmp_path / "pdfs",
        metadata_directory=tmp_path / "metadata",
        extracted_pages_directory=tmp_path / "extracted",
        chunks_directory=tmp_path / "chunks",
        embedding_model_directory=tmp_path / "model",
        reranker_model_directory=tmp_path / "reranker",
        qdrant_directory=tmp_path / "qdrant",
        evaluation_candidates=tmp_path / "evaluation" / "candidates.jsonl",
        evaluation_gold=tmp_path / "evaluation" / "gold.jsonl",
        evaluation_results_directory=tmp_path / "evaluation" / "results",
    )


def _opener(response: FakeResponse, calls: list[tuple[Any, int]] | None = None):
    def open_response(request: Any, timeout: int) -> FakeResponse:
        if calls is not None:
            calls.append((request, timeout))
        return response

    return open_response


def test_committed_manifest_contains_nine_unique_sabanci_sources() -> None:
    sources = load_manifest("data/manifest.json")

    assert len(sources) == 9
    assert len({source.id for source in sources}) == 9
    assert all(source.title for source in sources)
    assert all(source.source_page_url.startswith("https://www.sabanciuniv.edu/") for source in sources)
    assert all(source.pdf_url.startswith("https://www.sabanciuniv.edu/") for source in sources)


def test_download_writes_pdf_and_metadata_atomically(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    response = FakeResponse(PDF_BYTES, content_type="application/pdf; charset=binary")
    calls: list[tuple[Any, int]] = []
    fixed_now = datetime(2026, 9, 1, 12, 30, tzinfo=timezone.utc)

    result = download_document(
        SOURCE,
        paths,
        SETTINGS,
        opener=_opener(response, calls),
        now=lambda: fixed_now,
    )

    expected_hash = hashlib.sha256(PDF_BYTES).hexdigest()
    assert result.status == "downloaded"
    assert result.pdf_path.read_bytes() == PDF_BYTES
    assert result.metadata.size_bytes == len(PDF_BYTES)
    assert result.metadata.sha256 == expected_hash
    assert result.metadata.downloaded_at_utc == "2026-09-01T12:30:00Z"
    assert result.metadata.content_type == "application/pdf"
    assert result.metadata.final_url == SOURCE.pdf_url
    assert json.loads(result.metadata_path.read_text(encoding="utf-8"))["sha256"] == expected_hash
    assert calls[0][1] == SETTINGS.timeout_seconds
    assert calls[0][0].get_header("User-agent") == SETTINGS.user_agent
    assert response.closed is True
    assert not list(paths.pdf_directory.glob("*.tmp"))
    assert not list(paths.metadata_directory.glob("*.tmp"))


@pytest.mark.parametrize(
    ("body", "content_type", "message"),
    [
        (PDF_BYTES, "text/html", "unexpected content type"),
        (b"not a pdf", "application/pdf", "invalid PDF signature"),
    ],
)
def test_invalid_pdf_response_is_rejected(
    tmp_path: Path, body: bytes, content_type: str, message: str
) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(PDFValidationError, match=message):
        download_document(
            SOURCE,
            paths,
            SETTINGS,
            opener=_opener(FakeResponse(body, content_type=content_type)),
        )

    assert not (paths.pdf_directory / "test-document.pdf").exists()


def test_http_error_is_reported_without_writing_a_pdf(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    def failing_opener(request: Any, timeout: int) -> FakeResponse:
        raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

    with pytest.raises(HTTPDownloadError, match="HTTP 503"):
        download_document(SOURCE, paths, SETTINGS, opener=failing_opener)

    assert not (paths.pdf_directory / "test-document.pdf").exists()


def test_timeout_is_reported(tmp_path: Path) -> None:
    def timing_out_opener(request: Any, timeout: int) -> FakeResponse:
        raise TimeoutError

    with pytest.raises(HTTPDownloadError, match="timed out.*7s"):
        download_document(SOURCE, _paths(tmp_path), SETTINGS, opener=timing_out_opener)


def test_missing_content_length_is_streamed_and_validated(tmp_path: Path) -> None:
    response = FakeResponse(PDF_BYTES)
    response.headers.pop("Content-Length")

    result = download_document(
        SOURCE, _paths(tmp_path), SETTINGS, opener=_opener(response)
    )

    assert result.metadata.size_bytes == len(PDF_BYTES)


def test_incomplete_body_does_not_leave_a_pdf(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    response = FakeResponse(PDF_BYTES, content_length=len(PDF_BYTES) + 10)

    with pytest.raises(PDFValidationError, match="incomplete PDF download"):
        download_document(SOURCE, paths, SETTINGS, opener=_opener(response))

    assert not (paths.pdf_directory / "test-document.pdf").exists()
    assert not list(paths.pdf_directory.glob("*.tmp"))


def test_oversized_stream_without_content_length_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    response = FakeResponse(PDF_BYTES + b"x" * 1024)
    response.headers.pop("Content-Length")

    with pytest.raises(PDFValidationError, match="exceeds maximum_pdf_bytes"):
        download_document(SOURCE, paths, SETTINGS, opener=_opener(response))

    assert not (paths.pdf_directory / "test-document.pdf").exists()
    assert not list(paths.pdf_directory.glob("*.tmp"))


def test_missing_pdf_trailer_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(PDFValidationError, match="invalid PDF trailer"):
        download_document(
            SOURCE,
            paths,
            SETTINGS,
            opener=_opener(FakeResponse(b"%PDF-1.7\ntruncated")),
        )

    assert not (paths.pdf_directory / "test-document.pdf").exists()


def test_redirect_to_unexpected_domain_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    response = FakeResponse(PDF_BYTES, final_url="https://untrusted.example/file.pdf")

    with pytest.raises(HTTPDownloadError, match="unexpected redirect domain"):
        download_document(SOURCE, paths, SETTINGS, opener=_opener(response))

    assert not (paths.pdf_directory / "test-document.pdf").exists()


def test_existing_different_file_is_never_replaced(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.pdf_directory.mkdir(parents=True)
    target = paths.pdf_directory / "test-document.pdf"
    existing_bytes = b"%PDF-1.4\nexisting local file\n"
    target.write_bytes(existing_bytes)

    with pytest.raises(ExistingFileChangedError) as raised:
        download_document(
            SOURCE,
            paths,
            SETTINGS,
            opener=_opener(FakeResponse(PDF_BYTES)),
        )

    assert target.read_bytes() == existing_bytes
    assert raised.value.existing_sha256 == hashlib.sha256(existing_bytes).hexdigest()
    assert raised.value.downloaded_sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
    assert not list(paths.pdf_directory.glob("*.tmp"))


def test_existing_identical_file_is_reported_unchanged(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.pdf_directory.mkdir(parents=True)
    target = paths.pdf_directory / "test-document.pdf"
    target.write_bytes(PDF_BYTES)

    result = download_document(
        SOURCE,
        paths,
        SETTINGS,
        opener=_opener(FakeResponse(PDF_BYTES)),
    )

    assert result.status == "unchanged"
    assert target.read_bytes() == PDF_BYTES
    assert result.metadata_path.exists()
