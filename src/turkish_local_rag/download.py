"""Download manifest PDFs safely without third-party runtime dependencies."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from turkish_local_rag.config import (
    DownloaderConfig,
    ResolvedPaths,
    load_config,
)


PDF_SIGNATURE = b"%PDF-"
PDF_EOF_MARKER = b"%%EOF"
PDF_TRAILER_WINDOW_BYTES = 4096
DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class DownloadError(RuntimeError):
    """Base error for manifest and download failures."""


class ManifestError(DownloadError):
    """Raised when the committed source manifest is invalid."""


class HTTPDownloadError(DownloadError):
    """Raised when the remote server cannot provide a successful response."""


class PDFValidationError(DownloadError):
    """Raised when a response does not pass PDF content validation."""


class ExistingFileChangedError(DownloadError):
    """Raised when a remote download differs from an existing local PDF."""

    def __init__(
        self, document_id: str, path: Path, existing_sha256: str, downloaded_sha256: str
    ) -> None:
        self.document_id = document_id
        self.path = path
        self.existing_sha256 = existing_sha256
        self.downloaded_sha256 = downloaded_sha256
        super().__init__(
            f"hash change detected for {document_id}; existing file was not replaced: "
            f"existing_sha256={existing_sha256}, downloaded_sha256={downloaded_sha256}"
        )


class CorpusLockMismatchError(DownloadError):
    """Raised when downloaded bytes differ from the committed corpus lock."""


@dataclass(frozen=True, slots=True)
class SourceDocument:
    id: str
    title: str
    source_page_url: str
    pdf_url: str


@dataclass(frozen=True, slots=True)
class DownloadMetadata:
    id: str
    title: str
    source_page_url: str
    pdf_url: str
    final_url: str
    downloaded_at_utc: str
    size_bytes: int
    sha256: str
    content_type: str
    local_filename: str


@dataclass(frozen=True, slots=True)
class DownloadResult:
    status: str
    pdf_path: Path
    metadata_path: Path
    metadata: DownloadMetadata


def load_manifest(path: str | Path) -> tuple[SourceDocument, ...]:
    """Load and strictly validate the committed JSON source manifest."""

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {manifest_path}: {exc}") from exc

    root = _mapping(raw, "manifest root")
    _exact_keys(root, {"schema_version", "documents"}, "manifest root")
    if root["schema_version"] != 1:
        raise ManifestError(f"unsupported manifest schema_version: {root['schema_version']}")
    documents = root["documents"]
    if not isinstance(documents, list) or not documents:
        raise ManifestError("manifest documents must be a non-empty list")

    sources: list[SourceDocument] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(documents):
        section = f"documents[{index}]"
        document = _mapping(item, section)
        _exact_keys(document, {"id", "title", "source_page_url", "pdf_url"}, section)
        source = SourceDocument(
            id=_nonempty_string(document, "id", section),
            title=_nonempty_string(document, "title", section),
            source_page_url=_web_url(document, "source_page_url", section),
            pdf_url=_web_url(document, "pdf_url", section),
        )
        if not DOCUMENT_ID_PATTERN.fullmatch(source.id):
            raise ManifestError(f"{section}.id must be a lowercase ASCII slug")
        if source.id in seen_ids:
            raise ManifestError(f"duplicate document id: {source.id}")
        seen_ids.add(source.id)
        sources.append(source)
    return tuple(sources)


def download_document(
    source: SourceDocument,
    paths: ResolvedPaths,
    settings: DownloaderConfig,
    *,
    opener: Callable[..., Any] = urlopen,
    now: Callable[[], datetime] | None = None,
    lock_record: Any | None = None,
) -> DownloadResult:
    """Download one PDF atomically and refuse silent replacement on hash changes."""

    paths.pdf_directory.mkdir(parents=True, exist_ok=True)
    paths.metadata_directory.mkdir(parents=True, exist_ok=True)
    target_path = paths.pdf_directory / f"{source.id}.pdf"
    metadata_path = paths.metadata_directory / f"{source.id}.json"
    temporary_path: Path | None = None
    request = Request(
        source.pdf_url,
        headers={"Accept": "application/pdf", "User-Agent": settings.user_agent},
    )

    try:
        response = opener(request, timeout=settings.timeout_seconds)
        with closing(response):
            status = getattr(response, "status", None)
            if status is not None and not 200 <= status < 300:
                raise HTTPDownloadError(
                    f"HTTP {status} while downloading {source.id} from {source.pdf_url}"
                )
            content_type = _normalized_content_type(response.headers)
            if content_type != "application/pdf":
                raise PDFValidationError(
                    f"unexpected content type for {source.id}: {content_type or '<missing>'}"
                )
            final_url = _validated_final_url(response, source)
            content_length = _validate_content_length(
                response.headers, source.id, settings.maximum_pdf_bytes
            )

            first_chunk = response.read(settings.chunk_size_bytes)
            if not first_chunk.startswith(PDF_SIGNATURE):
                raise PDFValidationError(
                    f"invalid PDF signature for {source.id}; expected %PDF-"
                )

            digest = hashlib.sha256()
            size_bytes = 0
            trailing_bytes = b""
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=paths.pdf_directory,
                prefix=f".{source.id}-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                for chunk in _response_chunks(
                    response, first_chunk, settings.chunk_size_bytes
                ):
                    size_bytes += len(chunk)
                    if size_bytes > settings.maximum_pdf_bytes:
                        raise PDFValidationError(
                            f"PDF exceeds maximum_pdf_bytes for {source.id}: "
                            f"{settings.maximum_pdf_bytes}"
                        )
                    digest.update(chunk)
                    temporary_file.write(chunk)
                    trailing_bytes = (trailing_bytes + chunk)[
                        -PDF_TRAILER_WINDOW_BYTES:
                    ]
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            if content_length is not None and size_bytes != content_length:
                raise PDFValidationError(
                    f"incomplete PDF download for {source.id}: "
                    f"expected_bytes={content_length}, received_bytes={size_bytes}"
                )
            if PDF_EOF_MARKER not in trailing_bytes:
                raise PDFValidationError(
                    f"invalid PDF trailer for {source.id}; expected %%EOF near file end"
                )

        downloaded_sha256 = digest.hexdigest()
        if lock_record is not None and (
            lock_record.document_id != source.id
            or lock_record.title != source.title
            or lock_record.source_page_url != source.source_page_url
            or lock_record.pdf_url != source.pdf_url
            or lock_record.final_pdf_url != final_url
            or lock_record.size_bytes != size_bytes
            or lock_record.sha256 != downloaded_sha256
        ):
            raise CorpusLockMismatchError(
                f"download differs from committed corpus lock for {source.id}; "
                "lock and existing PDF were not replaced"
            )
        status = "downloaded"
        if target_path.exists():
            existing_sha256 = sha256_file(target_path, settings.chunk_size_bytes)
            if existing_sha256 != downloaded_sha256:
                raise ExistingFileChangedError(
                    source.id, target_path, existing_sha256, downloaded_sha256
                )
            status = "unchanged"
        else:
            os.replace(temporary_path, target_path)
            temporary_path = None

        timestamp = (now or _utc_now)().astimezone(timezone.utc)
        downloaded_at_utc = (
            lock_record.downloaded_at_utc
            if lock_record is not None
            else timestamp.isoformat().replace("+00:00", "Z")
        )
        metadata = DownloadMetadata(
            id=source.id,
            title=source.title,
            source_page_url=source.source_page_url,
            pdf_url=source.pdf_url,
            final_url=final_url,
            downloaded_at_utc=downloaded_at_utc,
            size_bytes=size_bytes,
            sha256=downloaded_sha256,
            content_type=content_type,
            local_filename=target_path.name,
        )
        _write_json_atomic(metadata_path, asdict(metadata))
        return DownloadResult(status, target_path, metadata_path, metadata)
    except HTTPError as exc:
        raise HTTPDownloadError(
            f"HTTP {exc.code} while downloading {source.id} from {source.pdf_url}"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise HTTPDownloadError(
            f"download timed out for {source.id} after {settings.timeout_seconds}s"
        ) from exc
    except URLError as exc:
        raise HTTPDownloadError(f"network error while downloading {source.id}: {exc.reason}") from exc
    except http.client.IncompleteRead as exc:
        raise HTTPDownloadError(f"incomplete HTTP response while downloading {source.id}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sha256_file(path: str | Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_downloads(
    config_path: str | Path,
    source_ids: Iterable[str] | None = None,
    *,
    opener: Callable[..., Any] = urlopen,
) -> tuple[list[DownloadResult], list[DownloadError]]:
    """Run configured downloads while allowing independent documents to continue."""

    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    sources = load_manifest(paths.source_manifest)
    lock_by_id: dict[str, Any] = {}
    if paths.corpus_lock.exists():
        from turkish_local_rag.corpus_lock import (
            CorpusLockError,
            verify_corpus_lock,
        )

        try:
            lock_by_id = {
                record.document_id: record
                for record in verify_corpus_lock(paths, require_local_files=False)
            }
        except CorpusLockError as exc:
            raise ManifestError(str(exc)) from exc
    selected_ids = set(source_ids or ())
    known_ids = {source.id for source in sources}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise ManifestError(f"unknown source id(s): {', '.join(sorted(unknown_ids))}")

    results: list[DownloadResult] = []
    errors: list[DownloadError] = []
    for source in sources:
        if selected_ids and source.id not in selected_ids:
            continue
        try:
            results.append(
                download_document(
                    source,
                    paths,
                    config.downloader,
                    opener=opener,
                    lock_record=lock_by_id.get(source.id),
                )
            )
        except DownloadError as exc:
            errors.append(exc)
    return results, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/default.toml", help="TOML config path"
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="download only this manifest id; may be repeated",
    )
    args = parser.parse_args(argv)

    try:
        results, errors = run_downloads(args.config, args.source_id)
    except (DownloadError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(
            json.dumps(
                {
                    "id": result.metadata.id,
                    "status": result.status,
                    "size_bytes": result.metadata.size_bytes,
                    "sha256": result.metadata.sha256,
                    "pdf_path": str(result.pdf_path),
                    "metadata_path": str(result.metadata_path),
                },
                ensure_ascii=False,
            )
        )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


def _response_chunks(response: Any, first_chunk: bytes, chunk_size: int) -> Iterable[bytes]:
    yield first_chunk
    while chunk := response.read(chunk_size):
        yield chunk


def _normalized_content_type(headers: Any) -> str:
    return str(headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()


def _validate_content_length(
    headers: Any, document_id: str, maximum_bytes: int
) -> int | None:
    raw_length = headers.get("Content-Length")
    if raw_length is None:
        return None
    try:
        content_length = int(raw_length)
    except (TypeError, ValueError) as exc:
        raise PDFValidationError(
            f"invalid Content-Length for {document_id}: {raw_length}"
        ) from exc
    if content_length < len(PDF_SIGNATURE):
        raise PDFValidationError(
            f"invalid Content-Length for {document_id}: {content_length}"
        )
    if content_length > maximum_bytes:
        raise PDFValidationError(
            f"PDF exceeds maximum_pdf_bytes for {document_id}: {content_length}"
        )
    return content_length


def _validated_final_url(response: Any, source: SourceDocument) -> str:
    geturl = getattr(response, "geturl", None)
    final_url = geturl() if callable(geturl) else source.pdf_url
    if not isinstance(final_url, str) or not final_url:
        raise HTTPDownloadError(f"invalid final URL while downloading {source.id}")
    original = urlparse(source.pdf_url)
    final = urlparse(final_url)
    if final.scheme not in {"http", "https"} or not final.hostname:
        raise HTTPDownloadError(
            f"invalid final URL while downloading {source.id}: {final_url}"
        )
    if original.scheme == "https" and final.scheme != "https":
        raise HTTPDownloadError(
            f"HTTPS downgrade rejected for {source.id}: {final_url}"
        )
    if final.hostname.casefold() != (original.hostname or "").casefold():
        raise HTTPDownloadError(
            f"unexpected redirect domain for {source.id}: "
            f"{original.hostname} -> {final.hostname}"
        )
    return final_url


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{section} must be a JSON object")
    return value


def _exact_keys(table: Mapping[str, Any], expected: set[str], section: str) -> None:
    actual = set(table)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ManifestError(f"missing field(s) in {section}: {', '.join(sorted(missing))}")
    if unknown:
        raise ManifestError(f"unknown field(s) in {section}: {', '.join(sorted(unknown))}")


def _nonempty_string(table: Mapping[str, Any], key: str, section: str) -> str:
    value = table[key]
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{section}.{key} must be a non-empty string")
    return value


def _web_url(table: Mapping[str, Any], key: str, section: str) -> str:
    value = _nonempty_string(table, key, section)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ManifestError(f"{section}.{key} must be an HTTP(S) URL")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
