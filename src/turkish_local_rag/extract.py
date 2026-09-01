"""Extract trusted, page-level text records from locally downloaded PDFs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import pymupdf

from turkish_local_rag.config import ExtractionConfig, ResolvedPaths, load_config
from turkish_local_rag.download import SourceDocument, load_manifest, sha256_file


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ExtractionError(RuntimeError):
    """Raised when a local PDF cannot be safely extracted."""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    document_id: str
    output_path: Path
    total_pdf_pages: int
    extracted_pages: int
    text_blocks: int
    characters: int
    pdf_sha256: str


def extract_document(
    source: SourceDocument,
    paths: ResolvedPaths,
    settings: ExtractionConfig,
) -> ExtractionResult:
    """Verify one downloaded PDF and atomically write one JSONL row per page."""

    pdf_path = paths.pdf_directory / f"{source.id}.pdf"
    metadata_path = paths.metadata_directory / f"{source.id}.json"
    pdf_sha256 = _verify_download(source, pdf_path, metadata_path)
    paths.extracted_pages_directory.mkdir(parents=True, exist_ok=True)
    output_path = paths.extracted_pages_directory / f"{source.id}.pages.jsonl"
    temporary_path: Path | None = None

    try:
        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise ExtractionError(f"cannot open PDF for {source.id}: {exc}") from exc

        try:
            if document.needs_pass:
                raise ExtractionError(f"encrypted PDF requires a password: {source.id}")
            total_pdf_pages = document.page_count
            extracted_pages = 0
            text_blocks = 0
            characters = 0

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=paths.extracted_pages_directory,
                prefix=f".{source.id}-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                for page_index in range(total_pdf_pages):
                    page = document.load_page(page_index)
                    blocks = _extract_text_blocks(
                        source.id, page_index + 1, page, settings.sort_blocks
                    )
                    page_text = "\n\n".join(block["text"] for block in blocks)
                    if not page_text and not settings.include_empty_pages:
                        continue
                    record = {
                        "schema_version": 1,
                        "document_id": source.id,
                        "title": source.title,
                        "page_number": page_index + 1,
                        "source_page_url": source.source_page_url,
                        "pdf_url": source.pdf_url,
                        "pdf_sha256": pdf_sha256,
                        "text": page_text,
                        "blocks": blocks,
                    }
                    json.dump(record, temporary_file, ensure_ascii=False)
                    temporary_file.write("\n")
                    extracted_pages += 1
                    text_blocks += len(blocks)
                    characters += len(page_text)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"text extraction failed for {source.id}: {exc}") from exc
        finally:
            document.close()

        os.replace(temporary_path, output_path)
        temporary_path = None
        return ExtractionResult(
            document_id=source.id,
            output_path=output_path,
            total_pdf_pages=total_pdf_pages,
            extracted_pages=extracted_pages,
            text_blocks=text_blocks,
            characters=characters,
            pdf_sha256=pdf_sha256,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_extractions(
    config_path: str | Path, source_ids: Iterable[str] | None = None
) -> tuple[list[ExtractionResult], list[ExtractionError]]:
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    sources = load_manifest(paths.source_manifest)
    selected_ids = set(source_ids or ())
    known_ids = {source.id for source in sources}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise ExtractionError(f"unknown source id(s): {', '.join(sorted(unknown_ids))}")

    results: list[ExtractionResult] = []
    errors: list[ExtractionError] = []
    for source in sources:
        if selected_ids and source.id not in selected_ids:
            continue
        try:
            results.append(extract_document(source, paths, config.extraction))
        except ExtractionError as exc:
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
        help="extract only this manifest id; may be repeated",
    )
    args = parser.parse_args(argv)

    try:
        results, errors = run_extractions(args.config, args.source_id)
    except (ExtractionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for result in results:
        payload = asdict(result)
        payload["output_path"] = str(result.output_path)
        print(json.dumps(payload, ensure_ascii=False))
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


def _extract_text_blocks(
    document_id: str, page_number: int, page: pymupdf.Page, sort_blocks: bool
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for raw_block in page.get_text("blocks", sort=sort_blocks):
        if len(raw_block) < 5:
            continue
        block_type = int(raw_block[6]) if len(raw_block) > 6 else 0
        if block_type != 0:
            continue
        text = str(raw_block[4]).strip()
        if not text:
            continue
        block_index = len(extracted)
        extracted.append(
            {
                "block_id": f"{document_id}:p{page_number}:b{block_index}",
                "bbox": [round(float(value), 3) for value in raw_block[:4]],
                "text": text,
            }
        )
    return extracted


def _verify_download(
    source: SourceDocument, pdf_path: Path, metadata_path: Path
) -> str:
    if not pdf_path.is_file():
        raise ExtractionError(f"downloaded PDF not found for {source.id}: {pdf_path}")
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExtractionError(
            f"download metadata not found for {source.id}: {metadata_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"invalid download metadata for {source.id}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExtractionError(f"download metadata must be an object for {source.id}")

    required = {
        "id",
        "title",
        "source_page_url",
        "pdf_url",
        "final_url",
        "downloaded_at_utc",
        "size_bytes",
        "sha256",
        "content_type",
        "local_filename",
    }
    missing = required - set(raw)
    if missing:
        raise ExtractionError(
            f"download metadata missing field(s) for {source.id}: "
            f"{', '.join(sorted(missing))}"
        )
    for field in ("id", "title", "source_page_url", "pdf_url"):
        if raw[field] != getattr(source, field):
            raise ExtractionError(f"untrusted metadata mismatch for {source.id}: {field}")
    final_url = raw["final_url"]
    parsed_final_url = urlparse(final_url) if isinstance(final_url, str) else None
    source_hostname = urlparse(source.pdf_url).hostname
    if (
        parsed_final_url is None
        or parsed_final_url.scheme not in {"http", "https"}
        or not parsed_final_url.hostname
        or parsed_final_url.hostname.casefold() != (source_hostname or "").casefold()
    ):
        raise ExtractionError(f"invalid metadata final_url for {source.id}")
    if raw["local_filename"] != pdf_path.name:
        raise ExtractionError(f"untrusted metadata mismatch for {source.id}: local_filename")
    if raw["content_type"] != "application/pdf":
        raise ExtractionError(f"invalid metadata content_type for {source.id}")
    if isinstance(raw["size_bytes"], bool) or not isinstance(raw["size_bytes"], int):
        raise ExtractionError(f"invalid metadata size_bytes for {source.id}")
    if raw["size_bytes"] != pdf_path.stat().st_size:
        raise ExtractionError(f"PDF size differs from download metadata for {source.id}")
    expected_sha256 = raw["sha256"]
    if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
        expected_sha256
    ):
        raise ExtractionError(f"invalid metadata sha256 for {source.id}")
    actual_sha256 = sha256_file(pdf_path)
    if actual_sha256 != expected_sha256:
        raise ExtractionError(
            f"PDF hash differs from download metadata for {source.id}: "
            f"expected_sha256={expected_sha256}, actual_sha256={actual_sha256}"
        )
    return actual_sha256


if __name__ == "__main__":
    raise SystemExit(main())
