"""Extract trusted, page-level text records from locally downloaded PDFs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
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
PAGE_COUNTER_PATTERN = re.compile(r"^(\d+)\s*/\s*(\d+)$")
MISDECODED_INITIAL_ILGILI_PATTERN = re.compile(r"(?<!\w)ılgili(?!\w)")

# Some embedded fonts in the trusted Turkish corpus expose an incorrect
# ToUnicode map. These two non-Turkish code points consistently represent a
# lowercase Latin "i" in the rendered source PDF.
PDF_GLYPH_REPAIRS = str.maketrans({"\u0d74": "i", "\u0d88": "i"})


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

            extracted_page_blocks: list[tuple[int, float, list[dict[str, Any]]]] = []
            for page_index in range(total_pdf_pages):
                page = document.load_page(page_index)
                extracted_page_blocks.append(
                    (
                        page_index + 1,
                        float(page.rect.height),
                        _extract_text_blocks(
                            source.id, page_index + 1, page, settings.sort_blocks
                        ),
                    )
                )
            _remove_repeated_marginal_content(extracted_page_blocks, total_pdf_pages)

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
                for page_number, _page_height, blocks in extracted_page_blocks:
                    page_text = "\n\n".join(block["text"] for block in blocks)
                    raw_page_text = "\n\n".join(block["raw_text"] for block in blocks)
                    if not page_text and not settings.include_empty_pages:
                        continue
                    record = {
                        "schema_version": 1,
                        "document_id": source.id,
                        "title": source.title,
                        "page_number": page_number,
                        "source_page_url": source.source_page_url,
                        "pdf_url": source.pdf_url,
                        "pdf_sha256": pdf_sha256,
                        "text": page_text,
                        "raw_text": raw_page_text,
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
    if paths.corpus_lock.exists():
        from turkish_local_rag.corpus_lock import CorpusLockError, verify_corpus_lock

        try:
            verify_corpus_lock(paths)
        except CorpusLockError as exc:
            raise ExtractionError(str(exc)) from exc
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
        raw_text = str(raw_block[4]).strip()
        text = _normalize_extracted_text(raw_text)
        if not text:
            continue
        block_index = len(extracted)
        extracted.append(
            {
                "block_id": f"{document_id}:p{page_number}:b{block_index}",
                "bbox": [round(float(value), 3) for value in raw_block[:4]],
                "text": text,
                "raw_text": raw_text,
            }
        )
    return extracted


def _repair_pdf_glyphs(text: str) -> str:
    """Repair evidenced non-Turkish glyph mappings in trusted Turkish PDFs."""

    return text.translate(PDF_GLYPH_REPAIRS)


def _normalize_extracted_text(text: str) -> str:
    """Apply only corpus-evidenced PDF text-layer repairs."""

    repaired = _repair_pdf_glyphs(text)
    return MISDECODED_INITIAL_ILGILI_PATTERN.sub("İlgili", repaired)


def _remove_repeated_marginal_content(
    pages: list[tuple[int, float, list[dict[str, Any]]]], total_pages: int
) -> None:
    """Remove repeated header/footer lines and physical page counters in place."""

    if total_pages < 3:
        return
    line_pages: dict[str, set[int]] = {}
    for page_number, page_height, blocks in pages:
        for block in blocks:
            if not _is_marginal(block["bbox"], page_height):
                continue
            for line in block["text"].splitlines():
                stripped = line.strip()
                if stripped:
                    line_pages.setdefault(stripped, set()).add(page_number)

    minimum_pages = max(3, math.ceil(total_pages * 0.8))
    repeated_lines = {
        line for line, page_numbers in line_pages.items() if len(page_numbers) >= minimum_pages
    }

    for page_number, page_height, blocks in pages:
        cleaned_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if _is_marginal(block["bbox"], page_height):
                kept_lines = []
                for line in block["text"].splitlines():
                    stripped = line.strip()
                    counter_match = PAGE_COUNTER_PATTERN.fullmatch(stripped)
                    is_physical_page_counter = bool(
                        counter_match
                        and int(counter_match.group(1)) == page_number
                        and int(counter_match.group(2)) == total_pages
                    )
                    if stripped not in repeated_lines and not is_physical_page_counter:
                        kept_lines.append(line)
                block["text"] = "\n".join(kept_lines).strip()
            if block["text"]:
                cleaned_blocks.append(block)
        for block_index, block in enumerate(cleaned_blocks):
            block["block_id"] = f"{block['block_id'].rsplit(':b', 1)[0]}:b{block_index}"
        blocks[:] = cleaned_blocks


def _is_marginal(bbox: list[float], page_height: float) -> bool:
    return bbox[1] <= page_height * 0.12 or bbox[3] >= page_height * 0.88


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
