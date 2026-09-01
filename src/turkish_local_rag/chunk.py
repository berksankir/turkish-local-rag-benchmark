"""Create deterministic, page-bounded chunks from extracted page JSONL."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from turkish_local_rag.config import ChunkingConfig, ResolvedPaths, load_config
from turkish_local_rag.download import SourceDocument, load_manifest


ARTICLE_PATTERN = re.compile(r"^(?:GEÇİCİ\s+)?MADDE\s+\d+", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^[A-ZÇĞİÖŞÜ]+\s+BÖLÜM\b", re.IGNORECASE)
PARAGRAPH_PATTERN = re.compile(r"^(?:\(\d+\)|[a-zçğıöşü]\))\s+", re.IGNORECASE)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
LEXEME_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PAGE_COUNTER_LINE_PATTERN = re.compile(
    r"^(?:\d+\s*/\s*\d+|.*\bSayfa\s+\d+\s*/\s*\d+)\s*$", re.IGNORECASE
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TOKEN_COUNT_METHOD = "max(unicode_lexemes,ceil(characters/configured_chars_per_token))"


class ChunkingError(RuntimeError):
    """Raised when extracted page data cannot be safely chunked."""


@dataclass(frozen=True, slots=True)
class StructuralUnit:
    text: str
    block_ids: tuple[str, ...]
    force_boundary_before: bool = False
    overlap: bool = False


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    document_id: str
    output_path: Path
    input_pages: int
    chunks: int
    maximum_estimated_tokens: int


def estimate_tokens(text: str, characters_per_token: int) -> int:
    """Return a deterministic conservative estimate, never a model-token claim."""

    if not text:
        return 0
    lexeme_count = len(LEXEME_PATTERN.findall(text))
    character_estimate = math.ceil(len(text) / characters_per_token)
    return max(1, lexeme_count, character_estimate)


def chunk_document(
    source: SourceDocument,
    paths: ResolvedPaths,
    settings: ChunkingConfig,
) -> ChunkingResult:
    input_path = paths.extracted_pages_directory / f"{source.id}.pages.jsonl"
    pages = _load_extracted_pages(input_path, source)
    paths.chunks_directory.mkdir(parents=True, exist_ok=True)
    output_path = paths.chunks_directory / f"{source.id}.chunks.jsonl"
    temporary_path: Path | None = None
    chunk_count = 0
    maximum_estimated_tokens = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=paths.chunks_directory,
            prefix=f".{source.id}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for page in pages:
                units = _structural_units(
                    page["blocks"], settings.preserve_article_boundaries
                )
                chunks = [
                    chunk_units
                    for chunk_units in _pack_units(units, settings)
                    if not _is_publication_masthead_only(_join_units(chunk_units))
                ]
                for page_chunk_index, chunk_units in enumerate(chunks):
                    text = _join_units(chunk_units)
                    estimated_tokens = estimate_tokens(
                        text, settings.estimated_characters_per_token
                    )
                    if estimated_tokens > settings.maximum_model_tokens:
                        raise ChunkingError(
                            f"chunk exceeds maximum_model_tokens for {source.id} "
                            f"page {page['page_number']}: {estimated_tokens}"
                        )
                    overlap_text = _join_units(
                        [unit for unit in chunk_units if unit.overlap]
                    )
                    block_ids = tuple(
                        dict.fromkeys(
                            block_id
                            for unit in chunk_units
                            for block_id in unit.block_ids
                        )
                    )
                    record = {
                        "schema_version": 1,
                        "chunk_id": (
                            f"{source.id}:p{page['page_number']}:c{page_chunk_index}"
                        ),
                        "document_id": source.id,
                        "title": source.title,
                        "page_number": page["page_number"],
                        "source_page_url": source.source_page_url,
                        "pdf_url": source.pdf_url,
                        "pdf_sha256": page["pdf_sha256"],
                        "source_block_ids": list(block_ids),
                        "text": text,
                        "estimated_tokens": estimated_tokens,
                        "overlap_estimated_tokens": estimate_tokens(
                            overlap_text, settings.estimated_characters_per_token
                        ),
                        "token_count_method": TOKEN_COUNT_METHOD,
                    }
                    json.dump(record, temporary_file, ensure_ascii=False)
                    temporary_file.write("\n")
                    chunk_count += 1
                    maximum_estimated_tokens = max(
                        maximum_estimated_tokens, estimated_tokens
                    )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ChunkingResult(
        document_id=source.id,
        output_path=output_path,
        input_pages=len(pages),
        chunks=chunk_count,
        maximum_estimated_tokens=maximum_estimated_tokens,
    )


def run_chunking(
    config_path: str | Path, source_ids: Iterable[str] | None = None
) -> tuple[list[ChunkingResult], list[ChunkingError]]:
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    sources = load_manifest(paths.source_manifest)
    selected_ids = set(source_ids or ())
    known_ids = {source.id for source in sources}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise ChunkingError(f"unknown source id(s): {', '.join(sorted(unknown_ids))}")

    results: list[ChunkingResult] = []
    errors: list[ChunkingError] = []
    for source in sources:
        if selected_ids and source.id not in selected_ids:
            continue
        try:
            results.append(chunk_document(source, paths, config.chunking))
        except ChunkingError as exc:
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
        help="chunk only this manifest id; may be repeated",
    )
    args = parser.parse_args(argv)

    try:
        results, errors = run_chunking(args.config, args.source_id)
    except (ChunkingError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for result in results:
        payload = asdict(result)
        payload["output_path"] = str(result.output_path)
        print(json.dumps(payload, ensure_ascii=False))
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


def _load_extracted_pages(
    path: Path, source: SourceDocument
) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ChunkingError(f"extracted pages not found for {source.id}: {path}") from exc

    pages: list[Mapping[str, Any]] = []
    previous_page_number = 0
    expected_hash: str | None = None
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ChunkingError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ChunkingError(f"page record must be an object at {path}:{line_number}")
        required = {
            "schema_version",
            "document_id",
            "title",
            "page_number",
            "source_page_url",
            "pdf_url",
            "pdf_sha256",
            "text",
            "blocks",
        }
        missing = required - set(raw)
        if missing:
            raise ChunkingError(
                f"page record missing field(s) at {path}:{line_number}: "
                f"{', '.join(sorted(missing))}"
            )
        if raw["schema_version"] != 1:
            raise ChunkingError(f"unsupported page schema at {path}:{line_number}")
        for field in ("document_id", "title", "source_page_url", "pdf_url"):
            expected = source.id if field == "document_id" else getattr(source, field)
            if raw[field] != expected:
                raise ChunkingError(
                    f"untrusted page metadata mismatch at {path}:{line_number}: {field}"
                )
        page_number = raw["page_number"]
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise ChunkingError(f"invalid page_number at {path}:{line_number}")
        if page_number <= previous_page_number:
            raise ChunkingError(f"page numbers must be strictly increasing in {path}")
        previous_page_number = page_number
        pdf_sha256 = raw["pdf_sha256"]
        if not isinstance(pdf_sha256, str) or not SHA256_PATTERN.fullmatch(pdf_sha256):
            raise ChunkingError(f"invalid pdf_sha256 at {path}:{line_number}")
        if expected_hash is not None and pdf_sha256 != expected_hash:
            raise ChunkingError(f"inconsistent pdf_sha256 values in {path}")
        expected_hash = pdf_sha256
        if not isinstance(raw["blocks"], list):
            raise ChunkingError(f"blocks must be a list at {path}:{line_number}")
        if not isinstance(raw["text"], str):
            raise ChunkingError(f"text must be a string at {path}:{line_number}")
        expected_block_prefix = f"{source.id}:p{page_number}:b"
        for block_index, block in enumerate(raw["blocks"]):
            if not isinstance(block, dict) or not isinstance(block.get("block_id"), str):
                raise ChunkingError(
                    f"invalid block at {path}:{line_number} index {block_index}"
                )
            if not block["block_id"].startswith(expected_block_prefix):
                raise ChunkingError(
                    f"untrusted block_id at {path}:{line_number} index {block_index}"
                )
        pages.append(raw)
    return pages


def _structural_units(
    blocks: Sequence[Any], preserve_article_boundaries: bool
) -> list[StructuralUnit]:
    units: list[StructuralUnit] = []
    for block_index, raw_block in enumerate(blocks):
        if not isinstance(raw_block, dict):
            raise ChunkingError(f"block {block_index} must be an object")
        block_id = raw_block.get("block_id")
        text = raw_block.get("text")
        if not isinstance(block_id, str) or not block_id:
            raise ChunkingError(f"block {block_index} has invalid block_id")
        if not isinstance(text, str):
            raise ChunkingError(f"block {block_index} has invalid text")
        current_lines: list[str] = []
        current_force_boundary = False

        def flush() -> None:
            nonlocal current_lines, current_force_boundary
            if current_lines:
                units.append(
                    StructuralUnit(
                        text="\n".join(current_lines),
                        block_ids=(block_id,),
                        force_boundary_before=current_force_boundary,
                    )
                )
            current_lines = []
            current_force_boundary = False

        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if PAGE_COUNTER_LINE_PATTERN.fullmatch(line):
                flush()
                continue
            is_article = bool(ARTICLE_PATTERN.match(line))
            if is_article:
                article_heading: list[str] = []
                while current_lines and _looks_like_article_heading(current_lines[-1]):
                    article_heading.insert(0, current_lines.pop())
                flush()
                current_lines.extend(article_heading)
                current_force_boundary = preserve_article_boundaries
                current_lines.append(line)
                continue
            starts_structure = (
                bool(SECTION_PATTERN.match(line))
                or bool(PARAGRAPH_PATTERN.match(line))
            )
            if starts_structure and current_lines:
                flush()
            if not current_lines:
                current_force_boundary = is_article and preserve_article_boundaries
            current_lines.append(line)
        flush()
    merged: list[StructuralUnit] = []
    for unit in units:
        if unit.force_boundary_before:
            headings: list[StructuralUnit] = []
            while merged and _is_heading_unit(merged[-1]):
                headings.insert(0, merged.pop())
            if headings:
                unit = replace(
                    unit,
                    text="\n".join([*(heading.text for heading in headings), unit.text]),
                    block_ids=tuple(
                        dict.fromkeys(
                            block_id
                            for item in [*headings, unit]
                            for block_id in item.block_ids
                        )
                    ),
                )
        merged.append(unit)
    return merged


def _looks_like_article_heading(line: str) -> bool:
    """Identify a short regulation heading immediately preceding a MADDE line."""

    stripped = line.strip()
    return bool(
        stripped
        and len(stripped) <= 100
        and len(stripped.split()) <= 12
        and not ARTICLE_PATTERN.match(stripped)
        and not PARAGRAPH_PATTERN.match(stripped)
        and stripped[-1] not in ".!?;:”“\""
    )


def _is_heading_unit(unit: StructuralUnit) -> bool:
    lines = [line.strip() for line in unit.text.splitlines() if line.strip()]
    return bool(
        lines
        and not _is_publication_masthead_only(unit.text)
        and len(unit.text) <= 200
        and len(unit.text.split()) <= 20
        and all(_looks_like_article_heading(line) for line in lines)
    )


def _is_publication_masthead_only(text: str) -> bool:
    folded = text.casefold().replace("î", "i")
    return bool(
        "resmi gazete" in folded
        and ("sayı" in folded or "sayi" in folded or "resmi gazete no" in folded)
        and "madde" not in folded
    )


def _pack_units(
    units: Sequence[StructuralUnit], settings: ChunkingConfig
) -> list[list[StructuralUnit]]:
    expanded = [
        piece
        for unit in units
        for piece in _split_oversized_unit(
            unit,
            settings.target_model_tokens,
            settings.estimated_characters_per_token,
        )
    ]
    chunks: list[list[StructuralUnit]] = []
    current: list[StructuralUnit] = []
    for unit in expanded:
        candidate = _join_units([*current, unit])
        candidate_tokens = estimate_tokens(
            candidate, settings.estimated_characters_per_token
        )
        if current and (
            unit.force_boundary_before
            or candidate_tokens > settings.target_model_tokens
        ):
            chunks.append(current)
            current = [] if unit.force_boundary_before else _overlap_tail(current, settings)
        if current:
            candidate = _join_units([*current, unit])
            if (
                estimate_tokens(candidate, settings.estimated_characters_per_token)
                > settings.maximum_model_tokens
            ):
                current = []
        current.append(unit)
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_unit(
    unit: StructuralUnit, target_tokens: int, characters_per_token: int
) -> list[StructuralUnit]:
    if estimate_tokens(unit.text, characters_per_token) <= target_tokens:
        return [unit]
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_PATTERN.split(unit.text)
        if sentence.strip()
    ]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        sentence_parts = _split_words(sentence, target_tokens, characters_per_token)
        for part in sentence_parts:
            candidate = f"{current} {part}".strip()
            if current and estimate_tokens(candidate, characters_per_token) > target_tokens:
                pieces.append(current)
                current = part
            else:
                current = candidate
    if current:
        pieces.append(current)
    return [
        StructuralUnit(
            text=piece,
            block_ids=unit.block_ids,
            force_boundary_before=unit.force_boundary_before and index == 0,
        )
        for index, piece in enumerate(pieces)
    ]


def _split_words(
    text: str, target_tokens: int, characters_per_token: int
) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and estimate_tokens(candidate, characters_per_token) > target_tokens:
            pieces.append(current)
            current = word
        else:
            current = candidate
        if estimate_tokens(current, characters_per_token) > target_tokens:
            maximum_characters = max(1, target_tokens * characters_per_token)
            pieces.extend(
                current[index : index + maximum_characters]
                for index in range(0, len(current), maximum_characters)
            )
            current = ""
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(
    units: Sequence[StructuralUnit], settings: ChunkingConfig
) -> list[StructuralUnit]:
    if settings.overlap_model_tokens == 0:
        return []
    selected: list[StructuralUnit] = []
    for unit in reversed(units):
        candidate = [replace(unit, overlap=True), *selected]
        if (
            estimate_tokens(
                _join_units(candidate), settings.estimated_characters_per_token
            )
            <= settings.overlap_model_tokens
        ):
            selected = candidate
        else:
            tail = _word_tail(unit, settings)
            if tail is not None:
                selected.insert(0, tail)
            break
    return selected


def _word_tail(
    unit: StructuralUnit, settings: ChunkingConfig
) -> StructuralUnit | None:
    selected_words: list[str] = []
    for word in reversed(unit.text.split()):
        candidate = " ".join([word, *selected_words])
        if (
            estimate_tokens(candidate, settings.estimated_characters_per_token)
            > settings.overlap_model_tokens
        ):
            break
        selected_words.insert(0, word)
    if not selected_words:
        return None
    return StructuralUnit(
        text=" ".join(selected_words),
        block_ids=unit.block_ids,
        overlap=True,
    )


def _join_units(units: Sequence[StructuralUnit]) -> str:
    return "\n\n".join(unit.text for unit in units)


if __name__ == "__main__":
    raise SystemExit(main())
