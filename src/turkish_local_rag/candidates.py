"""Validate review candidates against trusted extracted physical pages."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from turkish_local_rag.config import load_config
from turkish_local_rag.download import SourceDocument, load_manifest


EXPECTED_TOTAL = 50
EXPECTED_ANSWERABLE = 40
EXPECTED_UNANSWERABLE = 10
CANDIDATE_ID_PATTERN = re.compile(r"^candidate-[0-9]{3}$")
CANDIDATE_KEYS = {
    "candidate_id",
    "question",
    "proposed_reference_answer",
    "required_key_facts",
    "document_id",
    "physical_pages",
    "exact_source_span",
    "answerable",
}


class CandidateValidationError(ValueError):
    """Raised when a candidate set is malformed or not grounded in extraction."""


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    question: str
    proposed_reference_answer: str
    required_key_facts: tuple[str, ...]
    document_id: str | None
    physical_pages: tuple[int, ...]
    exact_source_span: str | None
    answerable: bool


@dataclass(frozen=True, slots=True)
class CandidateValidationReport:
    total: int
    answerable: int
    unanswerable: int
    verified_source_spans: int
    documents_covered: tuple[str, ...]


def load_candidates(path: str | Path) -> tuple[Candidate, ...]:
    """Load strict JSONL candidate records without silently skipping blank lines."""

    candidate_path = Path(path)
    try:
        lines = candidate_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise CandidateValidationError(
            f"candidate file not found: {candidate_path}"
        ) from exc
    if not lines:
        raise CandidateValidationError("candidate file is empty")

    candidates: list[Candidate] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise CandidateValidationError(
                f"blank JSONL record at line {line_number}"
            )
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateValidationError(
                f"invalid JSON at line {line_number}: {exc}"
            ) from exc
        candidates.append(parse_candidate_record(raw, line_number))
    return tuple(candidates)


def validate_candidate_set(
    candidates: Sequence[Candidate],
    sources: Sequence[SourceDocument],
    extracted_pages_directory: str | Path,
    *,
    expected_total: int = EXPECTED_TOTAL,
    expected_answerable: int = EXPECTED_ANSWERABLE,
    expected_unanswerable: int = EXPECTED_UNANSWERABLE,
) -> CandidateValidationReport:
    """Validate counts, identities, and exact spans against extracted page text."""

    if len(candidates) != expected_total:
        raise CandidateValidationError(
            f"expected {expected_total} candidates, found {len(candidates)}"
        )
    answerable = sum(candidate.answerable for candidate in candidates)
    unanswerable = len(candidates) - answerable
    if answerable != expected_answerable or unanswerable != expected_unanswerable:
        raise CandidateValidationError(
            "candidate balance mismatch: "
            f"expected {expected_answerable} answerable/{expected_unanswerable} "
            f"unanswerable, found {answerable}/{unanswerable}"
        )

    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise CandidateValidationError("candidate_id values must be unique")
    questions = [candidate.question.casefold() for candidate in candidates]
    if len(questions) != len(set(questions)):
        raise CandidateValidationError("questions must be unique")

    source_by_id = {source.id: source for source in sources}
    pages_by_document: dict[str, dict[int, str]] = {}
    verified_source_spans = 0
    documents_covered: set[str] = set()
    extracted_directory = Path(extracted_pages_directory)

    for candidate in candidates:
        if not candidate.answerable:
            _validate_unanswerable(candidate)
            continue
        assert candidate.document_id is not None
        assert candidate.exact_source_span is not None
        if candidate.document_id not in source_by_id:
            raise CandidateValidationError(
                f"{candidate.candidate_id}: unknown document_id "
                f"{candidate.document_id!r}"
            )
        if candidate.document_id not in pages_by_document:
            pages_by_document[candidate.document_id] = _load_extracted_pages(
                extracted_directory / f"{candidate.document_id}.pages.jsonl",
                source_by_id[candidate.document_id],
            )
        page_texts = pages_by_document[candidate.document_id]
        missing_pages = [
            page for page in candidate.physical_pages if page not in page_texts
        ]
        if missing_pages:
            raise CandidateValidationError(
                f"{candidate.candidate_id}: missing physical page(s) {missing_pages}"
            )
        matching_pages = [
            page
            for page in candidate.physical_pages
            if candidate.exact_source_span in page_texts[page]
        ]
        if not matching_pages:
            raise CandidateValidationError(
                f"{candidate.candidate_id}: exact_source_span does not exist verbatim "
                "on any declared physical page"
            )
        verified_source_spans += 1
        documents_covered.add(candidate.document_id)

    return CandidateValidationReport(
        total=len(candidates),
        answerable=answerable,
        unanswerable=unanswerable,
        verified_source_spans=verified_source_spans,
        documents_covered=tuple(sorted(documents_covered)),
    )


def parse_candidate_record(raw: Any, line_number: int) -> Candidate:
    """Parse one strict candidate object for candidate, silver, and gold loaders."""

    section = f"line {line_number}"
    record = _mapping(raw, section)
    actual_keys = set(record)
    if actual_keys != CANDIDATE_KEYS:
        missing = CANDIDATE_KEYS - actual_keys
        unknown = actual_keys - CANDIDATE_KEYS
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise CandidateValidationError(f"{section}: invalid fields ({', '.join(details)})")

    candidate_id = _nonempty_string(record, "candidate_id", section)
    if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        raise CandidateValidationError(
            f"{section}.candidate_id must match candidate-NNN"
        )
    question = _nonempty_string(record, "question", section)
    proposed_answer = _nonempty_string(
        record, "proposed_reference_answer", section
    )
    required_key_facts = _string_list(record, "required_key_facts", section)
    answerable = record["answerable"]
    if not isinstance(answerable, bool):
        raise CandidateValidationError(f"{section}.answerable must be boolean")

    document_id = record["document_id"]
    exact_source_span = record["exact_source_span"]
    physical_pages_raw = record["physical_pages"]
    if answerable:
        if not isinstance(document_id, str) or not document_id.strip():
            raise CandidateValidationError(
                f"{section}.document_id must be non-empty for answerable records"
            )
        if not isinstance(exact_source_span, str) or not exact_source_span.strip():
            raise CandidateValidationError(
                f"{section}.exact_source_span must be non-empty for answerable records"
            )
        if not required_key_facts:
            raise CandidateValidationError(
                f"{section}.required_key_facts cannot be empty for answerable records"
            )
    else:
        if document_id is not None or exact_source_span is not None:
            raise CandidateValidationError(
                f"{section}: unanswerable records cannot claim source grounding"
            )

    if not isinstance(physical_pages_raw, list) or any(
        isinstance(page, bool) or not isinstance(page, int) or page <= 0
        for page in physical_pages_raw
    ):
        raise CandidateValidationError(
            f"{section}.physical_pages must contain positive integers"
        )
    physical_pages = tuple(physical_pages_raw)
    if answerable and not physical_pages:
        raise CandidateValidationError(
            f"{section}.physical_pages cannot be empty for answerable records"
        )
    if len(physical_pages) != len(set(physical_pages)):
        raise CandidateValidationError(
            f"{section}.physical_pages cannot contain duplicates"
        )

    return Candidate(
        candidate_id=candidate_id,
        question=question,
        proposed_reference_answer=proposed_answer,
        required_key_facts=required_key_facts,
        document_id=document_id,
        physical_pages=physical_pages,
        exact_source_span=exact_source_span,
        answerable=answerable,
    )


def _validate_unanswerable(candidate: Candidate) -> None:
    if candidate.physical_pages:
        raise CandidateValidationError(
            f"{candidate.candidate_id}: unanswerable physical_pages must be empty"
        )
    if candidate.required_key_facts:
        raise CandidateValidationError(
            f"{candidate.candidate_id}: unanswerable required_key_facts must be empty"
        )


def _load_extracted_pages(path: Path, source: SourceDocument) -> dict[int, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise CandidateValidationError(
            f"extracted pages not found for {source.id}: {path}"
        ) from exc
    pages: dict[int, str] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateValidationError(
                f"invalid extracted JSON for {source.id} at line {line_number}: {exc}"
            ) from exc
        record = _mapping(raw, f"{source.id} extracted line {line_number}")
        if record.get("document_id") != source.id:
            raise CandidateValidationError(
                f"{source.id} extracted line {line_number}: document_id mismatch"
            )
        page_number = record.get("page_number")
        text = record.get("text")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise CandidateValidationError(
                f"{source.id} extracted line {line_number}: invalid page_number"
            )
        if not isinstance(text, str):
            raise CandidateValidationError(
                f"{source.id} extracted line {line_number}: invalid text"
            )
        if page_number in pages:
            raise CandidateValidationError(
                f"{source.id}: duplicate extracted physical page {page_number}"
            )
        pages[page_number] = text
    return pages


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CandidateValidationError(f"{section} must be a JSON object")
    return value


def _nonempty_string(record: Mapping[str, Any], key: str, section: str) -> str:
    value = record[key]
    if not isinstance(value, str) or not value.strip():
        raise CandidateValidationError(f"{section}.{key} must be a non-empty string")
    return value


def _string_list(
    record: Mapping[str, Any], key: str, section: str
) -> tuple[str, ...]:
    value = record[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CandidateValidationError(
            f"{section}.{key} must be a list of non-empty strings"
        )
    return tuple(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Phase 6 candidates against trusted extracted pages."
    )
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = config.resolve_paths(args.config)
    candidates = load_candidates(paths.evaluation_candidates)
    sources = load_manifest(paths.source_manifest)
    report = validate_candidate_set(
        candidates, sources, paths.extracted_pages_directory
    )
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
