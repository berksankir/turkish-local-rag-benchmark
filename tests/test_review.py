from dataclasses import replace
import csv
import json
from pathlib import Path

import pytest

from turkish_local_rag.candidates import Candidate
from turkish_local_rag.download import SourceDocument
from turkish_local_rag.review import (
    ReviewError,
    ReviewRecord,
    approved_candidates,
    approved_split_counts,
    build_gold_records,
    build_silver_records,
    load_review,
    select_silver_audit_candidates,
    validate_review_grounding,
    write_gold,
    write_pending_review,
    write_silver,
)


SOURCE = SourceDocument(
    id="test-document",
    title="Test Document",
    source_page_url="https://example.test/source",
    pdf_url="https://example.test/document.pdf",
)


def _candidate(
    identifier: str,
    *,
    answerable: bool = True,
    page: int = 3,
    document_id: str = "test-document",
) -> Candidate:
    return Candidate(
        candidate_id=identifier,
        question=f"Question {identifier}?",
        proposed_reference_answer="Reference answer.",
        required_key_facts=("fact",) if answerable else (),
        document_id=document_id if answerable else None,
        physical_pages=(page,) if answerable else (),
        exact_source_span="Exact source." if answerable else None,
        answerable=answerable,
    )


def _rewrite_cell(path: Path, candidate_id: str, field: str, value: str) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    for row in rows:
        if row["candidate_id"] == candidate_id:
            row[field] = value
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_review_is_created_all_pending_and_round_trips_candidates(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("candidate-001"),
        _candidate("candidate-002", answerable=False),
    ]
    path = tmp_path / "review.csv"

    write_pending_review(candidates, path)
    records = load_review(path, candidates)

    assert [record.candidate.candidate_id for record in records] == [
        "candidate-001",
        "candidate-002",
    ]
    assert {record.review_status for record in records} == {"pending"}
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_pending_needs_changes_and_rejected_never_enter_gold(tmp_path: Path) -> None:
    approved = ReviewRecord(_candidate("candidate-001"), "dev", "approved", "ok")
    pending = ReviewRecord(_candidate("candidate-002"), "test", "pending", "")
    needs_changes = ReviewRecord(
        _candidate("candidate-003"), "test", "needs_changes", "revise"
    )
    rejected = ReviewRecord(
        _candidate("candidate-004"), "test", "rejected", "unsupported"
    )

    payloads = build_gold_records([approved, pending, needs_changes, rejected])

    assert [payload["candidate_id"] for payload in payloads] == ["candidate-001"]
    assert [candidate.candidate_id for candidate in approved_candidates(
        [approved, pending, needs_changes, rejected]
    )] == ["candidate-001"]
    assert approved_split_counts(
        [approved, pending, needs_changes, rejected]
    ) == {
        "dev": {"total": 1, "answerable": 1, "unanswerable": 0},
        "test": {"total": 0, "answerable": 0, "unanswerable": 0},
    }

    empty_path = tmp_path / "empty-gold.jsonl"
    _, count = write_gold([pending, needs_changes, rejected], empty_path)
    assert count == 0
    assert empty_path.read_text(encoding="utf-8") == ""


def test_invalid_review_status_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate("candidate-001")
    path = tmp_path / "review.csv"
    write_pending_review([candidate], path)
    _rewrite_cell(path, candidate.candidate_id, "review_status", "auto_approved")

    with pytest.raises(ReviewError, match="invalid review_status"):
        load_review(path, [candidate])


def test_changed_candidate_id_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate("candidate-001")
    path = tmp_path / "review.csv"
    write_pending_review([candidate], path)
    _rewrite_cell(path, candidate.candidate_id, "candidate_id", "candidate-999")

    with pytest.raises(ReviewError, match="unknown candidate_id"):
        load_review(path, [candidate])


def test_bad_page_or_span_fails_grounding_check(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "test-document.pages.jsonl").write_text(
        '{"document_id":"test-document","page_number":3,'
        '"text":"Heading\\nExact source.\\nFooter"}\n',
        encoding="utf-8",
    )
    wrong_page = ReviewRecord(
        _candidate("candidate-001", page=4), "dev", "pending", ""
    )

    with pytest.raises(ValueError, match="missing physical page"):
        validate_review_grounding(
            [wrong_page],
            [SOURCE],
            extracted,
            expected_total=1,
            expected_answerable=1,
            expected_unanswerable=0,
        )

    wrong_span_candidate = replace(
        _candidate("candidate-001"), exact_source_span="Fabricated span."
    )
    wrong_span = ReviewRecord(wrong_span_candidate, "dev", "pending", "")
    with pytest.raises(ValueError, match="does not exist verbatim"):
        validate_review_grounding(
            [wrong_span],
            [SOURCE],
            extracted,
            expected_total=1,
            expected_answerable=1,
            expected_unanswerable=0,
        )


def test_review_file_is_never_silently_overwritten(tmp_path: Path) -> None:
    candidate = _candidate("candidate-001")
    path = tmp_path / "review.csv"
    write_pending_review([candidate], path)

    with pytest.raises(ReviewError, match="not overwritten"):
        write_pending_review([candidate], path)


def test_silver_contains_every_candidate_unchanged_and_requires_overwrite(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("candidate-001"),
        _candidate("candidate-041", answerable=False),
    ]
    path = tmp_path / "silver.jsonl"

    payloads = build_silver_records(candidates)
    output, count = write_silver(candidates, path)

    assert count == 2
    assert output == path
    assert [payload["candidate_id"] for payload in payloads] == [
        candidate.candidate_id for candidate in candidates
    ]
    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert written == list(payloads)
    with pytest.raises(ReviewError, match="use --overwrite"):
        write_silver(candidates, path)


def test_silver_audit_is_deterministic_document_stratified_and_keeps_unanswerable(
) -> None:
    candidates = [
        _candidate("candidate-001", document_id="doc-a"),
        _candidate("candidate-002", document_id="doc-a"),
        _candidate("candidate-003", document_id="doc-a"),
        _candidate("candidate-004", document_id="doc-b"),
        _candidate("candidate-005", document_id="doc-b"),
        _candidate("candidate-006", document_id="doc-c"),
        _candidate("candidate-041", answerable=False),
        _candidate("candidate-042", answerable=False),
    ]

    selected = select_silver_audit_candidates(candidates, answerable_count=4)

    assert [candidate.candidate_id for candidate in selected] == [
        "candidate-001",
        "candidate-002",
        "candidate-004",
        "candidate-006",
        "candidate-041",
        "candidate-042",
    ]
    assert select_silver_audit_candidates(candidates, answerable_count=4) == selected
    assert {candidate.document_id for candidate in selected if candidate.answerable} == {
        "doc-a",
        "doc-b",
        "doc-c",
    }
