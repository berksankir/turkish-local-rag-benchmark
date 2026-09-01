import json
from pathlib import Path

import pytest

from turkish_local_rag.candidates import (
    CandidateValidationError,
    load_candidates,
    validate_candidate_set,
)
from turkish_local_rag.download import SourceDocument


SOURCE = SourceDocument(
    id="test-document",
    title="Test Document",
    source_page_url="https://example.test/source",
    pdf_url="https://example.test/document.pdf",
)


def _record(*, answerable: bool = True) -> dict[str, object]:
    return {
        "candidate_id": "candidate-001" if answerable else "candidate-002",
        "question": "What is the exact rule?" if answerable else "What is today's menu?",
        "proposed_reference_answer": (
            "The exact rule is stated." if answerable else "The corpus cannot answer this."
        ),
        "required_key_facts": ["exact rule"] if answerable else [],
        "document_id": "test-document" if answerable else None,
        "physical_pages": [3] if answerable else [],
        "exact_source_span": "The exact rule is stated." if answerable else None,
        "answerable": answerable,
    }


def _write_fixture(tmp_path: Path, records: list[dict[str, object]]) -> tuple[Path, Path]:
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "test-document.pages.jsonl").write_text(
        json.dumps(
            {
                "document_id": "test-document",
                "page_number": 3,
                "text": "Heading\nThe exact rule is stated.\nFooter",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return candidate_path, extracted


def test_candidate_set_verifies_exact_span_and_balance(tmp_path: Path) -> None:
    candidate_path, extracted = _write_fixture(
        tmp_path, [_record(), _record(answerable=False)]
    )

    report = validate_candidate_set(
        load_candidates(candidate_path),
        [SOURCE],
        extracted,
        expected_total=2,
        expected_answerable=1,
        expected_unanswerable=1,
    )

    assert report.total == 2
    assert report.verified_source_spans == 1
    assert report.documents_covered == ("test-document",)


def test_candidate_set_rejects_nonexistent_exact_span(tmp_path: Path) -> None:
    record = _record()
    record["exact_source_span"] = "A fabricated quote."
    candidate_path, extracted = _write_fixture(tmp_path, [record])

    with pytest.raises(CandidateValidationError, match="does not exist verbatim"):
        validate_candidate_set(
            load_candidates(candidate_path),
            [SOURCE],
            extracted,
            expected_total=1,
            expected_answerable=1,
            expected_unanswerable=0,
        )


def test_unanswerable_candidate_cannot_claim_grounding(tmp_path: Path) -> None:
    record = _record(answerable=False)
    record["document_id"] = "test-document"
    candidate_path, _ = _write_fixture(tmp_path, [record])

    with pytest.raises(CandidateValidationError, match="cannot claim source grounding"):
        load_candidates(candidate_path)


def test_candidate_count_gate_is_enforced(tmp_path: Path) -> None:
    candidate_path, extracted = _write_fixture(tmp_path, [_record()])

    with pytest.raises(CandidateValidationError, match="expected 50 candidates"):
        validate_candidate_set(load_candidates(candidate_path), [SOURCE], extracted)
