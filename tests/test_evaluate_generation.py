from __future__ import annotations

from types import SimpleNamespace

import pytest

from turkish_local_rag.candidates import Candidate
from turkish_local_rag.evaluate_generation import (
    _aggregate,
    _key_fact_coverage,
    _score,
    _token_f1,
)


def _candidate(answerable: bool = True) -> Candidate:
    return Candidate(
        candidate_id="candidate-001",
        question="En yüksek karar organı hangisidir?",
        proposed_reference_answer="Mütevelli Heyettir.",
        required_key_facts=("Mütevelli Heyet en yüksek karar organıdır.",)
        if answerable
        else (),
        document_id="doc" if answerable else None,
        physical_pages=(3,) if answerable else (),
        exact_source_span="Mütevelli Heyet, en yüksek karar organıdır"
        if answerable
        else None,
        answerable=answerable,
    )


def _response(answerable: bool = True) -> dict[str, object]:
    hit = {
        "rank": 1,
        "chunk_id": "doc:p3:c1",
        "document_id": "doc",
        "physical_page": 3,
        "text": "Mütevelli Heyet, en yüksek karar organıdır.",
    }
    return {
        "pipeline": "hybrid_rrf",
        "answer": "Mütevelli Heyettir." if answerable else "Yeterli kanıt bulunamadı.",
        "abstained": not answerable,
        "abstention_reason": None if answerable else "query_coverage_below_threshold",
        "citations": ([{
            "document_id": "doc",
            "physical_page": 3,
            "chunk_id": "doc:p3:c1",
        }] if answerable else []),
        "retrieved_chunks": [hit],
        "scores": {"evidence_score": 1.0},
        "latency_ms": {
            "retrieval": 2.0,
            "reranking": 0.0,
            "generation": 5.0 if answerable else 0.0,
            "total": 7.0 if answerable else 2.0,
        },
    }


def test_generation_scoring_requires_exact_trusted_citation_support() -> None:
    row = _score(SimpleNamespace(candidate=_candidate(), split="dev"), _response())

    assert row["citation_accuracy"] is True
    assert row["retrieval"]["correct_page"] is True
    assert row["token_f1"] == 1.0


def test_deterministic_text_metrics() -> None:
    assert _token_f1("Mütevelli Heyettir", "Mütevelli Heyettir") == 1.0
    assert _key_fact_coverage("Mütevelli Heyet karar organıdır", ["Mütevelli Heyet"]) == 1.0


def test_aggregate_separates_answerable_and_abstention_metrics() -> None:
    rows = []
    for split in ("dev", "test"):
        for pipeline in ("hybrid_rrf", "hybrid_reranked"):
            answerable_response = _response(True)
            answerable_response["pipeline"] = pipeline
            unanswerable_response = _response(False)
            unanswerable_response["pipeline"] = pipeline
            rows.append(_score(SimpleNamespace(candidate=_candidate(True), split=split), answerable_response))
            rows.append(_score(SimpleNamespace(candidate=_candidate(False), split=split), unanswerable_response))

    summary = _aggregate(rows)

    assert summary["all"]["hybrid_rrf"]["answerable_coverage"] == 1.0
    assert summary["test"]["hybrid_reranked"]["correct_abstention_rate"] == 1.0
    assert summary["dev"]["hybrid_rrf"]["false_abstention_rate"] == 0.0
