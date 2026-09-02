from __future__ import annotations

import hashlib
import json
from pathlib import Path
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


def test_committed_generation_benchmark_is_self_describing_silver() -> None:
    report = json.loads(
        Path("evaluation/results/silver/generation_benchmark.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["schema_version"] == 3
    assert report["dataset"]["kind"] == "silver"
    assert report["dataset"]["creation_method"] == "ai_assisted"
    assert report["dataset"]["dataset_release_approved"] is True
    assert report["dataset"]["approved_by"] == "berksankir"
    assert report["dataset"]["approval_scope"] == "dataset_level_with_sample_audit"
    assert report["dataset"]["all_records_human_reviewed"] is False
    assert report["dataset"]["human_reviewed"] is False
    assert report["protocol"]["test_used_for_model_or_threshold_selection"] is False
    assert report["protocol"]["llm_as_a_judge"] is False
    assert report["protocol"]["generator_output_schema_version"] == "1.1"
    assert report["protocol"]["llama_cpp_response_format"]["schema_location"] == (
        "top-level json_schema"
    )
    assert report["runtime"]["generator_start_count"] == 1
    assert report["models"]["generator"]["model_size_bytes"] == 1_117_320_736
    assert len(report["models"]["generator"]["runtime_sha256"]) == 64
    assert len(report["queries"]) == 100
    assert _aggregate(report["queries"]) == report["summary"]


def test_phase8_baseline_is_preserved_byte_for_byte() -> None:
    baseline = Path(
        "evaluation/results/silver/phase8_baseline/generation_benchmark.json"
    ).read_bytes()

    assert hashlib.sha256(baseline).hexdigest() == (
        "2db9db2ffca6743c76ee678bf3739a263fd5b6770fa289f9bb06f791e07bf8a2"
    )


def test_committed_generation_citations_are_retrieved_and_abstentions_are_empty() -> None:
    report = json.loads(
        Path("evaluation/results/silver/generation_benchmark.json").read_text(
            encoding="utf-8"
        )
    )

    for row in report["queries"]:
        retrieved = {hit["chunk_id"] for hit in row["retrieved_chunks"]}
        if row["abstained"]:
            assert row["citations"] == []
            assert row["answer"] == "Yeterli kanıt bulunamadı."
        else:
            assert row["citations"]
            assert all(citation["chunk_id"] in retrieved for citation in row["citations"])
    diagnostics = [row["generation_error"] for row in report["queries"] if row["generation_error"]]
    assert len(diagnostics) == 10
    assert {item["category"] for item in diagnostics} == {
        "invalid_json_syntax",
        "invalid_context_id",
    }
    assert all(
        item["debug_excerpt"] is None or len(item["debug_excerpt"]) <= 240
        for item in diagnostics
    )


def test_generation_markdown_reports_separated_latency_and_limitations() -> None:
    report = Path("evaluation/results/silver/generation_benchmark.md").read_text(
        encoding="utf-8"
    )

    assert "`human_reviewed=false`" in report
    assert "`dataset_release_approved=true`" in report
    assert "Test split model, pipeline veya threshold seçimi için kullanılmamıştır" in report
    for stage in ("retrieval", "reranking", "generation", "total"):
        assert f"| hybrid_rrf | {stage} |" in report
        assert f"| hybrid_reranked | {stage} |" in report
