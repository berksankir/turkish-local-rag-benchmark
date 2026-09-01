from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from turkish_local_rag.candidates import Candidate
from turkish_local_rag.evaluate import (
    EvaluationError,
    EvaluationRecord,
    aggregate_results,
    load_gold,
    load_silver,
    _render_markdown,
    _split_policy,
    score_retrieval,
)
from turkish_local_rag import evaluate
from turkish_local_rag.retrieve import ChunkRecord, RetrievalHit


def _candidate(identifier: str, *, answerable: bool) -> Candidate:
    return Candidate(
        candidate_id=identifier,
        question=f"Question {identifier}?",
        proposed_reference_answer="Reference answer.",
        required_key_facts=("fact",) if answerable else (),
        document_id="doc" if answerable else None,
        physical_pages=(3,) if answerable else (),
        exact_source_span="Exact source." if answerable else None,
        answerable=answerable,
    )


def _gold_payload(candidate: Candidate, split: str) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "question": candidate.question,
        "proposed_reference_answer": candidate.proposed_reference_answer,
        "required_key_facts": list(candidate.required_key_facts),
        "document_id": candidate.document_id,
        "physical_pages": list(candidate.physical_pages),
        "exact_source_span": candidate.exact_source_span,
        "answerable": candidate.answerable,
        "split": split,
    }


def _chunk(document_id: str, page: int, index: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{document_id}:p{page}:c{index}",
        document_id=document_id,
        title="Document",
        page_number=page,
        source_page_url="https://example.test/source",
        pdf_url="https://example.test/doc.pdf",
        pdf_sha256="a" * 64,
        source_block_ids=(f"{document_id}:p{page}:b0",),
        text="text",
        estimated_tokens=2,
        token_count_method="test",
    )


def _hit(document_id: str, page: int, rank: int) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=1.0 / rank,
        retriever="test",
        chunk=_chunk(document_id, page, rank),
    )


def test_gold_loader_rejects_post_review_changes(tmp_path: Path) -> None:
    dev = _candidate("candidate-001", answerable=True)
    test = _candidate("candidate-002", answerable=False)
    changed = replace(dev, proposed_reference_answer="Changed answer.")
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_gold_payload(changed, "dev")),
                json.dumps(_gold_payload(test, "test")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="changed after review"):
        load_gold(
            path,
            [dev, test],
            expected_split_counts={
                "dev": {"total": 1, "answerable": 1, "unanswerable": 0},
                "test": {"total": 1, "answerable": 0, "unanswerable": 1},
            },
        )


def test_silver_loader_rejects_candidate_drift(tmp_path: Path) -> None:
    candidate = _candidate("candidate-001", answerable=True)
    changed = replace(candidate, proposed_reference_answer="Changed answer.")
    path = tmp_path / "silver.jsonl"
    path.write_text(
        json.dumps(_gold_payload(changed, "dev")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="differs from its source candidate"):
        load_silver(
            path,
            [candidate],
            expected_split_counts={
                "dev": {"total": 1, "answerable": 1, "unanswerable": 0},
                "test": {"total": 0, "answerable": 0, "unanswerable": 0},
            },
        )


def test_retrieval_scoring_requires_matching_document_and_page() -> None:
    record = EvaluationRecord(_candidate("candidate-001", answerable=True), "dev")
    hits = [_hit("doc", 2, 1), _hit("other", 3, 2), _hit("doc", 3, 3)]

    result = score_retrieval(record, "dense", hits, 12.5)

    assert result["metrics"] == {
        "recall_at_1": False,
        "recall_at_3": True,
        "recall_at_5": True,
        "reciprocal_rank": pytest.approx(1 / 3),
        "first_relevant_rank": 3,
        "correct_document_retrieval": True,
        "correct_page_retrieval": False,
    }
    assert result["hits"][2]["pdf_sha256"] == "a" * 64


def test_unanswerable_is_excluded_from_quality_metrics() -> None:
    record = EvaluationRecord(_candidate("candidate-002", answerable=False), "test")

    result = score_retrieval(record, "bm25", [_hit("doc", 3, 1)], 1.0)

    assert result["metrics"] is None


def test_aggregate_reports_each_pipeline_and_split() -> None:
    records: list[dict[str, object]] = []
    for split in ("dev", "test"):
        for pipeline in ("dense", "bm25", "hybrid_rrf", "hybrid_reranked"):
            record = EvaluationRecord(_candidate("candidate-001", answerable=True), split)
            records.append(
                score_retrieval(record, pipeline, [_hit("doc", 3, 1)], 2.0)
            )

    summary = aggregate_results(records)

    assert summary["test"]["dense"]["recall_at_5"] == 1.0
    assert summary["dev"]["hybrid_reranked"]["mrr"] == 1.0
    assert summary["all"]["bm25"]["average_retrieval_latency_ms"] == 2.0


def test_silver_report_is_explicitly_not_human_approved() -> None:
    summary = {
        split: {
            pipeline: {
                "recall_at_1": 1.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "correct_document_retrieval": 1.0,
                "correct_page_retrieval": 1.0,
                "average_retrieval_latency_ms": 1.0,
            }
            for pipeline in ("dense", "bm25", "hybrid_rrf", "hybrid_reranked")
        }
        for split in ("dev", "test", "all")
    }
    payload = {
        "dataset": {
            "kind": "silver",
            "audit_statuses": {
                "approved": 0,
                "needs_changes": 0,
                "pending": 20,
                "rejected": 0,
            },
        },
        "summary": summary,
        "runtime": {"benchmark_seconds": 2.0, "peak_process_rss_bytes": 100},
        "reproducibility": {
            "finished_at_utc": "2026-09-02T00:00:00Z",
            "dataset_kind": "silver",
            "evaluation_set_sha256": "a" * 64,
            "review_artifact_sha256": "b" * 64,
            "logical_corpus_sha256": "c" * 64,
            "embedding_model_revision": "revision-a",
            "reranker_model_revision": "revision-b",
        },
    }

    rendered = _render_markdown(payload)

    assert rendered.startswith("# Silver retrieval benchmark")
    assert "insan onayı değildir" in rendered
    assert "pending=20" in rendered
    assert "insan onaylı gold" not in rendered


def test_split_policy_is_derived_from_selected_records() -> None:
    records = [
        EvaluationRecord(_candidate("candidate-001", answerable=True), "dev"),
        EvaluationRecord(_candidate("candidate-002", answerable=False), "test"),
    ]

    assert _split_policy(records) == (
        "fixed before retrieval: dev=1 answerable+0 unanswerable; "
        "test=0 answerable+1 unanswerable"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows working-set API")
def test_windows_process_memory_measurement_returns_bytes() -> None:
    current, peak = evaluate._process_memory_bytes()

    assert current is not None and current > 0
    assert peak is not None and peak >= current
