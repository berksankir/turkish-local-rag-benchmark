import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

from turkish_local_rag.amend_provenance import _protected_digest
from turkish_local_rag.provenance import (
    SILVER_DESCRIPTION_EN,
    SILVER_DESCRIPTION_TR,
)


ARCHIVE = Path("evaluation/provisional/2026-09-01-ai-candidates")
CANONICAL_SILVER = Path("evaluation/results/silver")
ORIGINAL_CSV_SHA256 = (
    "b66889c2d20085142bdd3b2e2562ff2c75538af3b531e1806be0152c2b51c041"
)
SUMMARY_SHA256 = (
    "ad76e326f07a62114b4a5d4ac2a8c9c8a2fbe169d298195586dacf402c9ab04e"
)
MARKDOWN_METRIC_ROWS_SHA256 = (
    "12170f25e2e8188f263ad3c2ba8d8c16f0f3434ffb2253109fb7050b807cc0c6"
)
CURRENT_PROTECTED_CONTENT_SHA256 = {
    "retrieval_benchmark": "0d4fa7e84ded8f169f3274163356efe8d3a94b6d7807567b554264793bcf9eaa",
    "reranker_profile": "5792fc52ce39dff267945fe2428cc0b6c68926e8f0f0a0f66d26a221fb334488",
    "evidence_gate_tuning": "b49d59ec7fb6e1d9e0fafe995ee751a6f4ec070b9ef1c59c5e4fa0df0ebb3491",
    "generation_benchmark": "cae090e344c36a0ad558839f2b24c15f7319f7d28e0dd2e1b0ac19af3e99adb5",
}


def test_provisional_json_is_self_describing_and_metrics_are_unchanged() -> None:
    payload = json.loads(
        (ARCHIVE / "retrieval_benchmark.original.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == 2
    assert payload["dataset"] == {
        "kind": "provisional_ai_candidates",
        "human_reviewed": False,
        "final_gold": False,
        "description": (
            "AI-generated candidate set with automatic answerable span/page "
            "integrity checks; automatic validation is not human approval."
        ),
        "correction_note": (
            "The original run incorrectly labeled this dataset as approved gold. "
            "Metric values are preserved unchanged."
        ),
    }
    reproducibility = payload["reproducibility"]
    assert "approved_candidates_sha256" not in reproducibility
    assert "gold_sha256" not in reproducibility
    assert reproducibility["candidate_set_sha256"]
    assert reproducibility["evaluation_set_sha256"]
    normalized_summary = json.dumps(
        payload["summary"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(normalized_summary).hexdigest() == SUMMARY_SHA256


def test_provisional_csv_labels_every_row_without_changing_original_fields() -> None:
    path = ARCHIVE / "retrieval_benchmark.original.csv"
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        rows = list(reader)

    assert len(rows) == 201
    assert rows[0][:2] == ["dataset_kind", "human_reviewed"]
    assert all(row[:2] == ["provisional_ai_candidates", "false"] for row in rows[1:])

    reconstructed = io.StringIO(newline="")
    writer = csv.writer(reconstructed, lineterminator="\n")
    writer.writerows(row[2:] for row in rows)
    assert (
        hashlib.sha256(reconstructed.getvalue().encode("utf-8")).hexdigest()
        == ORIGINAL_CSV_SHA256
    )


def test_provisional_candidate_artifact_is_not_named_gold() -> None:
    corrected = ARCHIVE / "provisional_candidates.original.jsonl"

    assert corrected.is_file()
    assert not (ARCHIVE / "provisional_gold.original.jsonl").exists()
    assert hashlib.sha256(corrected.read_bytes()).hexdigest() == (
        "0599ae0c761db73e7512f4cd4eef0a2ced2d258319f127076a0949096c9aa268"
    )


def test_provisional_markdown_has_correct_labels_and_unchanged_metrics() -> None:
    text = (ARCHIVE / "retrieval_benchmark.original.md").read_text(encoding="utf-8")
    metric_rows = [
        line
        for line in text.splitlines()
        if line.startswith(
            ("| dense |", "| bm25 |", "| hybrid_rrf |", "| hybrid_reranked |")
        )
    ]

    assert text.startswith("# Provisional retrieval run — AI-generated candidates")
    assert "Bu rapor insan onaylı gold set" not in text
    assert "- Gold SHA-256:" not in text
    assert len(metric_rows) == 12
    metric_bytes = ("\n".join(metric_rows) + "\n").encode("utf-8")
    assert hashlib.sha256(metric_bytes).hexdigest() == MARKDOWN_METRIC_ROWS_SHA256


def test_full_review_remains_pending_without_human_provenance() -> None:
    with Path("evaluation/review.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        rows = list(csv.DictReader(source))

    assert len(rows) == 50
    assert all(row["review_status"] == "pending" for row in rows)
    assert all(row["reviewer"] == "" for row in rows)
    assert all(row["reviewed_at_utc"] == "" for row in rows)


def test_silver_audit_preserves_explicit_human_approvals() -> None:
    with Path("evaluation/silver_audit.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        rows = list(csv.DictReader(source))

    answerable = [row for row in rows if row["answerable"] == "true"]
    unanswerable = [row for row in rows if row["answerable"] == "false"]
    assert len(answerable) == len(unanswerable) == 10
    assert all(row["review_status"] == "approved" for row in rows)
    assert all(row["review_notes"] == "" for row in rows)
    assert all(row["reviewer"] == "berksankir" for row in rows)
    timestamps = [
        datetime.fromisoformat(row["reviewed_at_utc"].replace("Z", "+00:00"))
        for row in rows
    ]
    assert all(
        10 <= (later - earlier).total_seconds() <= 15
        for earlier, later in zip(timestamps, timestamps[1:])
    )


def test_canonical_silver_report_has_required_dataset_and_audit_metadata() -> None:
    payload = json.loads(
        (CANONICAL_SILVER / "retrieval_benchmark.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == 3
    assert payload["dataset"]["kind"] == "silver"
    assert payload["dataset"]["human_reviewed"] is False
    assert payload["dataset"]["all_records_human_reviewed"] is False
    assert payload["dataset"]["dataset_release_approved"] is True
    assert payload["dataset"]["audit_sample"] == {
        "reviewed_records": 20,
        "total_records": 50,
        "approved_records": 20,
    }
    assert payload["dataset"]["audit_statuses"] == {
        "approved": 20,
        "needs_changes": 0,
        "pending": 0,
        "rejected": 0,
    }
    assert payload["dataset"]["audit_provenance"] == {
        "decisions_with_provenance": 20,
        "reviewers": ["berksankir"],
        "latest_reviewed_at_utc": "2026-09-02T08:37:55Z",
    }
    assert payload["reproducibility"]["dataset_kind"] == "silver"
    assert payload["reproducibility"]["chunk_count"] == 436
    assert payload["reproducibility"]["review_artifact_sha256"] == hashlib.sha256(
        Path("evaluation/silver_audit.csv").read_bytes()
    ).hexdigest()


def test_canonical_silver_results_cover_both_splits_without_claiming_gold() -> None:
    payload = json.loads(
        (CANONICAL_SILVER / "retrieval_benchmark.json").read_text(encoding="utf-8")
    )
    rows = payload["queries"]

    assert len(rows) == 200
    assert {row["dataset_kind"] for row in rows} == {"silver"}
    assert sum(row["split"] == "dev" for row in rows) == 40
    assert sum(row["split"] == "test" for row in rows) == 160
    markdown = (CANONICAL_SILVER / "retrieval_benchmark.md").read_text(
        encoding="utf-8"
    )
    assert markdown.startswith("# Silver retrieval benchmark")
    assert "Human audit durumu: approved=20" in markdown
    assert "insan onaylı gold" not in markdown


def test_reranker_profile_is_dev_only_and_reuses_one_model_instance() -> None:
    payload = json.loads(
        (CANONICAL_SILVER / "reranker_profile.json").read_text(encoding="utf-8")
    )

    assert payload["dataset"]["kind"] == "silver"
    assert payload["schema_version"] == 2
    assert payload["dataset"]["human_reviewed"] is False
    assert payload["protocol"]["selection_split"] == "dev"
    assert payload["protocol"]["test_split_accessed_for_tuning"] is False
    assert payload["protocol"]["reranker_instance_reused"] is True
    assert payload["protocol"]["reranker_instances_created"] == 1
    assert payload["protocol"]["default_fast_pipeline"] == "hybrid_rrf"
    assert payload["runtime"]["chunk_count"] == 436
    assert len(payload["variants"]) == 4
    assert payload["selection"]["test_split_used"] is False


def test_all_current_silver_artifacts_share_release_provenance() -> None:
    report_names = (
        "retrieval_benchmark",
        "reranker_profile",
        "evidence_gate_tuning",
        "generation_benchmark",
    )
    for name in report_names:
        payload = json.loads(
            (CANONICAL_SILVER / f"{name}.json").read_text(encoding="utf-8")
        )
        assert _protected_digest(payload) == CURRENT_PROTECTED_CONTENT_SHA256[name]
        dataset = payload["dataset"]
        assert dataset["kind"] == "silver"
        assert dataset["creation_method"] == "ai_assisted"
        assert dataset["final_gold"] is False
        assert dataset["dataset_release_approved"] is True
        assert dataset["approved_by"] == "berksankir"
        assert dataset["approval_scope"] == "dataset_level_with_sample_audit"
        assert dataset["all_records_human_reviewed"] is False
        assert dataset["human_reviewed"] is False
        assert "dataset-level release approval is absent" in dataset[
            "human_reviewed_semantics"
        ]
        assert dataset["audit_sample"] == {
            "reviewed_records": 20,
            "total_records": 50,
            "approved_records": 20,
        }

        with (CANONICAL_SILVER / f"{name}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            csv_row = next(csv.DictReader(source))
        assert csv_row["dataset_kind"] == "silver"
        assert csv_row["creation_method"] == "ai_assisted"
        assert csv_row["dataset_release_approved"] == "true"
        assert csv_row["approved_by"] == "berksankir"
        assert csv_row["approval_scope"] == "dataset_level_with_sample_audit"
        assert csv_row["all_records_human_reviewed"] == "false"
        assert csv_row["audit_reviewed_records"] == "20"
        assert csv_row["audit_total_records"] == "50"

        markdown = (CANONICAL_SILVER / f"{name}.md").read_text(encoding="utf-8")
        assert SILVER_DESCRIPTION_EN in markdown
        assert SILVER_DESCRIPTION_TR in markdown
        assert "dataset_release_approved=true" in markdown
        assert "final_gold=false" in markdown


def test_remaining_thirty_records_have_no_fabricated_item_review() -> None:
    with Path("evaluation/silver_audit.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        audit_rows = list(csv.DictReader(source))
    with Path("evaluation/review.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        full_review_rows = list(csv.DictReader(source))
    audit_ids = {row["candidate_id"] for row in audit_rows}
    unaudited = [row for row in full_review_rows if row["candidate_id"] not in audit_ids]

    assert len(audit_ids) == 20
    assert len(unaudited) == 30
    assert all(row["review_status"] == "pending" for row in unaudited)
    assert all(row["reviewer"] == "" for row in unaudited)
    assert all(row["reviewed_at_utc"] == "" for row in unaudited)


def test_readme_matches_machine_readable_release_provenance() -> None:
    english = Path("README.md").read_text(encoding="utf-8")
    turkish = Path("README.tr.md").read_text(encoding="utf-8")
    normalized_english = " ".join(english.split())
    normalized_turkish = " ".join(turkish.split())

    assert SILVER_DESCRIPTION_EN in normalized_english
    assert SILVER_DESCRIPTION_TR in normalized_turkish
    for readme in (english, turkish):
        assert "dataset_release_approved=true" in readme
        assert "all_records_human_reviewed=false" in readme
