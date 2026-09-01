import csv
import hashlib
import io
import json
from pathlib import Path


ARCHIVE = Path("evaluation/provisional/2026-09-01-ai-candidates")
ORIGINAL_CSV_SHA256 = (
    "b66889c2d20085142bdd3b2e2562ff2c75538af3b531e1806be0152c2b51c041"
)
SUMMARY_SHA256 = (
    "ad76e326f07a62114b4a5d4ac2a8c9c8a2fbe169d298195586dacf402c9ab04e"
)
MARKDOWN_METRIC_ROWS_SHA256 = (
    "12170f25e2e8188f263ad3c2ba8d8c16f0f3434ffb2253109fb7050b807cc0c6"
)


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


def test_pending_review_artifacts_have_empty_human_provenance() -> None:
    expected_counts = {
        Path("evaluation/review.csv"): 50,
        Path("evaluation/silver_audit.csv"): 20,
    }
    for path, expected_count in expected_counts.items():
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == expected_count
        assert all(row["review_status"] == "pending" for row in rows)
        assert all(row["reviewer"] == "" for row in rows)
        assert all(row["reviewed_at_utc"] == "" for row in rows)
