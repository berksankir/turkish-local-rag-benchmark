from __future__ import annotations

import pytest

from turkish_local_rag.provenance import build_silver_provenance


AUDIT_STATUSES = {
    "approved": 20,
    "needs_changes": 0,
    "pending": 0,
    "rejected": 0,
}
AUDIT_PROVENANCE = {
    "decisions_with_provenance": 20,
    "reviewers": ["berksankir"],
    "latest_reviewed_at_utc": "2026-09-02T08:37:55Z",
}


def test_silver_provenance_distinguishes_release_from_full_item_review() -> None:
    dataset = build_silver_provenance(AUDIT_STATUSES, AUDIT_PROVENANCE)

    assert dataset["kind"] == "silver"
    assert dataset["creation_method"] == "ai_assisted"
    assert dataset["final_gold"] is False
    assert dataset["dataset_release_approved"] is True
    assert dataset["approved_by"] == "berksankir"
    assert dataset["approval_scope"] == "dataset_level_with_sample_audit"
    assert dataset["all_records_human_reviewed"] is False
    assert dataset["human_reviewed"] is False
    assert dataset["audit_sample"] == {
        "reviewed_records": 20,
        "total_records": 50,
        "approved_records": 20,
    }


def test_silver_provenance_rejects_incomplete_or_unattributed_audit() -> None:
    with pytest.raises(ValueError, match="approved 20-record audit"):
        build_silver_provenance({**AUDIT_STATUSES, "approved": 19}, AUDIT_PROVENANCE)
    with pytest.raises(ValueError, match="missing berksankir"):
        build_silver_provenance(
            AUDIT_STATUSES,
            {**AUDIT_PROVENANCE, "reviewers": ["someone_else"]},
        )
