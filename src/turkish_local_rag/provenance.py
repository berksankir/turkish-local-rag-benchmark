"""Versioned provenance for the released AI-assisted silver evaluation set."""

from __future__ import annotations

from typing import Any, Mapping


SILVER_PROVENANCE_SCHEMA_VERSION = 1
SILVER_TOTAL_RECORDS = 50
SILVER_APPROVED_BY = "berksankir"
SILVER_APPROVAL_SCOPE = "dataset_level_with_sample_audit"

SILVER_DESCRIPTION_EN = (
    "The benchmark uses an AI-assisted silver evaluation set. Its release and use "
    "were approved by the project owner after automated grounding checks and a "
    "human audit of 20 out of 50 records. The complete dataset was not reviewed "
    "item by item and is not presented as a human-reviewed gold set."
)
SILVER_DESCRIPTION_TR = (
    "Benchmark, AI destekli bir silver evaluation seti kullanmaktadır. Veri setinin "
    "yayımlanmasına ve benchmarkta kullanılmasına, otomatik grounding kontrolleri "
    "ve 50 kaydın 20’si üzerinde yapılan insan audit’i sonrasında proje sahibi "
    "tarafından onay verilmiştir. Kayıtların tamamı tek tek insan incelemesinden "
    "geçmemiştir ve veri seti human-reviewed gold set olarak sunulmamaktadır."
)
HUMAN_REVIEWED_SEMANTICS = (
    "Whether every record in the dataset was reviewed individually by a human; "
    "false does not mean that dataset-level release approval is absent."
)


def build_silver_provenance(
    audit_statuses: Mapping[str, int],
    audit_provenance: Mapping[str, Any],
    *,
    total_records: int = SILVER_TOTAL_RECORDS,
) -> dict[str, Any]:
    """Build and validate the canonical dataset-level and item-level provenance."""

    approved = int(audit_statuses.get("approved", 0))
    pending = int(audit_statuses.get("pending", 0))
    needs_changes = int(audit_statuses.get("needs_changes", 0))
    rejected = int(audit_statuses.get("rejected", 0))
    audit_total = approved + pending + needs_changes + rejected
    reviewed = audit_total - pending
    if total_records != SILVER_TOTAL_RECORDS:
        raise ValueError(f"silver provenance requires {SILVER_TOTAL_RECORDS} records")
    if audit_total != 20 or reviewed != 20 or approved != 20:
        raise ValueError("silver provenance requires the approved 20-record audit")
    if audit_provenance.get("decisions_with_provenance") != 20:
        raise ValueError("all 20 audit decisions must retain item-level provenance")
    if SILVER_APPROVED_BY not in audit_provenance.get("reviewers", []):
        raise ValueError("silver audit reviewer provenance is missing berksankir")
    return {
        "provenance_schema_version": SILVER_PROVENANCE_SCHEMA_VERSION,
        "kind": "silver",
        "creation_method": "ai_assisted",
        "final_gold": False,
        "dataset_release_approved": True,
        "approved_by": SILVER_APPROVED_BY,
        "approval_scope": SILVER_APPROVAL_SCOPE,
        "all_records_human_reviewed": False,
        # Backward-compatible alias. Its meaning is explicitly limited below.
        "human_reviewed": False,
        "human_reviewed_semantics": HUMAN_REVIEWED_SEMANTICS,
        "automated_grounding_checks": {
            "answerable_exact_source_span_and_physical_page": True,
            "counts_as_human_item_review": False,
        },
        "audit_sample": {
            "reviewed_records": reviewed,
            "total_records": total_records,
            "approved_records": approved,
        },
        "audit_provenance": dict(audit_provenance),
        "audit_statuses": dict(audit_statuses),
        "description_en": SILVER_DESCRIPTION_EN,
        "description_tr": SILVER_DESCRIPTION_TR,
    }


def provenance_csv_fields() -> tuple[str, ...]:
    return (
        "provenance_schema_version",
        "dataset_kind",
        "creation_method",
        "final_gold",
        "dataset_release_approved",
        "approved_by",
        "approval_scope",
        "all_records_human_reviewed",
        "human_reviewed",
        "audit_reviewed_records",
        "audit_total_records",
        "audit_approved_records",
    )


def provenance_csv_values(dataset: Mapping[str, Any]) -> dict[str, Any]:
    audit = dataset["audit_sample"]
    return {
        "provenance_schema_version": dataset["provenance_schema_version"],
        "dataset_kind": dataset["kind"],
        "creation_method": dataset["creation_method"],
        "final_gold": str(dataset["final_gold"]).lower(),
        "dataset_release_approved": str(dataset["dataset_release_approved"]).lower(),
        "approved_by": dataset["approved_by"],
        "approval_scope": dataset["approval_scope"],
        "all_records_human_reviewed": str(
            dataset["all_records_human_reviewed"]
        ).lower(),
        "human_reviewed": str(dataset["human_reviewed"]).lower(),
        "audit_reviewed_records": audit["reviewed_records"],
        "audit_total_records": audit["total_records"],
        "audit_approved_records": audit["approved_records"],
    }


def provenance_markdown_lines() -> list[str]:
    return [
        "## Dataset provenance",
        "",
        SILVER_DESCRIPTION_EN,
        "",
        SILVER_DESCRIPTION_TR,
        "",
        "`human_reviewed=false` ve `all_records_human_reviewed=false`, yalnızca "
        "50 kaydın tamamının item-level insan incelemesinden geçmediğini belirtir; "
        "dataset-level yayımlama ve benchmark kullanım onayının bulunmadığı anlamına gelmez.",
        "",
        "Machine-readable scope: `creation_method=ai_assisted`, "
        "`dataset_release_approved=true`, `approved_by=berksankir`, "
        "`approval_scope=dataset_level_with_sample_audit`, audit `20/50`, "
        "`final_gold=false`.",
        "",
    ]
