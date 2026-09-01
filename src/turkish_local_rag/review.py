"""Prepare human review data and build gold only from explicit approvals."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

from turkish_local_rag.candidates import (
    Candidate,
    CandidateValidationReport,
    CandidateValidationError,
    load_candidates,
    parse_candidate_record,
    validate_candidate_set,
)
from turkish_local_rag.config import load_config
from turkish_local_rag.download import SourceDocument, load_manifest


VALID_REVIEW_STATUSES = frozenset(
    {"pending", "approved", "needs_changes", "rejected"}
)
REVIEW_FIELDS = (
    "candidate_id",
    "question",
    "proposed_reference_answer",
    "required_key_facts",
    "document_id",
    "physical_pages",
    "exact_source_span",
    "answerable",
    "proposed_split",
    "review_status",
    "review_notes",
    "reviewer",
    "reviewed_at_utc",
)

# This split proposal was fixed before the provisional retrieval run and is not
# derived from benchmark outcomes. It is retained only to prevent later test-set
# selection based on observed scores.
PROPOSED_DEV_IDS = frozenset(
    {
        "candidate-001",
        "candidate-006",
        "candidate-010",
        "candidate-016",
        "candidate-023",
        "candidate-026",
        "candidate-029",
        "candidate-035",
        "candidate-041",
        "candidate-042",
    }
)
SILVER_AUDIT_ANSWERABLE_COUNT = 10


class ReviewError(ValueError):
    """Raised when a review artifact is invalid or unsafe to use for gold."""


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    candidate: Candidate
    proposed_split: str
    review_status: str
    review_notes: str
    reviewer: str = ""
    reviewed_at_utc: str = ""


def write_pending_review(
    candidates: Sequence[Candidate], path: str | Path
) -> Path:
    """Write a new all-pending review CSV and never overwrite human edits."""

    review_path = Path(path)
    if review_path.exists():
        raise ReviewError(f"review file already exists; not overwritten: {review_path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=REVIEW_FIELDS, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
    )
    writer.writeheader()
    for candidate in candidates:
        writer.writerow(_candidate_to_review_row(candidate))
    _write_text_atomic(review_path, stream.getvalue(), encoding="utf-8-sig")
    return review_path


def load_review(
    path: str | Path, candidates: Sequence[Candidate]
) -> tuple[ReviewRecord, ...]:
    """Load review decisions while proving candidate fields and IDs are unchanged."""

    review_path = Path(path)
    try:
        review_file = review_path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise ReviewError(f"review file not found: {review_path}") from exc
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    records: list[ReviewRecord] = []
    seen_ids: set[str] = set()
    with review_file:
        reader = csv.DictReader(review_file)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ReviewError(
                "review CSV header mismatch; expected: " + ", ".join(REVIEW_FIELDS)
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ReviewError(f"unexpected extra CSV value at line {line_number}")
            status = row["review_status"]
            if status not in VALID_REVIEW_STATUSES:
                raise ReviewError(
                    f"invalid review_status at line {line_number}: {status!r}"
                )
            proposed_split = row["proposed_split"]
            if proposed_split not in {"dev", "test"}:
                raise ReviewError(
                    f"invalid proposed_split at line {line_number}: {proposed_split!r}"
                )
            candidate = _candidate_from_review_row(row, line_number)
            review_notes = row["review_notes"]
            reviewer = row["reviewer"].strip()
            reviewed_at_utc = row["reviewed_at_utc"].strip()
            _validate_review_decision(
                status,
                review_notes,
                reviewer,
                reviewed_at_utc,
                context=f"line {line_number}",
            )
            if candidate.candidate_id in seen_ids:
                raise ReviewError(
                    f"duplicate candidate_id in review: {candidate.candidate_id}"
                )
            seen_ids.add(candidate.candidate_id)
            expected = candidate_by_id.get(candidate.candidate_id)
            if expected is None:
                raise ReviewError(
                    f"review contains unknown candidate_id: {candidate.candidate_id}"
                )
            if candidate != expected:
                raise ReviewError(
                    f"review candidate fields changed: {candidate.candidate_id}"
                )
            expected_split = _proposed_split(candidate.candidate_id)
            if proposed_split != expected_split:
                raise ReviewError(
                    f"review proposed_split changed: {candidate.candidate_id}"
                )
            records.append(
                ReviewRecord(
                    candidate=candidate,
                    proposed_split=proposed_split,
                    review_status=status,
                    review_notes=review_notes,
                    reviewer=reviewer,
                    reviewed_at_utc=reviewed_at_utc,
                )
            )
    if seen_ids != set(candidate_by_id):
        missing = sorted(set(candidate_by_id) - seen_ids)
        extra = sorted(seen_ids - set(candidate_by_id))
        raise ReviewError(
            f"review/candidate ID mismatch: missing={missing}, extra={extra}"
        )
    return tuple(records)


def validate_review_grounding(
    records: Sequence[ReviewRecord],
    sources: Sequence[SourceDocument],
    extracted_pages_directory: str | Path,
    *,
    expected_total: int = 50,
    expected_answerable: int = 40,
    expected_unanswerable: int = 10,
) -> CandidateValidationReport:
    """Check answerable spans only; this does not confer human approval."""

    return validate_candidate_set(
        [record.candidate for record in records],
        sources,
        extracted_pages_directory,
        expected_total=expected_total,
        expected_answerable=expected_answerable,
        expected_unanswerable=expected_unanswerable,
    )


def approved_candidates(records: Sequence[ReviewRecord]) -> tuple[Candidate, ...]:
    """Return only explicitly approved records, preserving candidate IDs and order."""

    return tuple(
        record.candidate
        for record in records
        if record.review_status == "approved"
    )


def approved_split_counts(
    records: Sequence[ReviewRecord],
) -> dict[str, dict[str, int]]:
    """Return expected gold split counts derived only from approved rows."""

    counts = {
        "dev": {"total": 0, "answerable": 0, "unanswerable": 0},
        "test": {"total": 0, "answerable": 0, "unanswerable": 0},
    }
    for record in records:
        if record.review_status != "approved":
            continue
        split_counts = counts[record.proposed_split]
        split_counts["total"] += 1
        key = "answerable" if record.candidate.answerable else "unanswerable"
        split_counts[key] += 1
    return counts


def select_silver_audit_candidates(
    candidates: Sequence[Candidate],
    *,
    answerable_count: int = SILVER_AUDIT_ANSWERABLE_COUNT,
) -> tuple[Candidate, ...]:
    """Select all unanswerable and a deterministic document-stratified sample."""

    if answerable_count < 1:
        raise ReviewError("silver audit answerable_count must be positive")
    by_document: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.answerable:
            if candidate.document_id is None:
                raise ReviewError(
                    f"answerable candidate lacks document_id: {candidate.candidate_id}"
                )
            by_document.setdefault(candidate.document_id, []).append(candidate)
    total_answerable = sum(len(group) for group in by_document.values())
    if total_answerable < answerable_count:
        raise ReviewError(
            "silver audit answerable sample exceeds available candidates: "
            f"requested={answerable_count}, available={total_answerable}"
        )
    for group in by_document.values():
        group.sort(key=lambda candidate: candidate.candidate_id)

    selected_ids: set[str] = {
        candidate.candidate_id for candidate in candidates if not candidate.answerable
    }
    document_order = sorted(
        by_document, key=lambda document_id: (-len(by_document[document_id]), document_id)
    )
    selected_answerable = 0
    for document_id in document_order[:answerable_count]:
        selected_ids.add(by_document[document_id].pop(0).candidate_id)
        selected_answerable += 1
    while selected_answerable < answerable_count:
        remaining_documents = [
            document_id for document_id in document_order if by_document[document_id]
        ]
        document_id = min(
            remaining_documents,
            key=lambda item: (-len(by_document[item]), item),
        )
        selected_ids.add(by_document[document_id].pop(0).candidate_id)
        selected_answerable += 1

    return tuple(
        candidate for candidate in candidates if candidate.candidate_id in selected_ids
    )


def build_silver_records(
    candidates: Sequence[Candidate],
) -> tuple[dict[str, Any], ...]:
    """Build the synthetic silver set without implying human approval."""

    return tuple(
        _candidate_payload(candidate, _proposed_split(candidate.candidate_id))
        for candidate in candidates
    )


def write_silver(
    candidates: Sequence[Candidate],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Atomically write all validated candidates as silver, with explicit overwrite."""

    silver_path = Path(path)
    if silver_path.exists() and not overwrite:
        raise ReviewError(f"silver file already exists; use --overwrite: {silver_path}")
    payloads = build_silver_records(candidates)
    text = "".join(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        for payload in payloads
    )
    _write_text_atomic(silver_path, text, encoding="utf-8")
    return silver_path, len(payloads)


def build_gold_records(records: Sequence[ReviewRecord]) -> tuple[dict[str, Any], ...]:
    """Build gold payloads from approved rows; all other statuses are excluded."""

    payloads: list[dict[str, Any]] = []
    for record in records:
        if record.review_status != "approved":
            continue
        _validate_review_decision(
            record.review_status,
            record.review_notes,
            record.reviewer,
            record.reviewed_at_utc,
            context=record.candidate.candidate_id,
        )
        payloads.append(_candidate_payload(record.candidate, record.proposed_split))
    return tuple(payloads)


def write_gold(
    records: Sequence[ReviewRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Atomically write approved records and require explicit replacement."""

    gold_path = Path(path)
    if gold_path.exists() and not overwrite:
        raise ReviewError(f"gold file already exists; use --overwrite: {gold_path}")
    payloads = build_gold_records(records)
    text = "".join(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        for payload in payloads
    )
    _write_text_atomic(gold_path, text, encoding="utf-8")
    return gold_path, len(payloads)


def review_status_counts(records: Sequence[ReviewRecord]) -> dict[str, int]:
    counts = Counter(record.review_status for record in records)
    return {status: counts.get(status, 0) for status in sorted(VALID_REVIEW_STATUSES)}


def review_provenance_summary(records: Sequence[ReviewRecord]) -> dict[str, Any]:
    """Summarize explicit human provenance without inferring approval."""

    decided = [record for record in records if record.review_status != "pending"]
    timestamps = sorted(record.reviewed_at_utc for record in decided)
    return {
        "decisions_with_provenance": len(decided),
        "reviewers": sorted({record.reviewer for record in decided}),
        "latest_reviewed_at_utc": timestamps[-1] if timestamps else None,
    }


def _validate_review_decision(
    status: str,
    review_notes: str,
    reviewer: str,
    reviewed_at_utc: str,
    *,
    context: str,
) -> None:
    if status == "pending":
        if reviewer or reviewed_at_utc:
            raise ReviewError(
                f"pending review cannot carry reviewer provenance at {context}"
            )
        return
    if not reviewer:
        raise ReviewError(f"{status} review requires reviewer at {context}")
    if not reviewed_at_utc:
        raise ReviewError(f"{status} review requires reviewed_at_utc at {context}")
    if not reviewed_at_utc.endswith("Z"):
        raise ReviewError(
            f"reviewed_at_utc must be an ISO-8601 UTC timestamp ending in Z at {context}"
        )
    try:
        parsed = datetime.fromisoformat(reviewed_at_utc[:-1] + "+00:00")
    except ValueError as exc:
        raise ReviewError(
            f"invalid reviewed_at_utc timestamp at {context}: {reviewed_at_utc!r}"
        ) from exc
    if parsed.utcoffset() != timedelta(0):
        raise ReviewError(f"reviewed_at_utc must use UTC at {context}")
    if status in {"needs_changes", "rejected"} and not review_notes.strip():
        raise ReviewError(f"{status} review requires review_notes at {context}")


def _candidate_payload(candidate: Candidate, split: str) -> dict[str, Any]:
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


def _candidate_to_review_row(candidate: Candidate) -> dict[str, str]:
    return {
        "candidate_id": candidate.candidate_id,
        "question": candidate.question,
        "proposed_reference_answer": candidate.proposed_reference_answer,
        "required_key_facts": json.dumps(
            list(candidate.required_key_facts), ensure_ascii=False, separators=(",", ":")
        ),
        "document_id": candidate.document_id or "",
        "physical_pages": json.dumps(list(candidate.physical_pages), separators=(",", ":")),
        "exact_source_span": _escape_multiline(candidate.exact_source_span or ""),
        "answerable": "true" if candidate.answerable else "false",
        "proposed_split": _proposed_split(candidate.candidate_id),
        "review_status": "pending",
        "review_notes": "",
        "reviewer": "",
        "reviewed_at_utc": "",
    }


def _candidate_from_review_row(
    row: Mapping[str, str], line_number: int
) -> Candidate:
    try:
        required_key_facts = json.loads(row["required_key_facts"])
        physical_pages = json.loads(row["physical_pages"])
    except json.JSONDecodeError as exc:
        raise ReviewError(
            f"invalid JSON list in review at line {line_number}: {exc}"
        ) from exc
    if row["answerable"] not in {"true", "false"}:
        raise ReviewError(
            f"invalid answerable value at line {line_number}: {row['answerable']!r}"
        )
    raw = {
        "candidate_id": row["candidate_id"],
        "question": row["question"],
        "proposed_reference_answer": row["proposed_reference_answer"],
        "required_key_facts": required_key_facts,
        "document_id": row["document_id"] or None,
        "physical_pages": physical_pages,
        "exact_source_span": (
            _unescape_multiline(row["exact_source_span"])
            if row["exact_source_span"]
            else None
        ),
        "answerable": row["answerable"] == "true",
    }
    try:
        return parse_candidate_record(raw, line_number)
    except CandidateValidationError as exc:
        raise ReviewError(str(exc)) from exc


def _proposed_split(candidate_id: str) -> str:
    return "dev" if candidate_id in PROPOSED_DEV_IDS else "test"


def _escape_multiline(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def _unescape_multiline(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            result.append(value[index])
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] not in {"\\", "r", "n"}:
            raise ReviewError("exact_source_span contains an invalid escape sequence")
        escaped = value[index + 1]
        result.append({"\\": "\\", "r": "\r", "n": "\n"}[escaped])
        index += 2
    return "".join(result)


def _load_review_context(
    config_path: str | Path,
) -> tuple[Any, Any, Any, Any, CandidateValidationReport]:
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    candidates = load_candidates(paths.evaluation_candidates)
    sources = load_manifest(paths.source_manifest)
    grounding = validate_candidate_set(
        candidates, sources, paths.extracted_pages_directory
    )
    return paths, candidates, sources, config, grounding


def _write_text_atomic(path: Path, text: str, *, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="create a new all-pending review CSV")
    silver_parser = subparsers.add_parser(
        "build-silver",
        help="write all automatically validated candidates as synthetic silver JSONL",
    )
    silver_parser.add_argument("--overwrite", action="store_true")
    subparsers.add_parser(
        "prepare-silver-audit",
        help="create a pending audit CSV with all unanswerable and 10 answerable rows",
    )
    subparsers.add_parser(
        "validate-silver-audit",
        help="validate silver audit identity, statuses, and answerable grounding",
    )
    subparsers.add_parser("validate", help="validate review identity, statuses, and spans")
    build_parser = subparsers.add_parser(
        "build-gold", help="write only explicitly approved records to gold JSONL"
    )
    build_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths, candidates, sources, _config, grounding = _load_review_context(
            args.config
        )
        if args.command == "prepare":
            output = write_pending_review(candidates, paths.evaluation_review)
            print(
                json.dumps(
                    {
                        "path": str(output),
                        "records": len(candidates),
                        "pending": len(candidates),
                        "verified_source_spans": grounding.verified_source_spans,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "build-silver":
            output, count = write_silver(
                candidates, paths.evaluation_silver, overwrite=args.overwrite
            )
            print(
                json.dumps(
                    {
                        "dataset_kind": "silver",
                        "human_reviewed": False,
                        "path": str(output),
                        "records": count,
                        "verified_source_spans": grounding.verified_source_spans,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "prepare-silver-audit":
            audit_candidates = select_silver_audit_candidates(candidates)
            output = write_pending_review(
                audit_candidates, paths.evaluation_silver_audit
            )
            print(
                json.dumps(
                    {
                        "answerable": sum(
                            candidate.answerable for candidate in audit_candidates
                        ),
                        "path": str(output),
                        "pending": len(audit_candidates),
                        "records": len(audit_candidates),
                        "unanswerable": sum(
                            not candidate.answerable for candidate in audit_candidates
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate-silver-audit":
            audit_candidates = select_silver_audit_candidates(candidates)
            audit_reviews = load_review(
                paths.evaluation_silver_audit, audit_candidates
            )
            audit_grounding = validate_review_grounding(
                audit_reviews,
                sources,
                paths.extracted_pages_directory,
                expected_total=len(audit_candidates),
                expected_answerable=sum(
                    candidate.answerable for candidate in audit_candidates
                ),
                expected_unanswerable=sum(
                    not candidate.answerable for candidate in audit_candidates
                ),
            )
            print(
                json.dumps(
                    {
                        "records": len(audit_reviews),
                        "provenance": review_provenance_summary(audit_reviews),
                        "statuses": review_status_counts(audit_reviews),
                        "verified_source_spans": (
                            audit_grounding.verified_source_spans
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        reviews = load_review(paths.evaluation_review, candidates)
        grounding = validate_review_grounding(
            reviews, sources, paths.extracted_pages_directory
        )
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "records": len(reviews),
                        "provenance": review_provenance_summary(reviews),
                        "statuses": review_status_counts(reviews),
                        "verified_source_spans": grounding.verified_source_spans,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        output, count = write_gold(
            reviews, paths.evaluation_gold, overwrite=args.overwrite
        )
        print(
            json.dumps(
                {
                    "path": str(output),
                    "gold_records": count,
                    "statuses": review_status_counts(reviews),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (CandidateValidationError, ReviewError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
