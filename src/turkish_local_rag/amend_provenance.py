"""Amend current silver reports with release provenance without rerunning metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from turkish_local_rag.candidates import load_candidates
from turkish_local_rag.config import load_config
from turkish_local_rag.evaluate import (
    _render_csv as render_retrieval_csv,
    _render_markdown as render_retrieval_markdown,
    _write_text_atomic,
)
from turkish_local_rag.evaluate_generation import (
    _render_csv as render_generation_csv,
    _render_markdown as render_generation_markdown,
)
from turkish_local_rag.profile_reranker import (
    render_profile_csv,
    render_profile_markdown,
)
from turkish_local_rag.provenance import build_silver_provenance
from turkish_local_rag.review import (
    load_review,
    review_provenance_summary,
    review_status_counts,
    select_silver_audit_candidates,
)
from turkish_local_rag.tune_evidence import (
    _render_csv as render_evidence_csv,
    _render_markdown as render_evidence_markdown,
)


REPORTS: Mapping[
    str,
    tuple[
        int,
        tuple[str, ...],
        Callable[[Mapping[str, Any]], str],
        Callable[[Mapping[str, Any]], str],
    ],
] = {
    "retrieval_benchmark": (
        3,
        (),
        lambda payload: render_retrieval_csv(payload["queries"], payload["dataset"]),
        render_retrieval_markdown,
    ),
    "reranker_profile": (
        2,
        ("scope",),
        render_profile_csv,
        render_profile_markdown,
    ),
    "evidence_gate_tuning": (
        2,
        (
            "split",
            "records",
            "answerable",
            "unanswerable",
            "test_split_accessed_for_tuning",
        ),
        render_evidence_csv,
        render_evidence_markdown,
    ),
    "generation_benchmark": (
        2,
        ("records",),
        lambda payload: render_generation_csv(
            payload["queries"], payload["dataset"]
        ),
        render_generation_markdown,
    ),
}


def amend_reports(config_path: str | Path) -> dict[str, Any]:
    """Rewrite provenance/rendering only and prove protected metrics are unchanged."""

    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    candidates = load_candidates(paths.evaluation_candidates)
    audit = load_review(
        paths.evaluation_silver_audit,
        select_silver_audit_candidates(candidates),
    )
    canonical = build_silver_provenance(
        review_status_counts(audit),
        review_provenance_summary(audit),
        total_records=len(candidates),
    )
    directory = paths.evaluation_results_directory / "silver"
    amended: list[dict[str, Any]] = []
    for name, (schema_version, extra_keys, csv_renderer, markdown_renderer) in REPORTS.items():
        json_path = directory / f"{name}.json"
        csv_path = directory / f"{name}.csv"
        markdown_path = directory / f"{name}.md"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        protected_before = _protected_digest(payload)
        extras = {
            key: payload["dataset"][key]
            for key in extra_keys
            if key in payload["dataset"]
        }
        payload["schema_version"] = schema_version
        payload["dataset"] = {**canonical, **extras}
        protected_after = _protected_digest(payload)
        if protected_before != protected_after:
            raise RuntimeError(f"metric/content drift while amending {name}")
        _write_text_atomic(
            json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        _write_text_atomic(csv_path, csv_renderer(payload))
        _write_text_atomic(markdown_path, markdown_renderer(payload))
        amended.append(
            {
                "report": name,
                "schema_version": schema_version,
                "protected_content_sha256": protected_after,
            }
        )
    return {"reports": amended, "metrics_recomputed": False}


def _protected_digest(payload: Mapping[str, Any]) -> str:
    protected = {
        key: value
        for key, value in payload.items()
        if key not in {"schema_version", "dataset"}
    }
    encoded = json.dumps(
        protected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args(argv)
    print(json.dumps(amend_reports(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
