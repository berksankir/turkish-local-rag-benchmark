"""Compare a small, fixed evidence-gate grid on the silver dev split only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import tempfile
import os
from typing import Any, Mapping, Sequence

from turkish_local_rag.candidates import load_candidates
from turkish_local_rag.config import load_config
from turkish_local_rag.evaluate import load_silver
from turkish_local_rag.generation import evaluate_evidence, select_context_hits
from turkish_local_rag.query import LocalRetrievalRuntime
from turkish_local_rag.retrieve import normalize_turkish


VARIANTS = (
    ("coverage_020_score_020", 0.20, 0.020),
    ("coverage_030_score_020", 0.30, 0.020),
    ("coverage_040_score_020", 0.40, 0.020),
    ("coverage_030_score_024", 0.30, 0.024),
)


def run_tuning(config_path: str | Path, *, overwrite: bool = False) -> dict[str, Path]:
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    candidates = load_candidates(paths.evaluation_candidates)
    dev = [
        record
        for record in load_silver(paths.evaluation_silver, candidates)
        if record.split == "dev"
    ]
    outputs = {
        extension: paths.evaluation_results_directory
        / "silver"
        / f"evidence_gate_tuning.{extension}"
        for extension in ("json", "csv", "md")
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise RuntimeError("evidence tuning output exists; use --overwrite explicitly")

    runtime = LocalRetrievalRuntime(config_path)
    try:
        observations: list[dict[str, Any]] = []
        for record in dev:
            execution = runtime(record.candidate.question, "hybrid_rrf")
            contexts = select_context_hits(
                record.candidate.question,
                execution.hits,
                config.evidence.context_top_k,
            )
            base = evaluate_evidence(
                record.candidate.question, contexts, config.evidence
            )
            span = record.candidate.exact_source_span
            normalized_context = " ".join(
                normalize_turkish("\n".join(hit.chunk.text for hit in contexts)).split()
            )
            observations.append(
                {
                    "candidate_id": record.candidate.candidate_id,
                    "answerable": record.candidate.answerable,
                    "question": record.candidate.question,
                    "query_coverage": base.query_coverage,
                    "top_retrieval_score": base.top_retrieval_score,
                    "context_chunk_ids": [hit.chunk.chunk_id for hit in contexts],
                    "exact_source_span_in_context": (
                        " ".join(normalize_turkish(span).split()) in normalized_context
                        if span
                        else None
                    ),
                }
            )
    finally:
        runtime.close()

    variants = []
    for name, coverage, score in VARIANTS:
        answerable = [row for row in observations if row["answerable"]]
        unanswerable = [row for row in observations if not row["answerable"]]
        passes = [
            row["query_coverage"] >= coverage
            and row["top_retrieval_score"] >= score
            for row in observations
        ]
        answerable_coverage = sum(
            passed for row, passed in zip(observations, passes, strict=True) if row["answerable"]
        ) / len(answerable)
        correct_abstention = sum(
            not passed
            for row, passed in zip(observations, passes, strict=True)
            if not row["answerable"]
        ) / len(unanswerable)
        variants.append(
            {
                "name": name,
                "minimum_query_coverage": coverage,
                "minimum_rrf_score": score,
                "answerable_coverage": answerable_coverage,
                "false_abstention_rate": 1.0 - answerable_coverage,
                "correct_abstention_rate": correct_abstention,
                "balanced_gate_accuracy": (answerable_coverage + correct_abstention) / 2,
            }
        )
    selected = sorted(
        variants,
        key=lambda row: (
            -row["balanced_gate_accuracy"],
            row["false_abstention_rate"],
            -row["correct_abstention_rate"],
            row["name"],
        ),
    )[0]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "kind": "silver",
            "human_reviewed": False,
            "split": "dev",
            "records": len(dev),
            "answerable": sum(record.candidate.answerable for record in dev),
            "unanswerable": sum(not record.candidate.answerable for record in dev),
            "test_split_accessed_for_tuning": False,
        },
        "selection_policy": (
            "maximize balanced answerable coverage/correct abstention; then minimize "
            "false abstention; fixed four-variant comparison"
        ),
        "selected": selected,
        "variants": variants,
        "observations": observations,
    }
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(variants[0]))
    writer.writeheader()
    writer.writerows(variants)
    markdown = _render_markdown(payload)
    _write_atomic(outputs["json"], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_atomic(outputs["csv"], csv_buffer.getvalue())
    _write_atomic(outputs["md"], markdown)
    return outputs


def _render_markdown(payload: Mapping[str, Any]) -> str:
    selected = payload["selected"]
    lines = [
        "# Silver dev evidence-gate tuning",
        "",
        "Bu çalışma yalnız AI-assisted silver `dev` split üzerindedir; `human_reviewed=false`. ",
        "Test split threshold seçimi sırasında okunmamıştır.",
        "",
        "| Variant | Min coverage | Min RRF | Answerable coverage | Correct abstention | False abstention | Balanced |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| {row['name']} | {row['minimum_query_coverage']:.3f} | "
            f"{row['minimum_rrf_score']:.3f} | {row['answerable_coverage']:.3f} | "
            f"{row['correct_abstention_rate']:.3f} | {row['false_abstention_rate']:.3f} | "
            f"{row['balanced_gate_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Seçim: `{selected['name']}`. Bu küçük dev set üzerinde yapılmış provisional "
            "bir threshold seçimidir; gold veya tamamen human-reviewed sonuç değildir.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    outputs = run_tuning(args.config, overwrite=args.overwrite)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
