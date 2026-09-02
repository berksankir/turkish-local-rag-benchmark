"""Evaluate grounded generation on the AI-assisted silver benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

from turkish_local_rag.candidates import load_candidates
from turkish_local_rag.config import load_config
from turkish_local_rag.evaluate import _process_memory_bytes, load_silver
from turkish_local_rag.query import build_service
from turkish_local_rag.provenance import (
    build_silver_provenance,
    provenance_csv_fields,
    provenance_csv_values,
    provenance_markdown_lines,
)
from turkish_local_rag.retrieve import normalize_turkish, turkish_tokenize
from turkish_local_rag.review import (
    load_review,
    review_provenance_summary,
    review_status_counts,
    select_silver_audit_candidates,
)


PIPELINES = ("hybrid_rrf", "hybrid_reranked")


def run_evaluation(config_path: str | Path, *, overwrite: bool = False) -> dict[str, Path]:
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    candidates = load_candidates(paths.evaluation_candidates)
    records = load_silver(paths.evaluation_silver, candidates)
    audit_candidates = select_silver_audit_candidates(candidates)
    audit = load_review(paths.evaluation_silver_audit, audit_candidates)
    outputs = {
        extension: paths.evaluation_results_directory
        / "silver"
        / f"generation_benchmark.{extension}"
        for extension in ("json", "csv", "md")
    }
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise RuntimeError("generation benchmark exists; use --overwrite explicitly")

    service, retriever, generator = build_service(config_path)
    started = datetime.now(timezone.utc)
    initialization_start = perf_counter()
    generator.start()
    initialization_seconds = perf_counter() - initialization_start
    results: list[dict[str, Any]] = []
    benchmark_start = perf_counter()
    try:
        ordered = sorted(records, key=lambda row: (row.split, row.candidate.candidate_id))
        total_queries = len(PIPELINES) * len(ordered)
        completed = 0
        for pipeline in PIPELINES:
            for record in ordered:
                response = service.answer(record.candidate.question, pipeline)
                results.append(_score(record, response))
                completed += 1
                print(
                    f"[{completed}/{total_queries}] {pipeline} "
                    f"{record.candidate.candidate_id} "
                    f"abstained={response['abstained']} "
                    f"total_ms={response['latency_ms']['total']:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
        benchmark_seconds = perf_counter() - benchmark_start
        current_rss, python_peak = _process_memory_bytes()
        llama_peak = _windows_peak_working_set(generator.process_id)
    finally:
        generator.close()
        retriever.close()
    finished = datetime.now(timezone.utc)
    dataset_metadata = build_silver_provenance(
        review_status_counts(audit),
        review_provenance_summary(audit),
        total_records=len(records),
    )
    dataset_metadata["records"] = len(records)
    payload = {
        "schema_version": 3,
        "dataset": dataset_metadata,
        "protocol": {
            "pipelines": list(PIPELINES),
            "threshold_source": "silver dev only",
            "test_used_for_model_or_threshold_selection": False,
            "llm_as_a_judge": False,
            "citation_accuracy": (
                "successful answer citation must match expected trusted document/page "
                "and contain the exact source span after whitespace/case normalization"
            ),
            "key_fact_coverage": "deterministic mean reference-fact token recall",
            "generator_output_schema_version": "1.1",
            "llama_cpp_response_format": {
                "runtime_release": config.generation.runtime_version,
                "type": "json_object",
                "schema_location": "top-level json_schema",
                "documentation": (
                    "https://github.com/ggml-org/llama.cpp/blob/"
                    "b10621/tools/server/README.md"
                ),
            },
        },
        "runtime": {
            "started_at_utc": started.isoformat().replace("+00:00", "Z"),
            "finished_at_utc": finished.isoformat().replace("+00:00", "Z"),
            "initialization_seconds": initialization_seconds,
            "benchmark_seconds": benchmark_seconds,
            "generator_start_count": generator.start_count,
            "current_python_rss_bytes": current_rss,
            "peak_python_rss_bytes": python_peak,
            "peak_llama_server_rss_bytes": llama_peak,
            "approximate_peak_process_tree_rss_bytes": (
                python_peak + llama_peak
                if python_peak is not None and llama_peak is not None
                else None
            ),
        },
        "models": {
            "embedding": {
                "model_id": config.dense.model_id,
                "revision": config.dense.model_revision,
                "sha256": config.dense.model_sha256,
            },
            "reranker": {
                "model_id": config.reranker.model_id,
                "revision": config.reranker.model_revision,
                "sha256": config.reranker.model_sha256,
            },
            "generator": dict(generator.metadata),
        },
        "summary": _aggregate(results),
        "queries": results,
    }
    _write_atomic(outputs["json"], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_atomic(outputs["csv"], _render_csv(results, dataset_metadata))
    _write_atomic(outputs["md"], _render_markdown(payload))
    return outputs


def _score(record: Any, response: Mapping[str, Any]) -> dict[str, Any]:
    candidate = record.candidate
    hits = response["retrieved_chunks"]
    relevant_ranks = [
        hit["rank"]
        for hit in hits
        if hit["document_id"] == candidate.document_id
        and hit["physical_page"] in candidate.physical_pages
    ] if candidate.answerable else []
    first = min(relevant_ranks) if relevant_ranks else None
    citation_accuracy = None
    token_f1 = None
    key_fact_coverage = None
    if candidate.answerable and not response["abstained"]:
        retrieved_by_id = {hit["chunk_id"]: hit for hit in hits}
        span = _normalized_text(candidate.exact_source_span or "")
        citations = response["citations"]
        citation_accuracy = bool(citations) and all(
            citation["document_id"] == candidate.document_id
            and citation["physical_page"] in candidate.physical_pages
            and span in _normalized_text(retrieved_by_id[citation["chunk_id"]]["text"])
            for citation in citations
        )
        token_f1 = _token_f1(response["answer"], candidate.proposed_reference_answer)
        key_fact_coverage = _key_fact_coverage(
            response["answer"], candidate.required_key_facts
        )
    return {
        "candidate_id": candidate.candidate_id,
        "split": record.split,
        "pipeline": response["pipeline"],
        "question": candidate.question,
        "answerable": candidate.answerable,
        "answer": response["answer"],
        "abstained": response["abstained"],
        "abstention_reason": response["abstention_reason"],
        "generation_error": response.get("generation_error"),
        "citations": response["citations"],
        "retrieved_chunks": hits,
        "retrieval": {
            "recall_at_1": bool(first and first <= 1) if candidate.answerable else None,
            "recall_at_3": bool(first and first <= 3) if candidate.answerable else None,
            "recall_at_5": bool(first and first <= 5) if candidate.answerable else None,
            "reciprocal_rank": (1.0 / first if first else 0.0) if candidate.answerable else None,
            "correct_document": (
                bool(hits and hits[0]["document_id"] == candidate.document_id)
                if candidate.answerable else None
            ),
            "correct_page": (
                bool(
                    hits
                    and hits[0]["document_id"] == candidate.document_id
                    and hits[0]["physical_page"] in candidate.physical_pages
                ) if candidate.answerable else None
            ),
        },
        "citation_accuracy": citation_accuracy,
        "token_f1": token_f1,
        "key_fact_coverage": key_fact_coverage,
        "scores": response["scores"],
        "latency_ms": response["latency_ms"],
    }


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("dev", "test", "all"):
        output[split] = {}
        for pipeline in PIPELINES:
            rows = [
                row for row in results
                if row["pipeline"] == pipeline and (split == "all" or row["split"] == split)
            ]
            answerable = [row for row in rows if row["answerable"]]
            unanswerable = [row for row in rows if not row["answerable"]]
            successful = [row for row in answerable if not row["abstained"]]
            output[split][pipeline] = {
                "queries": len(rows),
                "answerable": len(answerable),
                "unanswerable": len(unanswerable),
                "recall_at_1": _mean(answerable, lambda r: r["retrieval"]["recall_at_1"]),
                "recall_at_3": _mean(answerable, lambda r: r["retrieval"]["recall_at_3"]),
                "recall_at_5": _mean(answerable, lambda r: r["retrieval"]["recall_at_5"]),
                "mrr": _mean(answerable, lambda r: r["retrieval"]["reciprocal_rank"]),
                "correct_document_retrieval": _mean(answerable, lambda r: r["retrieval"]["correct_document"]),
                "correct_page_retrieval": _mean(answerable, lambda r: r["retrieval"]["correct_page"]),
                "citation_accuracy": _mean(successful, lambda r: r["citation_accuracy"]),
                "answerable_coverage": sum(not row["abstained"] for row in answerable) / len(answerable),
                "correct_abstention_rate": sum(row["abstained"] for row in unanswerable) / len(unanswerable),
                "false_abstention_rate": sum(row["abstained"] for row in answerable) / len(answerable),
                "token_f1": _mean(successful, lambda r: r["token_f1"]),
                "key_fact_coverage": _mean(successful, lambda r: r["key_fact_coverage"]),
                "latency_ms": {
                    name: _latency_summary([float(row["latency_ms"][name]) for row in rows])
                    for name in ("retrieval", "reranking", "generation", "total")
                },
            }
    return output


def _token_f1(answer: str, reference: str) -> float:
    predicted = turkish_tokenize(answer)
    expected = turkish_tokenize(reference)
    if not predicted or not expected:
        return 0.0
    common = 0
    remaining = list(expected)
    for token in predicted:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def _key_fact_coverage(answer: str, facts: Sequence[str]) -> float:
    answer_tokens = set(turkish_tokenize(answer))
    recalls = []
    for fact in facts:
        tokens = set(turkish_tokenize(fact))
        recalls.append(len(tokens & answer_tokens) / len(tokens) if tokens else 0.0)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _normalized_text(text: str) -> str:
    return " ".join(normalize_turkish(text).split())


def _mean(rows: Sequence[Mapping[str, Any]], value: Any) -> float | None:
    return sum(float(value(row)) for row in rows) / len(rows) if rows else None


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


def _render_csv(
    results: Sequence[Mapping[str, Any]], dataset: Mapping[str, Any]
) -> str:
    output = io.StringIO(newline="")
    result_fields = [
        "candidate_id", "split", "pipeline", "answerable", "abstained",
        "abstention_reason", "answer", "citation_accuracy", "token_f1",
        "key_fact_coverage", "generator_error_category",
        "generator_error_summary", "generator_error_attempts",
    ]
    fields = [
        *provenance_csv_fields(),
        *result_fields,
        "retrieval_ms", "reranking_ms", "generation_ms", "total_ms",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow({
            **provenance_csv_values(dataset),
            **{key: _csv_result_value(row, key) for key in result_fields},
            "retrieval_ms": row["latency_ms"]["retrieval"],
            "reranking_ms": row["latency_ms"]["reranking"],
            "generation_ms": row["latency_ms"]["generation"],
            "total_ms": row["latency_ms"]["total"],
        })
    return output.getvalue()


def _csv_result_value(row: Mapping[str, Any], key: str) -> Any:
    diagnostic = row.get("generation_error") or {}
    diagnostic_fields = {
        "generator_error_category": "category",
        "generator_error_summary": "validation_summary",
        "generator_error_attempts": "attempts",
    }
    if key in diagnostic_fields:
        return diagnostic.get(diagnostic_fields[key])
    return row[key]


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# AI-assisted silver grounded-generation benchmark",
        "",
        *provenance_markdown_lines(),
        "Test split model, pipeline veya threshold seçimi için kullanılmamıştır. "
        "LLM-as-a-judge yoktur.",
        "",
        "| Split | Pipeline | R@1 | R@3 | R@5 | MRR | Citation | Coverage | Correct abstain | False abstain | Token F1 | Key facts | Mean total ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "test", "all"):
        for pipeline in PIPELINES:
            row = payload["summary"][split][pipeline]
            lines.append(
                f"| {split} | {pipeline} | {_fmt(row['recall_at_1'])} | {_fmt(row['recall_at_3'])} | "
                f"{_fmt(row['recall_at_5'])} | {_fmt(row['mrr'])} | {_fmt(row['citation_accuracy'])} | "
                f"{_fmt(row['answerable_coverage'])} | {_fmt(row['correct_abstention_rate'])} | "
                f"{_fmt(row['false_abstention_rate'])} | {_fmt(row['token_f1'])} | "
                f"{_fmt(row['key_fact_coverage'])} | {row['latency_ms']['total']['mean']:.1f} |"
            )
    runtime = payload["runtime"]
    generator_failures: dict[str, int] = {}
    for row in payload["queries"]:
        diagnostic = row.get("generation_error")
        if diagnostic:
            category = diagnostic["category"]
            generator_failures[category] = generator_failures.get(category, 0) + 1
    lines.extend([
        "",
        "## Latency (all split, milliseconds)",
        "",
        "| Pipeline | Stage | Mean | p50 | p95 |",
        "|---|---|---:|---:|---:|",
    ])
    for pipeline in PIPELINES:
        for stage in ("retrieval", "reranking", "generation", "total"):
            latency = payload["summary"]["all"][pipeline]["latency_ms"][stage]
            lines.append(
                f"| {pipeline} | {stage} | {latency['mean']:.1f} | "
                f"{latency['p50']:.1f} | {latency['p95']:.1f} |"
            )
    lines.extend([
        "",
        f"Generator initialization: {runtime['initialization_seconds']:.3f} s; benchmark: "
        f"{runtime['benchmark_seconds']:.3f} s; generator start count: {runtime['generator_start_count']}.",
        f"Approximate peak process-tree RAM: {runtime['approximate_peak_process_tree_rss_bytes']} bytes.",
        "",
        "Latency alanları retrieval, reranking, generation ve total olarak ayrı ölçülmüştür. "
        "Citation ve cevap metrikleri deterministic karşılaştırmalardır; semantik judge kullanılmaz.",
        "Pipeline'lar ardışık çalıştırıldığı, response uzunlukları değiştiği ve cache ısındığı için "
        "generation latency farkı doğrudan reranker hız etkisi olarak yorumlanamaz.",
        "Generator validation failures (all pipelines combined): "
        + (
            ", ".join(f"`{key}`={value}" for key, value in sorted(generator_failures.items()))
            if generator_failures
            else "none"
        )
        + ". All such failures remain fail-closed abstentions; raw model responses are not stored.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


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


def _windows_peak_working_set(pid: int | None) -> int | None:
    if os.name != "nt" or pid is None:
        return None
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    outputs = run_evaluation(args.config, overwrite=args.overwrite)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
