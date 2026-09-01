"""Validate reviewed gold data and benchmark local retrieval pipelines."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

from turkish_local_rag.candidates import (
    CANDIDATE_KEYS,
    Candidate,
    CandidateValidationError,
    load_candidates,
    parse_candidate_record,
    validate_candidate_set,
)
from turkish_local_rag.config import ProjectConfig, ResolvedPaths, load_config
from turkish_local_rag.download import SourceDocument, load_manifest
from turkish_local_rag.retrieve import (
    BM25Retriever,
    FusedHit,
    RetrievalError,
    RetrievalHit,
    load_chunk_corpus,
    reciprocal_rank_fusion,
)


PIPELINES = ("dense", "bm25", "hybrid_rrf", "hybrid_reranked")
SPLITS = ("dev", "test")
EXPECTED_SPLIT_COUNTS = {
    "dev": {"total": 10, "answerable": 8, "unanswerable": 2},
    "test": {"total": 40, "answerable": 32, "unanswerable": 8},
}
RESULT_FILENAMES = {
    "json": "retrieval_benchmark.json",
    "csv": "retrieval_benchmark.csv",
    "markdown": "retrieval_benchmark.md",
}


class EvaluationError(RuntimeError):
    """Raised when reviewed gold data or benchmark execution is invalid."""


@dataclass(frozen=True, slots=True)
class GoldRecord:
    candidate: Candidate
    split: str


def load_gold(
    path: str | Path,
    approved_candidates: Sequence[Candidate],
    *,
    expected_split_counts: Mapping[str, Mapping[str, int]] = EXPECTED_SPLIT_COUNTS,
) -> tuple[GoldRecord, ...]:
    """Load gold JSONL and prove every record equals its approved candidate."""

    gold_path = Path(path)
    try:
        lines = gold_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise EvaluationError(f"gold file not found: {gold_path}") from exc
    approved_by_id = {
        candidate.candidate_id: candidate for candidate in approved_candidates
    }
    records: list[GoldRecord] = []
    seen_ids: set[str] = set()
    expected_keys = CANDIDATE_KEYS | {"split"}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationError(f"blank gold JSONL record at line {line_number}")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"invalid gold JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise EvaluationError(
                f"gold line {line_number} must contain approved candidate fields and split"
            )
        split = raw.pop("split")
        if split not in SPLITS:
            raise EvaluationError(
                f"gold line {line_number}.split must be dev or test"
            )
        try:
            candidate = parse_candidate_record(raw, line_number)
        except CandidateValidationError as exc:
            raise EvaluationError(str(exc)) from exc
        if candidate.candidate_id in seen_ids:
            raise EvaluationError(
                f"duplicate gold candidate_id: {candidate.candidate_id}"
            )
        seen_ids.add(candidate.candidate_id)
        approved = approved_by_id.get(candidate.candidate_id)
        if approved is None:
            raise EvaluationError(
                f"gold record is not an approved candidate: {candidate.candidate_id}"
            )
        if candidate != approved:
            raise EvaluationError(
                f"gold record changed after review: {candidate.candidate_id}"
            )
        records.append(GoldRecord(candidate=candidate, split=split))

    if seen_ids != set(approved_by_id):
        missing = sorted(set(approved_by_id) - seen_ids)
        extra = sorted(seen_ids - set(approved_by_id))
        raise EvaluationError(
            f"gold/approved candidate ID mismatch: missing={missing}, extra={extra}"
        )
    _validate_split_counts(records, expected_split_counts)
    return tuple(records)


def _validate_split_counts(
    records: Sequence[GoldRecord],
    expected_split_counts: Mapping[str, Mapping[str, int]],
) -> None:
    for split, expected in expected_split_counts.items():
        selected = [record for record in records if record.split == split]
        answerable = sum(record.candidate.answerable for record in selected)
        actual = {
            "total": len(selected),
            "answerable": answerable,
            "unanswerable": len(selected) - answerable,
        }
        if actual != expected:
            raise EvaluationError(
                f"invalid {split} split counts: expected={expected}, actual={actual}"
            )


def score_retrieval(
    gold: GoldRecord,
    pipeline: str,
    hits: Sequence[Any],
    latency_ms: float,
) -> dict[str, Any]:
    """Score one ranking; relevance requires both trusted document and page."""

    candidate = gold.candidate
    serialized_hits = [_serialize_hit(hit) for hit in hits]
    result: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "split": gold.split,
        "pipeline": pipeline,
        "question": candidate.question,
        "answerable": candidate.answerable,
        "expected_document_id": candidate.document_id,
        "expected_physical_pages": list(candidate.physical_pages),
        "latency_ms": latency_ms,
        "hits": serialized_hits,
    }
    if not candidate.answerable:
        result["metrics"] = None
        return result

    relevant_ranks = [
        index
        for index, hit in enumerate(hits, start=1)
        if hit.chunk.document_id == candidate.document_id
        and hit.chunk.page_number in candidate.physical_pages
    ]
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
    top_hit = hits[0] if hits else None
    result["metrics"] = {
        "recall_at_1": bool(first_relevant_rank and first_relevant_rank <= 1),
        "recall_at_3": bool(first_relevant_rank and first_relevant_rank <= 3),
        "recall_at_5": bool(first_relevant_rank and first_relevant_rank <= 5),
        "reciprocal_rank": (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
        "correct_document_retrieval": bool(
            top_hit and top_hit.chunk.document_id == candidate.document_id
        ),
        "correct_page_retrieval": bool(
            top_hit
            and top_hit.chunk.document_id == candidate.document_id
            and top_hit.chunk.page_number in candidate.physical_pages
        ),
    }
    return result


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate answerable retrieval quality and all-query latency by split."""

    aggregate: dict[str, Any] = {}
    for split in (*SPLITS, "all"):
        aggregate[split] = {}
        for pipeline in PIPELINES:
            selected = [
                result
                for result in results
                if result["pipeline"] == pipeline
                and (split == "all" or result["split"] == split)
            ]
            scored = [result for result in selected if result["metrics"] is not None]
            if not selected or not scored:
                raise EvaluationError(
                    f"cannot aggregate empty result group: {split}/{pipeline}"
                )
            aggregate[split][pipeline] = {
                "total_queries": len(selected),
                "answerable_queries": len(scored),
                "unanswerable_queries": len(selected) - len(scored),
                "recall_at_1": _mean_metric(scored, "recall_at_1"),
                "recall_at_3": _mean_metric(scored, "recall_at_3"),
                "recall_at_5": _mean_metric(scored, "recall_at_5"),
                "mrr": _mean_metric(scored, "reciprocal_rank"),
                "correct_document_retrieval": _mean_metric(
                    scored, "correct_document_retrieval"
                ),
                "correct_page_retrieval": _mean_metric(
                    scored, "correct_page_retrieval"
                ),
                "average_retrieval_latency_ms": sum(
                    float(result["latency_ms"]) for result in selected
                )
                / len(selected),
            }
    return aggregate


def run_benchmark(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run all four local pipelines and atomically write JSON, CSV, and Markdown."""

    started_at = datetime.now(timezone.utc)
    initialization_start = perf_counter()
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    sources = load_manifest(paths.source_manifest)
    candidates = load_candidates(paths.evaluation_candidates)
    validate_candidate_set(candidates, sources, paths.extracted_pages_directory)
    gold = load_gold(paths.evaluation_gold, candidates)
    chunks = load_chunk_corpus(paths.chunks_directory, sources)
    if config.reranker.top_k < 5:
        raise EvaluationError("common retrieval result limit must be at least 5")
    result_limit = config.reranker.top_k
    output_paths = {
        key: paths.evaluation_results_directory / filename
        for key, filename in RESULT_FILENAMES.items()
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise EvaluationError(
            "benchmark result already exists; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )

    from qdrant_client import QdrantClient

    from turkish_local_rag.dense import SentenceTransformerE5Encoder, dense_search
    from turkish_local_rag.rerank import CrossEncoderReranker, rerank_hits

    bm25 = BM25Retriever(chunks, config.bm25)
    encoder = SentenceTransformerE5Encoder(
        paths.embedding_model_directory, config.dense
    )
    reranker = CrossEncoderReranker(
        paths.reranker_model_directory, config.reranker
    )
    if not paths.qdrant_directory.is_dir():
        raise EvaluationError(f"Qdrant directory not found: {paths.qdrant_directory}")
    client = QdrantClient(path=str(paths.qdrant_directory))
    try:
        if not client.collection_exists(config.dense.collection_name):
            raise EvaluationError(
                f"Qdrant collection not found: {config.dense.collection_name}"
            )
        indexed_points = client.count(
            config.dense.collection_name, exact=True
        ).count
        if indexed_points != len(chunks):
            raise EvaluationError(
                f"Qdrant point/chunk mismatch: points={indexed_points}, chunks={len(chunks)}"
            )
        _warm_up(
            bm25,
            chunks,
            encoder,
            reranker,
            client,
            config,
            result_limit,
            dense_search,
            rerank_hits,
        )
        initialization_seconds = perf_counter() - initialization_start
        ordered_gold = sorted(
            gold,
            key=lambda record: (
                SPLITS.index(record.split), record.candidate.candidate_id
            ),
        )
        results: list[dict[str, Any]] = []
        benchmark_start = perf_counter()
        for pipeline in PIPELINES:
            for record in ordered_gold:
                start = perf_counter()
                hits = _run_pipeline(
                    pipeline,
                    record.candidate.question,
                    bm25,
                    chunks,
                    encoder,
                    reranker,
                    client,
                    config,
                    result_limit,
                    dense_search,
                    rerank_hits,
                )
                latency_ms = (perf_counter() - start) * 1000.0
                results.append(
                    score_retrieval(record, pipeline, hits, latency_ms)
                )
        benchmark_seconds = perf_counter() - benchmark_start
    finally:
        client.close()

    current_rss, peak_rss = _process_memory_bytes()
    finished_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "protocol": {
            "pipelines": list(PIPELINES),
            "common_result_limit": result_limit,
            "relevance": "matching trusted document_id and physical page",
            "mrr_scope": f"returned top-{result_limit}",
            "latency_scope": "per-query retrieval; model loading and warm-up excluded",
            "split_policy": (
                "fixed before retrieval: dev=8 answerable+2 unanswerable; "
                "test=32 answerable+8 unanswerable"
            ),
        },
        "reproducibility": _reproducibility_metadata(
            config_path, config, paths, sources, chunks, started_at, finished_at
        ),
        "runtime": {
            "initialization_seconds": initialization_seconds,
            "benchmark_seconds": benchmark_seconds,
            "current_process_rss_bytes": current_rss,
            "peak_process_rss_bytes": peak_rss,
            "memory_measurement": "process working set; approximate",
            "qdrant_point_count": indexed_points,
            "chunk_count": len(chunks),
        },
        "summary": aggregate_results(results),
        "queries": results,
    }
    rendered = {
        "json": json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        "csv": _render_csv(results),
        "markdown": _render_markdown(payload),
    }
    for key, path in output_paths.items():
        _write_text_atomic(path, rendered[key])
    return output_paths


def _run_pipeline(
    pipeline: str,
    question: str,
    bm25: BM25Retriever,
    chunks: Sequence[Any],
    encoder: Any,
    reranker: Any,
    client: Any,
    config: ProjectConfig,
    result_limit: int,
    dense_search_function: Any,
    rerank_function: Any,
) -> Sequence[Any]:
    if pipeline == "bm25":
        return bm25.search(question, result_limit)
    if pipeline == "dense":
        return dense_search_function(
            question,
            chunks,
            encoder,
            client,
            config.dense,
            top_k=result_limit,
        )
    sparse_hits = bm25.search(question, config.rrf.sparse_candidates)
    dense_hits = dense_search_function(
        question,
        chunks,
        encoder,
        client,
        config.dense,
        top_k=config.rrf.dense_candidates,
    )
    fusion_limit = (
        config.reranker.candidate_count
        if pipeline == "hybrid_reranked"
        else result_limit
    )
    fused = reciprocal_rank_fusion(
        {"bm25": sparse_hits, "dense": dense_hits},
        rank_constant=config.rrf.rank_constant,
        limit=fusion_limit,
    )
    if pipeline == "hybrid_rrf":
        return fused
    if pipeline == "hybrid_reranked":
        return rerank_function(question, fused, reranker, limit=result_limit)
    raise EvaluationError(f"unknown pipeline: {pipeline}")


def _warm_up(
    bm25: BM25Retriever,
    chunks: Sequence[Any],
    encoder: Any,
    reranker: Any,
    client: Any,
    config: ProjectConfig,
    result_limit: int,
    dense_search_function: Any,
    rerank_function: Any,
) -> None:
    question = "Üniversite yönetim kurulu görevleri"
    sparse = bm25.search(question, config.rrf.sparse_candidates)
    dense = dense_search_function(
        question,
        chunks,
        encoder,
        client,
        config.dense,
        top_k=config.rrf.dense_candidates,
    )
    fused = reciprocal_rank_fusion(
        {"bm25": sparse, "dense": dense},
        rank_constant=config.rrf.rank_constant,
        limit=config.reranker.candidate_count,
    )
    rerank_function(question, fused, reranker, limit=result_limit)


def _serialize_hit(hit: Any) -> dict[str, Any]:
    chunk = hit.chunk
    result: dict[str, Any] = {
        "rank": hit.rank,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "page_number": chunk.page_number,
        "title": chunk.title,
        "source_page_url": chunk.source_page_url,
        "pdf_url": chunk.pdf_url,
        "pdf_sha256": chunk.pdf_sha256,
    }
    for field in ("score", "rrf_score", "reranker_score"):
        if hasattr(hit, field):
            result[field] = float(getattr(hit, field))
    return result


def _mean_metric(results: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(result["metrics"][key]) for result in results) / len(results)


def _reproducibility_metadata(
    config_path: str | Path,
    config: ProjectConfig,
    paths: ResolvedPaths,
    sources: Sequence[SourceDocument],
    chunks: Sequence[Any],
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    chunk_hashes = {
        source.id: _sha256_file(
            paths.chunks_directory / f"{source.id}.chunks.jsonl"
        )
        for source in sources
    }
    logical_corpus = hashlib.sha256()
    for source in sources:
        logical_corpus.update(source.id.encode("utf-8"))
        logical_corpus.update(chunk_hashes[source.id].encode("ascii"))
    return {
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at_utc": finished_at.isoformat().replace("+00:00", "Z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "embedding_model_id": config.dense.model_id,
        "embedding_model_revision": config.dense.model_revision,
        "embedding_model_sha256": config.dense.model_sha256,
        "reranker_model_id": config.reranker.model_id,
        "reranker_model_revision": config.reranker.model_revision,
        "reranker_model_sha256": config.reranker.model_sha256,
        "config_sha256": _sha256_file(Path(config_path)),
        "config": asdict(config),
        "manifest_sha256": _sha256_file(paths.source_manifest),
        "approved_candidates_sha256": _sha256_file(paths.evaluation_candidates),
        "gold_sha256": _sha256_file(paths.evaluation_gold),
        "chunk_file_sha256": chunk_hashes,
        "logical_corpus_sha256": logical_corpus.hexdigest(),
        "document_count": len(sources),
        "chunk_count": len(chunks),
    }


def _render_csv(results: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "candidate_id",
        "split",
        "pipeline",
        "answerable",
        "expected_document_id",
        "expected_physical_pages",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank",
        "correct_document_retrieval",
        "correct_page_retrieval",
        "latency_ms",
        "top_1_chunk_id",
        "top_1_document_id",
        "top_1_page_number",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for result in results:
        metrics = result["metrics"] or {}
        top_hit = result["hits"][0] if result["hits"] else {}
        writer.writerow(
            {
                "candidate_id": result["candidate_id"],
                "split": result["split"],
                "pipeline": result["pipeline"],
                "answerable": str(result["answerable"]).lower(),
                "expected_document_id": result["expected_document_id"] or "",
                "expected_physical_pages": ";".join(
                    str(page) for page in result["expected_physical_pages"]
                ),
                "recall_at_1": metrics.get("recall_at_1", ""),
                "recall_at_3": metrics.get("recall_at_3", ""),
                "recall_at_5": metrics.get("recall_at_5", ""),
                "reciprocal_rank": metrics.get("reciprocal_rank", ""),
                "correct_document_retrieval": metrics.get(
                    "correct_document_retrieval", ""
                ),
                "correct_page_retrieval": metrics.get(
                    "correct_page_retrieval", ""
                ),
                "latency_ms": result["latency_ms"],
                "top_1_chunk_id": top_hit.get("chunk_id", ""),
                "top_1_document_id": top_hit.get("document_id", ""),
                "top_1_page_number": top_hit.get("page_number", ""),
            }
        )
    return stream.getvalue()


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Retrieval benchmark",
        "",
        "Bu rapor insan onaylı gold set üzerinde CPU ve local modellerle üretilmiştir.",
        "Unanswerable kayıtlar retrieval kalite paydalarına alınmamış, latency ölçümüne dahil edilmiştir.",
        "MRR, ortak top-10 sonuç listesi üzerinde hesaplanmıştır.",
        "",
    ]
    for split in (*SPLITS, "all"):
        lines.extend(
            [
                f"## {split}",
                "",
                "| Pipeline | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Avg ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pipeline in PIPELINES:
            metrics = payload["summary"][split][pipeline]
            lines.append(
                "| "
                + " | ".join(
                    [
                        pipeline,
                        _format_metric(metrics["recall_at_1"]),
                        _format_metric(metrics["recall_at_3"]),
                        _format_metric(metrics["recall_at_5"]),
                        _format_metric(metrics["mrr"]),
                        _format_metric(metrics["correct_document_retrieval"]),
                        _format_metric(metrics["correct_page_retrieval"]),
                        f"{metrics['average_retrieval_latency_ms']:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    runtime = payload["runtime"]
    reproducibility = payload["reproducibility"]
    lines.extend(
        [
            "## Reproducibility",
            "",
            f"- Timestamp: `{reproducibility['finished_at_utc']}`",
            f"- Gold SHA-256: `{reproducibility['gold_sha256']}`",
            f"- Logical corpus SHA-256: `{reproducibility['logical_corpus_sha256']}`",
            f"- Embedding revision: `{reproducibility['embedding_model_revision']}`",
            f"- Reranker revision: `{reproducibility['reranker_model_revision']}`",
            f"- Benchmark süresi: `{runtime['benchmark_seconds']:.3f} s`",
            f"- Peak process working set: `{runtime['peak_process_rss_bytes']} byte`",
            "",
            "Bu sonuçlar yalnız bu küçük corpus ve makine çalıştırması için geçerlidir. "
            "Reranker Türkçe için zero-shot'tır; test split'ine göre ayar yapılmamıştır.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
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


def _process_memory_bytes() -> tuple[int | None, int | None]:
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if success:
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        return None, None
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak *= 1024
        return None, peak
    except (ImportError, ValueError):
        return None, None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing benchmark result files",
    )
    args = parser.parse_args(argv)
    try:
        outputs = run_benchmark(args.config, overwrite=args.overwrite)
    except (
        CandidateValidationError,
        EvaluationError,
        RetrievalError,
        ValueError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {key: str(path) for key, path in outputs.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
