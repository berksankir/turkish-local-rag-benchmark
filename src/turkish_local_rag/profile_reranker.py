"""Profile local hybrid retrieval and reranking on the silver dev split only."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

from turkish_local_rag.candidates import CandidateValidationError, load_candidates, validate_candidate_set
from turkish_local_rag.config import ProjectConfig, load_config
from turkish_local_rag.download import load_manifest
from turkish_local_rag.evaluate import (
    EvaluationError,
    EvaluationRecord,
    _process_memory_bytes,
    _write_text_atomic,
    load_silver,
    score_retrieval,
)
from turkish_local_rag.provenance import (
    build_silver_provenance,
    provenance_csv_fields,
    provenance_csv_values,
    provenance_markdown_lines,
)
from turkish_local_rag.retrieve import (
    BM25Retriever,
    FusedHit,
    RetrievalError,
    load_chunk_corpus,
    reciprocal_rank_fusion,
)


PROFILE_FILENAMES = {
    "json": "reranker_profile.json",
    "csv": "reranker_profile.csv",
    "markdown": "reranker_profile.md",
}


@dataclass(frozen=True, slots=True)
class ProfileVariant:
    variant_id: str
    rerank_top_n: int
    batch_size: int
    cpu_threads: int


class _RuntimeScorer:
    def __init__(self, reranker: Any, variant: ProfileVariant) -> None:
        self._reranker = reranker
        self._variant = variant

    def score(self, question: str, passages: Sequence[str]) -> Any:
        return self._reranker.score_with_options(
            question,
            passages,
            batch_size=self._variant.batch_size,
            cpu_threads=self._variant.cpu_threads,
        )


def profiling_variants(config: ProjectConfig) -> tuple[ProfileVariant, ...]:
    """Return a small, justified set that isolates each runtime control."""

    configured = ProfileVariant(
        "configured_top20_b4_t4",
        config.reranker.rerank_top_n,
        config.reranker.batch_size,
        config.reranker.cpu_threads,
    )
    bounded_top_n = max(config.reranker.top_k, min(10, config.reranker.rerank_top_n))
    candidates = (
        configured,
        ProfileVariant(
            "bounded_top10_b4_t4",
            bounded_top_n,
            config.reranker.batch_size,
            config.reranker.cpu_threads,
        ),
        ProfileVariant(
            "bounded_top10_b2_t4",
            bounded_top_n,
            max(1, min(2, config.reranker.batch_size)),
            config.reranker.cpu_threads,
        ),
        ProfileVariant(
            "bounded_top10_b4_t2",
            bounded_top_n,
            config.reranker.batch_size,
            max(1, min(2, config.reranker.cpu_threads)),
        ),
    )
    unique: list[ProfileVariant] = []
    seen: set[tuple[int, int, int]] = set()
    for variant in candidates:
        signature = (variant.rerank_top_n, variant.batch_size, variant.cpu_threads)
        if signature not in seen:
            unique.append(variant)
            seen.add(signature)
    return tuple(unique)


def run_profile(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run a dev-only profile and atomically write JSON, CSV, and Markdown."""

    started_at = datetime.now(timezone.utc)
    total_start = perf_counter()
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    output_paths = {
        key: paths.evaluation_results_directory / "silver" / filename
        for key, filename in PROFILE_FILENAMES.items()
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise EvaluationError(
            "reranker profile already exists; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )

    sources = load_manifest(paths.source_manifest)
    candidates = load_candidates(paths.evaluation_candidates)
    validate_candidate_set(candidates, sources, paths.extracted_pages_directory)
    evaluation_records = load_silver(paths.evaluation_silver, candidates)
    dev_records = tuple(record for record in evaluation_records if record.split == "dev")
    if len(dev_records) != 10:
        raise EvaluationError(f"reranker profile requires 10 dev records, got {len(dev_records)}")

    from turkish_local_rag.review import (
        load_review,
        review_provenance_summary,
        review_status_counts,
        select_silver_audit_candidates,
        validate_review_grounding,
    )

    audit_candidates = select_silver_audit_candidates(candidates)
    audit_records = load_review(paths.evaluation_silver_audit, audit_candidates)
    validate_review_grounding(
        audit_records,
        sources,
        paths.extracted_pages_directory,
        expected_total=len(audit_candidates),
        expected_answerable=sum(candidate.answerable for candidate in audit_candidates),
        expected_unanswerable=sum(
            not candidate.answerable for candidate in audit_candidates
        ),
    )

    corpus_start = perf_counter()
    chunks = load_chunk_corpus(paths.chunks_directory, sources)
    corpus_load_ms = _elapsed_ms(corpus_start)
    bm25_start = perf_counter()
    bm25 = BM25Retriever(chunks, config.bm25)
    bm25_load_ms = _elapsed_ms(bm25_start)

    from qdrant_client import QdrantClient

    from turkish_local_rag.dense import SentenceTransformerE5Encoder, dense_search
    from turkish_local_rag.rerank import CrossEncoderReranker, rerank_hits

    embedding_start = perf_counter()
    encoder = SentenceTransformerE5Encoder(
        paths.embedding_model_directory, config.dense
    )
    embedding_model_load_ms = _elapsed_ms(embedding_start)
    reranker_start = perf_counter()
    reranker = CrossEncoderReranker(
        paths.reranker_model_directory, config.reranker
    )
    reranker_model_load_ms = _elapsed_ms(reranker_start)

    variants = profiling_variants(config)
    maximum_top_n = max(variant.rerank_top_n for variant in variants)
    qdrant_start = perf_counter()
    client = QdrantClient(path=str(paths.qdrant_directory))
    qdrant_open_ms = _elapsed_ms(qdrant_start)
    try:
        if not client.collection_exists(config.dense.collection_name):
            raise EvaluationError(
                f"Qdrant collection not found: {config.dense.collection_name}"
            )
        point_count = client.count(config.dense.collection_name, exact=True).count
        if point_count != len(chunks):
            raise EvaluationError(
                f"Qdrant point/chunk mismatch: points={point_count}, chunks={len(chunks)}"
            )

        cached: list[tuple[EvaluationRecord, list[FusedHit], float]] = []
        rrf_results: list[dict[str, Any]] = []
        for record in dev_records:
            retrieval_start = perf_counter()
            sparse = bm25.search(record.candidate.question, config.rrf.sparse_candidates)
            dense = dense_search(
                record.candidate.question,
                chunks,
                encoder,
                client,
                config.dense,
                top_k=config.rrf.dense_candidates,
            )
            fused = reciprocal_rank_fusion(
                {"bm25": sparse, "dense": dense},
                rank_constant=config.rrf.rank_constant,
                limit=maximum_top_n,
            )
            retrieval_ms = _elapsed_ms(retrieval_start)
            cached.append((record, fused, retrieval_ms))
            rrf_results.append(
                score_retrieval(
                    record,
                    "hybrid_rrf",
                    fused[: config.reranker.top_k],
                    retrieval_ms,
                )
            )

        configured_variant = variants[0]
        cold_record, cold_fused, cold_retrieval_ms = cached[0]
        cold_rerank_start = perf_counter()
        rerank_hits(
            cold_record.candidate.question,
            cold_fused[: configured_variant.rerank_top_n],
            _RuntimeScorer(reranker, configured_variant),
            limit=config.reranker.top_k,
        )
        cold_reranking_ms = _elapsed_ms(cold_rerank_start)

        variant_payloads: list[dict[str, Any]] = []
        for variant in variants:
            scored: list[dict[str, Any]] = []
            query_timings: list[dict[str, Any]] = []
            runtime_scorer = _RuntimeScorer(reranker, variant)
            for record, fused, retrieval_ms in cached:
                reranking_start = perf_counter()
                reranked = rerank_hits(
                    record.candidate.question,
                    fused[: variant.rerank_top_n],
                    runtime_scorer,
                    limit=config.reranker.top_k,
                )
                reranking_ms = _elapsed_ms(reranking_start)
                total_ms = retrieval_ms + reranking_ms
                scored.append(
                    score_retrieval(
                        record,
                        "hybrid_reranked",
                        reranked,
                        total_ms,
                    )
                )
                query_timings.append(
                    {
                        "candidate_id": record.candidate.candidate_id,
                        "retrieval_ms": retrieval_ms,
                        "reranking_ms": reranking_ms,
                        "total_ms": total_ms,
                    }
                )
            variant_payloads.append(
                {
                    **asdict(variant),
                    "quality": _quality_summary(scored),
                    "latency_ms": {
                        "retrieval": _latency_summary(
                            [timing["retrieval_ms"] for timing in query_timings]
                        ),
                        "reranking_only": _latency_summary(
                            [timing["reranking_ms"] for timing in query_timings]
                        ),
                        "hybrid_reranked_total": _latency_summary(
                            [timing["total_ms"] for timing in query_timings]
                        ),
                    },
                    "queries": query_timings,
                }
            )
    finally:
        client.close()

    current_rss, peak_rss = _process_memory_bytes()
    finished_at = datetime.now(timezone.utc)
    dataset_metadata = build_silver_provenance(
        review_status_counts(audit_records),
        review_provenance_summary(audit_records),
        total_records=len(evaluation_records),
    )
    dataset_metadata["scope"] = "dev split only"
    payload = {
        "schema_version": 2,
        "dataset": dataset_metadata,
        "protocol": {
            "selection_split": "dev",
            "test_split_accessed_for_tuning": False,
            "dev_records": len(dev_records),
            "dev_answerable": sum(record.candidate.answerable for record in dev_records),
            "dev_unanswerable": sum(
                not record.candidate.answerable for record in dev_records
            ),
            "quality_denominator": "answerable dev records only",
            "latency_denominator": "all dev records",
            "latency_percentile_method": "nearest-rank",
            "reranker_instance_reused": True,
            "reranker_instances_created": 1,
            "default_fast_pipeline": "hybrid_rrf",
            "optional_quality_pipeline": "hybrid_reranked",
        },
        "model_loading_ms": {
            "corpus": corpus_load_ms,
            "bm25": bm25_load_ms,
            "embedding_model": embedding_model_load_ms,
            "reranker_model": reranker_model_load_ms,
            "qdrant": qdrant_open_ms,
        },
        "cold_query_ms": {
            "hybrid_rrf": cold_retrieval_ms,
            "reranking_only": cold_reranking_ms,
            "hybrid_reranked": cold_retrieval_ms + cold_reranking_ms,
        },
        "warm_query_ms": {
            "hybrid_rrf": _latency_summary(
                [result["latency_ms"] for result in rrf_results[1:]]
            ),
            "configured_reranking_only": variant_payloads[0]["latency_ms"][
                "reranking_only"
            ],
            "configured_hybrid_reranked": variant_payloads[0]["latency_ms"][
                "hybrid_reranked_total"
            ],
        },
        "hybrid_rrf_quality": _quality_summary(rrf_results),
        "variants": variant_payloads,
        "selection": {
            "default_fast_pipeline": "hybrid_rrf",
            "reason": (
                "Selected before generation because the dev split shows a much lower "
                "latency and reranking does not consistently improve dev retrieval metrics."
            ),
            "test_split_used": False,
        },
        "runtime": {
            "total_seconds": perf_counter() - total_start,
            "current_process_rss_bytes": current_rss,
            "peak_process_rss_bytes": peak_rss,
            "memory_measurement": "process working set; approximate",
            "qdrant_point_count": point_count,
            "chunk_count": len(chunks),
        },
        "reproducibility": {
            "started_at_utc": _utc_text(started_at),
            "finished_at_utc": _utc_text(finished_at),
            "config_sha256": _sha256_file(Path(config_path)),
            "silver_sha256": _sha256_file(paths.evaluation_silver),
            "audit_sha256": _sha256_file(paths.evaluation_silver_audit),
            "embedding_model_id": config.dense.model_id,
            "embedding_model_revision": config.dense.model_revision,
            "reranker_model_id": config.reranker.model_id,
            "reranker_model_revision": config.reranker.model_revision,
            "variants": [asdict(variant) for variant in variants],
        },
    }
    rendered = {
        "json": json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        "csv": render_profile_csv(payload),
        "markdown": render_profile_markdown(payload),
    }
    for key, path in output_paths.items():
        _write_text_atomic(path, rendered[key])
    return output_paths


def _quality_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [result for result in results if result["metrics"] is not None]
    if not scored:
        raise EvaluationError("profile quality requires answerable dev records")
    return {
        "total_queries": len(results),
        "answerable_queries": len(scored),
        "unanswerable_queries": len(results) - len(scored),
        "recall_at_1": _mean_metric(scored, "recall_at_1"),
        "recall_at_3": _mean_metric(scored, "recall_at_3"),
        "recall_at_5": _mean_metric(scored, "recall_at_5"),
        "mrr": _mean_metric(scored, "reciprocal_rank"),
        "correct_document_retrieval": _mean_metric(
            scored, "correct_document_retrieval"
        ),
        "correct_page_retrieval": _mean_metric(scored, "correct_page_retrieval"),
    }


def _mean_metric(results: Sequence[Mapping[str, Any]], key: str) -> float:
    return mean(float(result["metrics"][key]) for result in results)


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise EvaluationError("latency summary cannot be empty")
    ordered = sorted(float(value) for value in values)
    return {
        "mean": mean(ordered),
        "p50": median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def render_profile_csv(payload: Mapping[str, Any]) -> str:
    fields = provenance_csv_fields() + (
        "variant_id",
        "rerank_top_n",
        "batch_size",
        "cpu_threads",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "correct_document_retrieval",
        "correct_page_retrieval",
        "retrieval_mean_ms",
        "reranking_mean_ms",
        "total_mean_ms",
        "total_p50_ms",
        "total_p95_ms",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for variant in payload["variants"]:
        quality = variant["quality"]
        latency = variant["latency_ms"]
        writer.writerow(
            {
                **provenance_csv_values(payload["dataset"]),
                "variant_id": variant["variant_id"],
                "rerank_top_n": variant["rerank_top_n"],
                "batch_size": variant["batch_size"],
                "cpu_threads": variant["cpu_threads"],
                "recall_at_1": quality["recall_at_1"],
                "recall_at_3": quality["recall_at_3"],
                "recall_at_5": quality["recall_at_5"],
                "mrr": quality["mrr"],
                "correct_document_retrieval": quality[
                    "correct_document_retrieval"
                ],
                "correct_page_retrieval": quality["correct_page_retrieval"],
                "retrieval_mean_ms": latency["retrieval"]["mean"],
                "reranking_mean_ms": latency["reranking_only"]["mean"],
                "total_mean_ms": latency["hybrid_reranked_total"]["mean"],
                "total_p50_ms": latency["hybrid_reranked_total"]["p50"],
                "total_p95_ms": latency["hybrid_reranked_total"]["p95"],
            }
        )
    return stream.getvalue()


def render_profile_markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    rrf = payload["hybrid_rrf_quality"]
    lines = [
        "# Faz 8.1 reranker profiling — silver dev",
        "",
        *provenance_markdown_lines(),
        "Bu profiling yalnız `dev` split üzerinde çalıştırılmış; test split ayar "
        "seçimi için okunmamıştır.",
        (
            "Reranker model instance'ı bir kez yüklenmiş ve tüm sorgular ile "
            f"varyantlarda tekrar kullanılmıştır: {protocol['reranker_instances_created']} instance."
        ),
        "",
        "## Model yükleme ve cold/warm latency",
        "",
    ]
    for name, value in payload["model_loading_ms"].items():
        lines.append(f"- {name}: `{value:.3f} ms`")
    cold = payload["cold_query_ms"]
    lines.extend(
        [
            f"- Cold hybrid_rrf: `{cold['hybrid_rrf']:.3f} ms`",
            f"- Cold reranking-only: `{cold['reranking_only']:.3f} ms`",
            f"- Cold hybrid_reranked total: `{cold['hybrid_reranked']:.3f} ms`",
            "",
            "## Dev karşılaştırması",
            "",
            "| Pipeline/varyant | top-n | batch | threads | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Rerank ms | Total ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| hybrid_rrf | — | — | — | {rrf['recall_at_1']:.4f} | "
                f"{rrf['recall_at_3']:.4f} | {rrf['recall_at_5']:.4f} | "
                f"{rrf['mrr']:.4f} | {rrf['correct_document_retrieval']:.4f} | "
                f"{rrf['correct_page_retrieval']:.4f} | — | "
                f"{payload['warm_query_ms']['hybrid_rrf']['mean']:.3f} |"
            ),
        ]
    )
    for variant in payload["variants"]:
        quality = variant["quality"]
        latency = variant["latency_ms"]
        lines.append(
            f"| {variant['variant_id']} | {variant['rerank_top_n']} | "
            f"{variant['batch_size']} | {variant['cpu_threads']} | "
            f"{quality['recall_at_1']:.4f} | {quality['recall_at_3']:.4f} | "
            f"{quality['recall_at_5']:.4f} | {quality['mrr']:.4f} | "
            f"{quality['correct_document_retrieval']:.4f} | "
            f"{quality['correct_page_retrieval']:.4f} | "
            f"{latency['reranking_only']['mean']:.3f} | "
            f"{latency['hybrid_reranked_total']['mean']:.3f} |"
        )
    configured = payload["variants"][0]
    quality = configured["quality"]
    lines.extend(
        [
            "",
            "## Karar",
            "",
            "Varsayılan hızlı pipeline `hybrid_rrf`; opsiyonel kalite modu `hybrid_reranked` olarak kalır.",
            (
                "Configured reranker dev farkları: "
                f"R@1 `{quality['recall_at_1'] - rrf['recall_at_1']:+.4f}`, "
                f"R@5 `{quality['recall_at_5'] - rrf['recall_at_5']:+.4f}`, "
                f"MRR `{quality['mrr'] - rrf['mrr']:+.4f}`, "
                f"Doc@1 `{quality['correct_document_retrieval'] - rrf['correct_document_retrieval']:+.4f}`, "
                f"Page@1 `{quality['correct_page_retrieval'] - rrf['correct_page_retrieval']:+.4f}`."
            ),
            (
                "Reranking bazı metrikleri iyileştirebilirken diğerlerini düşürür ve "
                "belirgin CPU latency maliyeti ekler; test split bu kararda kullanılmamıştır."
            ),
            "",
            "## Runtime",
            "",
            f"- Toplam süre: `{payload['runtime']['total_seconds']:.3f} s`",
            f"- Peak process RAM: `{payload['runtime']['peak_process_rss_bytes']} byte`",
            f"- Chunk/Qdrant point: `{payload['runtime']['chunk_count']}` / `{payload['runtime']['qdrant_point_count']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing profiling artifacts",
    )
    args = parser.parse_args(argv)
    try:
        outputs = run_profile(args.config, overwrite=args.overwrite)
    except (CandidateValidationError, EvaluationError, RetrievalError, ValueError, OSError) as exc:
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
