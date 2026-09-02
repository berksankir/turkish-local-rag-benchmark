from pathlib import Path

from turkish_local_rag.config import load_config
from turkish_local_rag.profile_reranker import (
    _latency_summary,
    profiling_variants,
    render_profile_csv,
    render_profile_markdown,
)


DEFAULT_CONFIG = Path("config/default.toml")


def _payload() -> dict:
    quality = {
        "total_queries": 10,
        "answerable_queries": 8,
        "unanswerable_queries": 2,
        "recall_at_1": 0.5,
        "recall_at_3": 0.75,
        "recall_at_5": 1.0,
        "mrr": 0.625,
        "correct_document_retrieval": 0.75,
        "correct_page_retrieval": 0.5,
    }
    latency = {
        "retrieval": {"mean": 20.0, "p50": 19.0, "p95": 30.0},
        "reranking_only": {"mean": 100.0, "p50": 95.0, "p95": 140.0},
        "hybrid_reranked_total": {
            "mean": 120.0,
            "p50": 114.0,
            "p95": 170.0,
        },
    }
    return {
        "dataset": {"human_reviewed": False},
        "protocol": {"reranker_instances_created": 1},
        "model_loading_ms": {"embedding_model": 10.0, "reranker_model": 20.0},
        "cold_query_ms": {
            "hybrid_rrf": 30.0,
            "reranking_only": 150.0,
            "hybrid_reranked": 180.0,
        },
        "warm_query_ms": {"hybrid_rrf": {"mean": 20.0}},
        "hybrid_rrf_quality": quality,
        "variants": [
            {
                "variant_id": "configured_top20_b4_t4",
                "rerank_top_n": 20,
                "batch_size": 4,
                "cpu_threads": 4,
                "quality": quality,
                "latency_ms": latency,
            }
        ],
        "runtime": {
            "total_seconds": 2.0,
            "peak_process_rss_bytes": 100,
            "chunk_count": 436,
            "qdrant_point_count": 436,
        },
    }


def test_profile_variants_are_small_and_isolate_runtime_controls() -> None:
    config = load_config(DEFAULT_CONFIG)

    variants = profiling_variants(config)

    assert [(item.rerank_top_n, item.batch_size, item.cpu_threads) for item in variants] == [
        (20, 4, 4),
        (10, 4, 4),
        (10, 2, 4),
        (10, 4, 2),
    ]


def test_latency_summary_uses_nearest_rank_p95() -> None:
    summary = _latency_summary([10.0, 40.0, 20.0, 30.0])

    assert summary == {"mean": 25.0, "p50": 25.0, "p95": 40.0}


def test_profile_renderers_label_silver_dev_and_runtime_controls() -> None:
    payload = _payload()

    csv_text = render_profile_csv(payload)
    markdown = render_profile_markdown(payload)

    assert "configured_top20_b4_t4,20,4,4" in csv_text
    assert markdown.startswith("# Faz 8.1 reranker profiling — silver dev")
    assert "human_reviewed=false" in markdown
    assert "test split bu kararda kullanılmamıştır" in markdown
    assert "1 instance" in markdown
