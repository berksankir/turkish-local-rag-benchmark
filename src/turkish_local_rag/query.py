"""Run a grounded local RAG query with trusted citations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

from turkish_local_rag.config import ProjectConfig, load_config
from turkish_local_rag.generation import (
    GenerationError,
    GroundedRAGService,
    LlamaCppServerGenerator,
    RetrievalExecution,
)
from turkish_local_rag.retrieve import (
    BM25Retriever,
    load_retrieval_corpus,
    reciprocal_rank_fusion,
)


class LocalRetrievalRuntime:
    """Reusable BM25/E5/Qdrant runtime with an optional reusable reranker."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config, self.paths, self.sources, self.chunks = load_retrieval_corpus(
            self.config_path
        )
        from qdrant_client import QdrantClient

        from turkish_local_rag.dense import SentenceTransformerE5Encoder

        self.bm25 = BM25Retriever(self.chunks, self.config.bm25)
        self.encoder = SentenceTransformerE5Encoder(
            self.paths.embedding_model_directory, self.config.dense
        )
        self.client = QdrantClient(path=str(self.paths.qdrant_directory))
        if not self.client.collection_exists(self.config.dense.collection_name):
            self.client.close()
            raise GenerationError(
                f"Qdrant collection not found: {self.config.dense.collection_name}"
            )
        self._reranker: Any | None = None

    def __call__(self, question: str, pipeline: str) -> RetrievalExecution:
        from turkish_local_rag.dense import dense_search

        start = perf_counter()
        sparse = self.bm25.search(question, self.config.rrf.sparse_candidates)
        dense = dense_search(
            question,
            self.chunks,
            self.encoder,
            self.client,
            self.config.dense,
            top_k=self.config.rrf.dense_candidates,
        )
        fusion_limit = (
            self.config.reranker.rerank_top_n
            if pipeline == "hybrid_reranked"
            else self.config.rrf.fused_candidates
        )
        fused = reciprocal_rank_fusion(
            {"bm25": sparse, "dense": dense},
            rank_constant=self.config.rrf.rank_constant,
            limit=fusion_limit,
        )
        retrieval_ms = (perf_counter() - start) * 1000.0
        reranking_ms = 0.0
        reranker_metadata = None
        if pipeline == "hybrid_reranked" and fused:
            from turkish_local_rag.rerank import CrossEncoderReranker, rerank_hits

            if self._reranker is None:
                self._reranker = CrossEncoderReranker(
                    self.paths.reranker_model_directory, self.config.reranker
                )
            rerank_start = perf_counter()
            hits = tuple(
                rerank_hits(
                    question,
                    fused,
                    self._reranker,
                    limit=self.config.reranker.top_k,
                )
            )
            reranking_ms = (perf_counter() - rerank_start) * 1000.0
            reranker_metadata = {
                "model_id": self.config.reranker.model_id,
                "revision": self.config.reranker.model_revision,
                "sha256": self.config.reranker.model_sha256,
                "rerank_top_n": self.config.reranker.rerank_top_n,
                "batch_size": self.config.reranker.batch_size,
                "cpu_threads": self.config.reranker.cpu_threads,
            }
        else:
            hits = tuple(fused)
        return RetrievalExecution(
            hits=hits,
            retrieval_latency_ms=retrieval_ms,
            reranking_latency_ms=reranking_ms,
            embedding_metadata={
                "model_id": self.config.dense.model_id,
                "revision": self.config.dense.model_revision,
                "sha256": self.config.dense.model_sha256,
            },
            reranker_metadata=reranker_metadata,
        )

    def close(self) -> None:
        self.client.close()


def build_service(
    config_path: str | Path,
) -> tuple[GroundedRAGService, LocalRetrievalRuntime, LlamaCppServerGenerator]:
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    retriever = LocalRetrievalRuntime(config_path)
    generator = LlamaCppServerGenerator(
        paths.generator_model_file,
        paths.llama_server_executable,
        config.generation,
    )
    service = GroundedRAGService(
        retriever, generator, config.generation, config.evidence
    )
    return service, retriever, generator


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--pipeline",
        choices=("hybrid_rrf", "hybrid_reranked"),
        default="hybrid_rrf",
    )
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args(argv)

    retriever: LocalRetrievalRuntime | None = None
    generator: LlamaCppServerGenerator | None = None
    try:
        service, retriever, generator = build_service(args.config)
        response = service.answer(args.question, args.pipeline)
    except (GenerationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if generator is not None:
            generator.close()
        if retriever is not None:
            retriever.close()
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
