"""Trusted BM25, dense, and rank-based hybrid retrieval."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from rank_bm25 import BM25Okapi

from turkish_local_rag.config import BM25Config, ProjectConfig, ResolvedPaths, load_config
from turkish_local_rag.download import SourceDocument, load_manifest


TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TURKISH_LOWER_TRANSLATION = str.maketrans({"I": "ı", "İ": "i"})


class RetrievalError(RuntimeError):
    """Raised when trusted chunk data cannot be loaded or retrieved."""


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    title: str
    page_number: int
    source_page_url: str
    pdf_url: str
    pdf_sha256: str
    source_block_ids: tuple[str, ...]
    text: str
    estimated_tokens: int
    token_count_method: str


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    score: float
    retriever: str
    chunk: ChunkRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "retriever": self.retriever,
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "title": self.chunk.title,
            "page_number": self.chunk.page_number,
            "source_page_url": self.chunk.source_page_url,
            "pdf_url": self.chunk.pdf_url,
            "text": self.chunk.text,
        }


@dataclass(frozen=True, slots=True)
class FusedHit:
    rank: int
    rrf_score: float
    component_ranks: Mapping[str, int]
    chunk: ChunkRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rrf_score": self.rrf_score,
            "component_ranks": dict(self.component_ranks),
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "title": self.chunk.title,
            "page_number": self.chunk.page_number,
            "source_page_url": self.chunk.source_page_url,
            "pdf_url": self.chunk.pdf_url,
            "text": self.chunk.text,
        }


class BM25Retriever:
    """In-memory BM25Okapi retriever over validated chunk records."""

    def __init__(self, chunks: Sequence[ChunkRecord], settings: BM25Config) -> None:
        if not chunks:
            raise RetrievalError("cannot build BM25 index from an empty chunk corpus")
        self._chunks = tuple(chunks)
        self._settings = settings
        tokenized_corpus = [turkish_tokenize(chunk.text) for chunk in self._chunks]
        if not any(tokenized_corpus):
            raise RetrievalError("cannot build BM25 index from chunks without searchable tokens")
        self._index = BM25Okapi(
            tokenized_corpus,
            k1=settings.k1,
            b=settings.b,
            epsilon=settings.epsilon,
        )

    def search(self, question: str, top_k: int | None = None) -> list[RetrievalHit]:
        query_tokens = turkish_tokenize(question)
        if not query_tokens:
            return []
        result_limit = self._settings.top_k if top_k is None else top_k
        if result_limit <= 0:
            raise RetrievalError("top_k must be positive")
        scores = self._index.get_scores(query_tokens)
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if float(score) > self._settings.minimum_score
        ]
        candidates.sort(key=lambda item: (-item[1], self._chunks[item[0]].chunk_id))
        return [
            RetrievalHit(
                rank=rank,
                score=score,
                retriever="bm25",
                chunk=self._chunks[index],
            )
            for rank, (index, score) in enumerate(candidates[:result_limit], start=1)
        ]


def normalize_turkish(text: str) -> str:
    """Normalize Unicode and lowercase with Turkish dotted/dotless-I semantics."""

    normalized = unicodedata.normalize("NFKC", text)
    return normalized.translate(TURKISH_LOWER_TRANSLATION).lower()


def turkish_tokenize(text: str) -> list[str]:
    """Tokenize letters/numbers while preserving Turkish diacritics and suffixes."""

    return TOKEN_PATTERN.findall(normalize_turkish(text))


def load_chunk_corpus(
    chunks_directory: str | Path, sources: Sequence[SourceDocument]
) -> tuple[ChunkRecord, ...]:
    """Load chunk JSONL in manifest order and validate all trusted metadata."""

    directory = Path(chunks_directory)
    chunks: list[ChunkRecord] = []
    seen_chunk_ids: set[str] = set()
    for source in sources:
        chunk_path = directory / f"{source.id}.chunks.jsonl"
        try:
            lines = chunk_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise RetrievalError(
                f"chunk file not found for {source.id}: {chunk_path}"
            ) from exc
        for line_number, line in enumerate(lines, start=1):
            chunk = _parse_chunk(line, chunk_path, line_number, source)
            if chunk.chunk_id in seen_chunk_ids:
                raise RetrievalError(f"duplicate chunk_id: {chunk.chunk_id}")
            seen_chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)
    return tuple(chunks)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    *,
    rank_constant: int,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse rankings using positions only; component score scales are ignored."""

    if rank_constant <= 0:
        raise RetrievalError("rank_constant must be positive")
    if limit is not None and limit <= 0:
        raise RetrievalError("fusion limit must be positive")

    combined: dict[str, dict[str, Any]] = {}
    for retriever_name in sorted(rankings):
        seen_in_ranking: set[str] = set()
        for position, hit in enumerate(rankings[retriever_name], start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            existing = combined.get(chunk_id)
            if existing is None:
                existing = {
                    "chunk": hit.chunk,
                    "score": 0.0,
                    "component_ranks": {},
                }
                combined[chunk_id] = existing
            elif existing["chunk"] != hit.chunk:
                raise RetrievalError(
                    f"conflicting trusted metadata for fused chunk_id: {chunk_id}"
                )
            existing["score"] += 1.0 / (rank_constant + position)
            existing["component_ranks"][retriever_name] = position

    ordered = sorted(
        combined.values(),
        key=lambda item: (
            -item["score"],
            min(item["component_ranks"].values()),
            item["chunk"].chunk_id,
        ),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return [
        FusedHit(
            rank=rank,
            rrf_score=item["score"],
            component_ranks=dict(sorted(item["component_ranks"].items())),
            chunk=item["chunk"],
        )
        for rank, item in enumerate(ordered, start=1)
    ]


def build_bm25_retriever(
    config_path: str | Path, source_ids: Iterable[str] | None = None
) -> BM25Retriever:
    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    sources = load_manifest(paths.source_manifest)
    selected_ids = set(source_ids or ())
    known_ids = {source.id for source in sources}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise RetrievalError(f"unknown source id(s): {', '.join(sorted(unknown_ids))}")
    selected_sources = tuple(
        source for source in sources if not selected_ids or source.id in selected_ids
    )
    chunks = load_chunk_corpus(paths.chunks_directory, selected_sources)
    return BM25Retriever(chunks, config.bm25)


def load_retrieval_corpus(
    config_path: str | Path, source_ids: Iterable[str] | None = None
) -> tuple[ProjectConfig, ResolvedPaths, tuple[SourceDocument, ...], tuple[ChunkRecord, ...]]:
    """Load config and a manifest-filtered trusted chunk corpus once."""

    config = load_config(config_path)
    paths = config.resolve_paths(config_path)
    sources = load_manifest(paths.source_manifest)
    selected_ids = set(source_ids or ())
    known_ids = {source.id for source in sources}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise RetrievalError(f"unknown source id(s): {', '.join(sorted(unknown_ids))}")
    selected_sources = tuple(
        source for source in sources if not selected_ids or source.id in selected_ids
    )
    chunks = load_chunk_corpus(paths.chunks_directory, selected_sources)
    return config, paths, selected_sources, chunks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Turkish retrieval question")
    parser.add_argument(
        "--config", default="config/default.toml", help="TOML config path"
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="search only this manifest id; may be repeated",
    )
    parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid", "hybrid-reranked"),
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        if args.top_k is not None and args.top_k <= 0:
            raise RetrievalError("top_k must be positive")
        config, paths, sources, chunks = load_retrieval_corpus(
            args.config, args.source_id
        )
        if args.top_k is not None:
            output_limit = args.top_k
        elif args.mode == "bm25":
            output_limit = config.bm25.top_k
        elif args.mode == "hybrid-reranked":
            output_limit = config.reranker.top_k
        else:
            output_limit = config.rrf.fused_candidates
        bm25_hits: list[RetrievalHit] = []
        if args.mode in {"bm25", "hybrid", "hybrid-reranked"}:
            sparse_limit = (
                output_limit if args.mode == "bm25" else config.rrf.sparse_candidates
            )
            bm25_hits = BM25Retriever(chunks, config.bm25).search(
                args.question, sparse_limit
            )

        dense_hits: list[RetrievalHit] = []
        if args.mode in {"dense", "hybrid", "hybrid-reranked"}:
            from qdrant_client import QdrantClient

            from turkish_local_rag.dense import (
                SentenceTransformerE5Encoder,
                dense_search,
            )

            if not paths.qdrant_directory.is_dir():
                raise RetrievalError(
                    "dense index directory not found; run turkish_local_rag.index first: "
                    f"{paths.qdrant_directory}"
                )
            dense_limit = (
                output_limit if args.mode == "dense" else config.rrf.dense_candidates
            )
            encoder = SentenceTransformerE5Encoder(
                paths.embedding_model_directory, config.dense
            )
            client = QdrantClient(path=str(paths.qdrant_directory))
            try:
                dense_hits = dense_search(
                    args.question,
                    chunks,
                    encoder,
                    client,
                    config.dense,
                    top_k=dense_limit,
                    document_ids=[source.id for source in sources]
                    if args.source_id
                    else None,
                )
            finally:
                client.close()

        if args.mode == "bm25":
            hits: Sequence[RetrievalHit | FusedHit] = bm25_hits
        elif args.mode == "dense":
            hits = dense_hits
        else:
            fusion_limit = (
                config.reranker.rerank_top_n
                if args.mode == "hybrid-reranked"
                else output_limit
            )
            fused_hits = reciprocal_rank_fusion(
                {"bm25": bm25_hits, "dense": dense_hits},
                rank_constant=config.rrf.rank_constant,
                limit=fusion_limit,
            )
            if args.mode == "hybrid":
                hits = fused_hits
            elif not fused_hits:
                hits = []
            else:
                from turkish_local_rag.rerank import (
                    CrossEncoderReranker,
                    rerank_hits,
                )

                if not paths.reranker_model_directory.is_dir():
                    raise RetrievalError(
                        "reranker model directory not found; run "
                        "turkish_local_rag.index --download-reranker-only first: "
                        f"{paths.reranker_model_directory}"
                    )
                scorer = CrossEncoderReranker(
                    paths.reranker_model_directory, config.reranker
                )
                hits = rerank_hits(
                    args.question, fused_hits, scorer, limit=output_limit
                )
    except (RetrievalError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False, indent=2))
    return 0


def _parse_chunk(
    line: str, path: Path, line_number: int, source: SourceDocument
) -> ChunkRecord:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RetrievalError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RetrievalError(f"chunk record must be an object at {path}:{line_number}")
    required = {
        "schema_version",
        "chunk_id",
        "document_id",
        "title",
        "page_number",
        "source_page_url",
        "pdf_url",
        "pdf_sha256",
        "source_block_ids",
        "text",
        "estimated_tokens",
        "overlap_estimated_tokens",
        "token_count_method",
    }
    missing = required - set(raw)
    if missing:
        raise RetrievalError(
            f"chunk record missing field(s) at {path}:{line_number}: "
            f"{', '.join(sorted(missing))}"
        )
    if raw["schema_version"] != 1:
        raise RetrievalError(f"unsupported chunk schema at {path}:{line_number}")
    for field in ("document_id", "title", "source_page_url", "pdf_url"):
        expected = source.id if field == "document_id" else getattr(source, field)
        if raw[field] != expected:
            raise RetrievalError(
                f"untrusted chunk metadata mismatch at {path}:{line_number}: {field}"
            )
    page_number = _positive_integer(raw, "page_number", path, line_number)
    chunk_id = raw["chunk_id"]
    if not isinstance(chunk_id, str) or not chunk_id.startswith(
        f"{source.id}:p{page_number}:c"
    ):
        raise RetrievalError(f"invalid chunk_id at {path}:{line_number}")
    pdf_sha256 = raw["pdf_sha256"]
    if not isinstance(pdf_sha256, str) or not SHA256_PATTERN.fullmatch(pdf_sha256):
        raise RetrievalError(f"invalid pdf_sha256 at {path}:{line_number}")
    source_block_ids = raw["source_block_ids"]
    if not isinstance(source_block_ids, list) or not all(
        isinstance(block_id, str) and block_id for block_id in source_block_ids
    ):
        raise RetrievalError(f"invalid source_block_ids at {path}:{line_number}")
    text = raw["text"]
    if not isinstance(text, str) or not text.strip():
        raise RetrievalError(f"chunk text must be non-empty at {path}:{line_number}")
    token_count_method = raw["token_count_method"]
    if not isinstance(token_count_method, str) or not token_count_method:
        raise RetrievalError(f"invalid token_count_method at {path}:{line_number}")
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=source.id,
        title=source.title,
        page_number=page_number,
        source_page_url=source.source_page_url,
        pdf_url=source.pdf_url,
        pdf_sha256=pdf_sha256,
        source_block_ids=tuple(source_block_ids),
        text=text,
        estimated_tokens=_positive_integer(
            raw, "estimated_tokens", path, line_number
        ),
        token_count_method=token_count_method,
    )


def _positive_integer(
    table: Mapping[str, Any], key: str, path: Path, line_number: int
) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RetrievalError(f"invalid {key} at {path}:{line_number}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
