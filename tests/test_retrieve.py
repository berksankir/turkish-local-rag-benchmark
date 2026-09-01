from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from turkish_local_rag.config import BM25Config
from turkish_local_rag.download import SourceDocument
from turkish_local_rag.retrieve import (
    BM25Retriever,
    ChunkRecord,
    RetrievalError,
    RetrievalHit,
    load_chunk_corpus,
    normalize_turkish,
    reciprocal_rank_fusion,
    turkish_tokenize,
)


SETTINGS = BM25Config(k1=1.5, b=0.75, epsilon=0.25, minimum_score=0.0, top_k=3)


def _chunk(index: int, text: str, *, page: int = 1) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"doc:p{page}:c{index}",
        document_id="doc",
        title="Belge",
        page_number=page,
        source_page_url="https://example.test/source",
        pdf_url="https://example.test/doc.pdf",
        pdf_sha256="a" * 64,
        source_block_ids=(f"doc:p{page}:b{index}",),
        text=text,
        estimated_tokens=10,
        token_count_method="test",
    )


def _hit(chunk: ChunkRecord, rank: int, score: float, retriever: str) -> RetrievalHit:
    return RetrievalHit(rank=rank, score=score, retriever=retriever, chunk=chunk)


def test_turkish_normalization_handles_dotted_and_dotless_i() -> None:
    assert normalize_turkish("İSTANBUL IĞDIR İzmir ısı") == "istanbul ığdır izmir ısı"
    assert turkish_tokenize("Türkiye'nin İHALE’Sİ") == ["türkiye", "nin", "ihale", "si"]
    assert turkish_tokenize("I\u0307stanbul") == ["istanbul"]


def test_bm25_returns_relevant_chunk_with_trusted_citation_metadata() -> None:
    chunks = [
        _chunk(0, "Öğrenci burs başvurusunu belirtilen tarihte yapar.", page=3),
        _chunk(1, "Kütüphane çalışma saatleri ilan edilir.", page=4),
        _chunk(2, "İhale komisyonu üyeleri belirlenir.", page=5),
        _chunk(3, "Lisansüstü tez savunması jüri önünde yapılır.", page=6),
    ]
    retriever = BM25Retriever(chunks, SETTINGS)

    hits = retriever.search("BURS başvurusu", top_k=2)

    assert hits[0].chunk.chunk_id == "doc:p3:c0"
    assert hits[0].chunk.page_number == 3
    assert hits[0].chunk.source_page_url == "https://example.test/source"
    assert hits[0].retriever == "bm25"


def test_bm25_empty_or_punctuation_query_returns_no_hits() -> None:
    retriever = BM25Retriever(
        [_chunk(0, "bir"), _chunk(1, "iki"), _chunk(2, "üç")], SETTINGS
    )

    assert retriever.search("... -- !!!") == []


def test_rrf_uses_rank_positions_not_raw_score_scales() -> None:
    first = _chunk(0, "birinci")
    second = _chunk(1, "ikinci")
    third = _chunk(2, "üçüncü")
    rankings = {
        "bm25": [
            _hit(first, 1, 5000.0, "bm25"),
            _hit(second, 2, 1.0, "bm25"),
        ],
        "dense": [
            _hit(second, 1, 0.01, "dense"),
            _hit(third, 2, 0.99, "dense"),
        ],
    }

    fused = reciprocal_rank_fusion(rankings, rank_constant=60)

    assert [hit.chunk.chunk_id for hit in fused] == [second.chunk_id, first.chunk_id, third.chunk_id]
    assert fused[0].component_ranks == {"bm25": 2, "dense": 1}
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_rejects_conflicting_metadata_for_same_chunk_id() -> None:
    trusted = _chunk(0, "trusted")
    conflicting = replace(trusted, text="conflicting")

    with pytest.raises(RetrievalError, match="conflicting trusted metadata"):
        reciprocal_rank_fusion(
            {
                "bm25": [_hit(trusted, 1, 1.0, "bm25")],
                "dense": [_hit(conflicting, 1, 1.0, "dense")],
            },
            rank_constant=60,
        )


def test_chunk_loader_rejects_manifest_metadata_mismatch(tmp_path: Path) -> None:
    source = SourceDocument(
        id="doc",
        title="Belge",
        source_page_url="https://example.test/source",
        pdf_url="https://example.test/doc.pdf",
    )
    chunks_directory = tmp_path / "chunks"
    chunks_directory.mkdir()
    payload = {
        "schema_version": 1,
        "chunk_id": "doc:p1:c0",
        "document_id": "doc",
        "title": "Sahte Başlık",
        "page_number": 1,
        "source_page_url": source.source_page_url,
        "pdf_url": source.pdf_url,
        "pdf_sha256": "a" * 64,
        "source_block_ids": ["doc:p1:b0"],
        "text": "İçerik",
        "estimated_tokens": 3,
        "overlap_estimated_tokens": 0,
        "token_count_method": "test",
    }
    (chunks_directory / "doc.chunks.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with pytest.raises(RetrievalError, match="untrusted chunk metadata mismatch"):
        load_chunk_corpus(chunks_directory, [source])
