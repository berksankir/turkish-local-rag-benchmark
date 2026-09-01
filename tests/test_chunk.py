from __future__ import annotations

import json
from pathlib import Path

import pytest

from turkish_local_rag.chunk import ChunkingError, chunk_document, estimate_tokens
from turkish_local_rag.config import ChunkingConfig, ResolvedPaths
from turkish_local_rag.download import SourceDocument


SOURCE = SourceDocument(
    id="test-document",
    title="Test Yönetmeliği",
    source_page_url="https://example.test/sources",
    pdf_url="https://example.test/document.pdf",
)
PDF_SHA256 = "a" * 64


def _paths(tmp_path: Path) -> ResolvedPaths:
    return ResolvedPaths(
        project_root=tmp_path,
        source_manifest=tmp_path / "manifest.json",
        pdf_directory=tmp_path / "pdfs",
        metadata_directory=tmp_path / "metadata",
        extracted_pages_directory=tmp_path / "extracted",
        chunks_directory=tmp_path / "chunks",
        embedding_model_directory=tmp_path / "model",
        reranker_model_directory=tmp_path / "reranker",
        qdrant_directory=tmp_path / "qdrant",
        evaluation_candidates=tmp_path / "evaluation" / "candidates.jsonl",
        evaluation_review=tmp_path / "evaluation" / "review.csv",
        evaluation_silver=tmp_path / "evaluation" / "silver.jsonl",
        evaluation_silver_audit=tmp_path / "evaluation" / "silver_audit.csv",
        evaluation_gold=tmp_path / "evaluation" / "gold.jsonl",
        evaluation_results_directory=tmp_path / "evaluation" / "results",
    )


def _settings(
    *, target: int = 20, maximum: int = 25, overlap: int = 3
) -> ChunkingConfig:
    return ChunkingConfig(
        target_model_tokens=target,
        maximum_model_tokens=maximum,
        overlap_model_tokens=overlap,
        estimated_characters_per_token=3,
        respect_page_boundaries=True,
        preserve_article_boundaries=True,
    )


def _page(page_number: int, block_texts: list[str]) -> dict[str, object]:
    blocks = [
        {
            "block_id": f"test-document:p{page_number}:b{index}",
            "bbox": [0.0, float(index), 100.0, float(index + 1)],
            "text": text,
        }
        for index, text in enumerate(block_texts)
    ]
    return {
        "schema_version": 1,
        "document_id": SOURCE.id,
        "title": SOURCE.title,
        "page_number": page_number,
        "source_page_url": SOURCE.source_page_url,
        "pdf_url": SOURCE.pdf_url,
        "pdf_sha256": PDF_SHA256,
        "text": "\n\n".join(block_texts),
        "blocks": blocks,
    }


def _write_pages(paths: ResolvedPaths, pages: list[dict[str, object]]) -> None:
    paths.extracted_pages_directory.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(page, ensure_ascii=False) + "\n" for page in pages
    )
    (paths.extracted_pages_directory / "test-document.pages.jsonl").write_text(
        content, encoding="utf-8"
    )


def _read_chunks(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_chunks_never_cross_pages_or_article_boundaries(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_pages(
        paths,
        [
            _page(
                1,
                [
                    "BİRİNCİ BÖLÜM\nAmaç\nMADDE 1 - Birinci hüküm.\n(1) Devam paragrafı.\nMADDE 2 - İkinci hüküm."
                ],
            ),
            _page(2, ["MADDE 3 - Yalnızca ikinci fiziksel sayfa."]),
        ],
    )

    result = chunk_document(SOURCE, paths, _settings())
    chunks = _read_chunks(result.output_path)

    assert {chunk["page_number"] for chunk in chunks} == {1, 2}
    assert all(
        not ("MADDE 1" in chunk["text"] and "MADDE 2" in chunk["text"])
        for chunk in chunks
    )
    assert all(
        not ("Birinci hüküm" in chunk["text"] and "ikinci fiziksel" in chunk["text"])
        for chunk in chunks
    )
    assert all(chunk["source_page_url"] == SOURCE.source_page_url for chunk in chunks)
    assert all(chunk["pdf_sha256"] == PDF_SHA256 for chunk in chunks)
    assert not list(paths.chunks_directory.glob("*.tmp"))


def test_long_text_is_split_with_overlap_inside_hard_limit(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    text = " ".join(f"kelime{index}" for index in range(40))
    _write_pages(paths, [_page(1, [text])])
    settings = _settings(target=8, maximum=12, overlap=4)

    result = chunk_document(SOURCE, paths, settings)
    chunks = _read_chunks(result.output_path)

    assert len(chunks) > 1
    assert all(chunk["estimated_tokens"] <= 12 for chunk in chunks)
    assert any(chunk["overlap_estimated_tokens"] > 0 for chunk in chunks[1:])
    assert all(chunk["page_number"] == 1 for chunk in chunks)


def test_turkish_text_is_not_case_folded_or_ascii_normalized(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    original = "MADDE 1 - İşleyiş, ısı ve ölçüm ilkeleri düzenlenmiştir."
    _write_pages(paths, [_page(1, [original])])

    result = chunk_document(SOURCE, paths, _settings())
    chunks = _read_chunks(result.output_path)

    assert original in "\n".join(str(chunk["text"]) for chunk in chunks)
    assert estimate_tokens(original, 3) == estimate_tokens(original, 3)
    assert chunks[0]["token_count_method"].startswith("max(unicode_lexemes")


def test_untrusted_page_metadata_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    page = _page(1, ["MADDE 1 - İçerik."])
    page["source_page_url"] = "https://untrusted.example.test/"
    _write_pages(paths, [page])

    with pytest.raises(ChunkingError, match="untrusted page metadata mismatch"):
        chunk_document(SOURCE, paths, _settings())

    assert not paths.chunks_directory.exists()


def test_article_heading_is_attached_to_following_article(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_pages(
        paths,
        [
            _page(
                1,
                [
                    "Amaç\nMADDE 1 - İlk hüküm.\n"
                    "Tanımlar\nMADDE 2 - İkinci hüküm."
                ],
            )
        ],
    )

    result = chunk_document(SOURCE, paths, _settings(target=100, maximum=120))
    chunks = _read_chunks(result.output_path)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "Amaç\nMADDE 1 - İlk hüküm."
    assert chunks[1]["text"] == "Tanımlar\nMADDE 2 - İkinci hüküm."


def test_cross_block_section_heading_is_attached_and_page_footer_is_removed(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_pages(
        paths,
        [
            _page(
                1,
                [
                    "İKİNCİ BÖLÜM\nEğitim Esasları\nAkademik yıl",
                    "MADDE 5 - Akademik yıl iki dönemdir.",
                    "Test Yönetmeliği   Sayfa 1 / 2",
                ],
            )
        ],
    )

    result = chunk_document(SOURCE, paths, _settings(target=100, maximum=120))
    chunks = _read_chunks(result.output_path)

    assert len(chunks) == 1
    assert chunks[0]["text"] == (
        "İKİNCİ BÖLÜM\nEğitim Esasları\nAkademik yıl\n"
        "MADDE 5 - Akademik yıl iki dönemdir."
    )
    assert chunks[0]["source_block_ids"] == [
        "test-document:p1:b0",
        "test-document:p1:b1",
    ]


def test_publication_masthead_only_chunk_is_removed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_pages(
        paths,
        [
            _page(
                1,
                [
                    "21 Ağustos 2023\nResmî Gazete No: 32286\nYÖNETMELİK",
                    "Amaç\nMADDE 1 - Yönetmeliğin amacı açıklanır.",
                ],
            )
        ],
    )

    result = chunk_document(SOURCE, paths, _settings(target=100, maximum=120))
    chunks = _read_chunks(result.output_path)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Amaç\nMADDE 1 - Yönetmeliğin amacı açıklanır."
    assert chunks[0]["chunk_id"] == "test-document:p1:c0"
