from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf
import pytest

from turkish_local_rag.config import ExtractionConfig, ResolvedPaths
from turkish_local_rag.download import SourceDocument
from turkish_local_rag.extract import (
    ExtractionError,
    _normalize_extracted_text,
    _remove_repeated_marginal_content,
    _repair_pdf_glyphs,
    extract_document,
)


SOURCE = SourceDocument(
    id="test-document",
    title="Test Yönetmeliği",
    source_page_url="https://example.test/sources",
    pdf_url="https://example.test/document.pdf",
)
SETTINGS = ExtractionConfig(sort_blocks=True, include_empty_pages=False)


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
    )


def _make_pdf(path: Path, pages: list[list[tuple[float, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    try:
        for text_blocks in pages:
            page = document.new_page()
            for y_position, text in text_blocks:
                page.insert_text((72, y_position), text)
        document.save(path)
    finally:
        document.close()


def _write_metadata(paths: ResolvedPaths) -> str:
    pdf_path = paths.pdf_directory / "test-document.pdf"
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    paths.metadata_directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": SOURCE.id,
        "title": SOURCE.title,
        "source_page_url": SOURCE.source_page_url,
        "pdf_url": SOURCE.pdf_url,
        "final_url": SOURCE.pdf_url,
        "downloaded_at_utc": "2026-09-01T12:30:00Z",
        "size_bytes": pdf_path.stat().st_size,
        "sha256": digest,
        "content_type": "application/pdf",
        "local_filename": pdf_path.name,
    }
    (paths.metadata_directory / "test-document.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return digest


def test_extraction_preserves_pages_order_and_trusted_metadata(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    pdf_path = paths.pdf_directory / "test-document.pdf"
    _make_pdf(
        pdf_path,
        [
            [(200, "Ikinci blok"), (72, "MADDE 1 - Birinci blok")],
            [(72, "MADDE 2 - Ikinci sayfa")],
        ],
    )
    expected_hash = _write_metadata(paths)

    result = extract_document(SOURCE, paths, SETTINGS)

    records = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.total_pdf_pages == 2
    assert result.extracted_pages == 2
    assert [record["page_number"] for record in records] == [1, 2]
    assert records[0]["text"].index("Birinci") < records[0]["text"].index("Ikinci")
    assert "Ikinci sayfa" not in records[0]["text"]
    assert records[0]["title"] == SOURCE.title
    assert records[0]["source_page_url"] == SOURCE.source_page_url
    assert records[0]["pdf_sha256"] == expected_hash
    assert records[0]["blocks"][0]["block_id"] == "test-document:p1:b0"
    assert records[0]["raw_text"] == "MADDE 1 - Birinci blok\n\nIkinci blok"
    assert records[0]["blocks"][0]["raw_text"] == "MADDE 1 - Birinci blok"
    assert not list(paths.extracted_pages_directory.glob("*.tmp"))


def test_empty_pages_are_skipped_without_changing_page_numbers(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    pdf_path = paths.pdf_directory / "test-document.pdf"
    _make_pdf(pdf_path, [[], [(72, "Second physical page")]])
    _write_metadata(paths)

    result = extract_document(SOURCE, paths, SETTINGS)
    records = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.total_pdf_pages == 2
    assert result.extracted_pages == 1
    assert records[0]["page_number"] == 2


def test_hash_mismatch_blocks_extraction(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    pdf_path = paths.pdf_directory / "test-document.pdf"
    _make_pdf(pdf_path, [[(72, "Original")]])
    _write_metadata(paths)
    changed = bytearray(pdf_path.read_bytes())
    changed[-1] ^= 1
    pdf_path.write_bytes(changed)

    with pytest.raises(ExtractionError, match="PDF hash differs"):
        extract_document(SOURCE, paths, SETTINGS)

    assert not paths.extracted_pages_directory.exists()


def test_missing_download_metadata_blocks_extraction(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _make_pdf(paths.pdf_directory / "test-document.pdf", [[(72, "Content")]])

    with pytest.raises(ExtractionError, match="download metadata not found"):
        extract_document(SOURCE, paths, SETTINGS)

    assert not paths.extracted_pages_directory.exists()


def test_evidenced_pdf_glyph_mappings_are_repaired() -> None:
    assert _repair_pdf_glyphs("Ün\u0d74vers\u0d74tes\u0d74 resm\u0d88gazete") == (
        "Üniversitesi resmigazete"
    )


def test_only_evidenced_misdecoded_initial_ilgili_is_repaired() -> None:
    raw = "ğ) ılgili mevzuat; ilgili birim, ılık hava ve kırılgılı yapı."

    assert _normalize_extracted_text(raw) == (
        "ğ) İlgili mevzuat; ilgili birim, ılık hava ve kırılgılı yapı."
    )


def test_repeated_marginal_lines_and_page_counters_are_removed() -> None:
    pages = [
        (
            page_number,
            1000.0,
            [
                {
                    "block_id": f"doc:p{page_number}:b0",
                    "bbox": [20.0, 10.0, 300.0, 30.0],
                    "text": "Repeated header",
                },
                {
                    "block_id": f"doc:p{page_number}:b1",
                    "bbox": [20.0, 200.0, 300.0, 500.0],
                    "text": f"Body {page_number}",
                },
                {
                    "block_id": f"doc:p{page_number}:b2",
                    "bbox": [20.0, 950.0, 300.0, 980.0],
                    "text": f"Repeated footer\n{page_number}/5",
                },
            ],
        )
        for page_number in range(1, 6)
    ]

    _remove_repeated_marginal_content(pages, total_pages=5)

    assert [[block["text"] for block in blocks] for _, _, blocks in pages] == [
        [f"Body {page_number}"] for page_number in range(1, 6)
    ]
    assert [blocks[0]["block_id"] for _, _, blocks in pages] == [
        f"doc:p{page_number}:b0" for page_number in range(1, 6)
    ]
