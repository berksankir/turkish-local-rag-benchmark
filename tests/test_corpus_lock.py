from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from turkish_local_rag.config import load_config
from turkish_local_rag.corpus_lock import (
    CorpusLockError,
    load_corpus_lock,
    verify_corpus_lock,
    write_corpus_lock,
)


def _local_corpus(tmp_path: Path):
    config = load_config("config/default.toml")
    paths = replace(
        config.resolve_paths("config/default.toml"),
        project_root=tmp_path,
        source_manifest=tmp_path / "data" / "manifest.json",
        corpus_lock=tmp_path / "data" / "corpus.lock.json",
        pdf_directory=tmp_path / "data" / "pdfs",
        metadata_directory=tmp_path / "data" / "downloads",
    )
    paths.source_manifest.parent.mkdir(parents=True)
    paths.pdf_directory.mkdir(parents=True)
    paths.metadata_directory.mkdir(parents=True)
    source = {
        "id": "doc-one",
        "title": "Belge",
        "source_page_url": "https://example.test/source",
        "pdf_url": "https://example.test/document.pdf",
    }
    paths.source_manifest.write_text(
        json.dumps({"schema_version": 1, "documents": [source]}), encoding="utf-8"
    )
    pdf = b"%PDF-1.4\nfixture\n%%EOF"
    pdf_path = paths.pdf_directory / "doc-one.pdf"
    pdf_path.write_bytes(pdf)
    metadata = {
        **source,
        "final_url": source["pdf_url"],
        "downloaded_at_utc": "2026-09-01T12:30:00Z",
        "size_bytes": len(pdf),
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "content_type": "application/pdf",
        "local_filename": "doc-one.pdf",
    }
    (paths.metadata_directory / "doc-one.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return paths, pdf_path


def test_corpus_lock_is_built_from_and_verifies_real_local_bytes(tmp_path: Path) -> None:
    paths, _ = _local_corpus(tmp_path)

    assert write_corpus_lock(paths) == "written"
    assert write_corpus_lock(paths) == "unchanged"
    records = verify_corpus_lock(paths)

    assert len(records) == 1
    assert records[0].document_id == "doc-one"
    assert records[0].downloaded_at_utc == "2026-09-01T12:30:00Z"
    assert load_corpus_lock(paths.corpus_lock) == records


def test_corpus_lock_rejects_changed_pdf_without_overwriting_lock(tmp_path: Path) -> None:
    paths, pdf_path = _local_corpus(tmp_path)
    write_corpus_lock(paths)
    original_lock = paths.corpus_lock.read_bytes()
    pdf_path.write_bytes(b"%PDF-1.4\nchanged\n%%EOF")

    with pytest.raises(CorpusLockError, match="differs from lock"):
        verify_corpus_lock(paths)

    assert paths.corpus_lock.read_bytes() == original_lock


def test_committed_corpus_lock_matches_nine_local_verified_pdfs() -> None:
    config = load_config("config/default.toml")
    paths = config.resolve_paths("config/default.toml")

    records = verify_corpus_lock(paths)

    assert len(records) == 9
    assert all(record.size_bytes > 0 for record in records)
    assert all(len(record.sha256) == 64 for record in records)
