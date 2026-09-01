from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import pytest
from qdrant_client import QdrantClient

from turkish_local_rag.config import DenseConfig
from turkish_local_rag.dense import (
    MODEL_REQUIRED_FILES,
    DenseRetrievalError,
    SentenceTransformerE5Encoder,
    build_dense_index,
    dense_point_id,
    dense_search,
)
from turkish_local_rag.retrieve import ChunkRecord


class FakeSentenceTransformer:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.max_seq_length = 0

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: Any) -> NDArray[np.float32]:
        self.calls.append((texts, kwargs))
        vectors = []
        for text in texts:
            if "burs" in text.casefold():
                vectors.append([1.0, 0.0, 0.0])
            elif "ihale" in text.casefold():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeEncoder:
    dimension = 3

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> NDArray[np.float32]:
        return np.asarray(self._vector(text), dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "burs" in lowered:
            return [1.0, 0.0, 0.0]
        if "ihale" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _settings(weights_sha256: str) -> DenseConfig:
    return DenseConfig(
        model_id="test/model",
        model_revision="a" * 40,
        model_sha256=weights_sha256,
        vector_size=3,
        max_sequence_length=32,
        batch_size=2,
        query_prefix="query: ",
        passage_prefix="passage: ",
        normalize_embeddings=True,
        collection_name="test_dense",
        minimum_score=0.0,
    )


def _model_directory(tmp_path: Path) -> tuple[Path, str]:
    model_directory = tmp_path / "model"
    for relative_path in MODEL_REQUIRED_FILES:
        path = model_directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights" if relative_path == "model.safetensors" else b"{}")
    weights_hash = hashlib.sha256(b"weights").hexdigest()
    size_bytes = sum(
        (model_directory / relative_path).stat().st_size
        for relative_path in MODEL_REQUIRED_FILES
    )
    (model_directory / "download_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "revision": "a" * 40,
                "weights_sha256": weights_hash,
                "size_bytes": size_bytes,
                "downloaded_at_utc": "2026-09-01T12:00:00Z",
                "files": list(MODEL_REQUIRED_FILES),
            }
        ),
        encoding="utf-8",
    )
    return model_directory, weights_hash


def _chunk(index: int, text: str, page: int) -> ChunkRecord:
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
        estimated_tokens=8,
        token_count_method="test",
    )


def test_e5_encoder_applies_required_prefixes_and_cpu_options(tmp_path: Path) -> None:
    model_directory, weights_hash = _model_directory(tmp_path)
    fake_model = FakeSentenceTransformer()
    factory_calls: list[tuple[str, dict[str, Any]]] = []

    def factory(path: str, **kwargs: Any) -> FakeSentenceTransformer:
        factory_calls.append((path, kwargs))
        return fake_model

    encoder = SentenceTransformerE5Encoder(
        model_directory, _settings(weights_hash), model_factory=factory
    )
    encoder.encode_query("burs koşulları")
    encoder.encode_passages(["ihale usulü"])

    assert factory_calls[0][1] == {"device": "cpu", "local_files_only": True}
    assert fake_model.calls[0][0] == ["query: burs koşulları"]
    assert fake_model.calls[1][0] == ["passage: ihale usulü"]
    assert all(call[1]["normalize_embeddings"] is True for call in fake_model.calls)
    assert all(call[1]["device"] == "cpu" for call in fake_model.calls)
    assert fake_model.max_seq_length == 32


def test_e5_encoder_rejects_manifest_revision_before_model_load(tmp_path: Path) -> None:
    model_directory, weights_hash = _model_directory(tmp_path)
    manifest_path = model_directory / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "wrong-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    factory_called = False

    def factory(path: str, **kwargs: Any) -> FakeSentenceTransformer:
        nonlocal factory_called
        factory_called = True
        return FakeSentenceTransformer()

    with pytest.raises(DenseRetrievalError, match="manifest mismatch for revision"):
        SentenceTransformerE5Encoder(
            model_directory, _settings(weights_hash), model_factory=factory
        )

    assert factory_called is False


def test_qdrant_local_index_and_dense_search_preserve_metadata() -> None:
    settings = _settings("a" * 64)
    chunks = [_chunk(0, "Burs başvurusu", 3), _chunk(1, "İhale komisyonu", 5)]
    client = QdrantClient(":memory:")
    try:
        result = build_dense_index(chunks, FakeEncoder(), client, settings)
        hits = dense_search(
            "burs şartları", chunks, FakeEncoder(), client, settings, top_k=2
        )

        assert result.indexed_chunks == 2
        assert hits[0].chunk.chunk_id == "doc:p3:c0"
        assert hits[0].chunk.page_number == 3
        assert hits[0].chunk.source_page_url == "https://example.test/source"
        assert hits[0].retriever == "dense"
        assert client.count(settings.collection_name, exact=True).count == 2
        stored = client.retrieve(
            collection_name=settings.collection_name,
            ids=[dense_point_id(chunks[0].chunk_id)],
            with_payload=True,
            with_vectors=False,
        )[0]
        assert stored.payload == {
            "schema_version": 1,
            "chunk_id": chunks[0].chunk_id,
            "document_id": chunks[0].document_id,
            "title": chunks[0].title,
            "page_number": chunks[0].page_number,
            "source_page_url": chunks[0].source_page_url,
            "pdf_url": chunks[0].pdf_url,
            "pdf_sha256": chunks[0].pdf_sha256,
            "source_block_ids": list(chunks[0].source_block_ids),
            "text": chunks[0].text,
            "estimated_tokens": chunks[0].estimated_tokens,
            "token_count_method": chunks[0].token_count_method,
            "embedding_model_id": settings.model_id,
            "embedding_model_revision": settings.model_revision,
            "embedding_prefix": settings.passage_prefix,
        }
    finally:
        client.close()


def test_qdrant_disk_index_survives_client_restart(tmp_path: Path) -> None:
    settings = _settings("a" * 64)
    chunks = [_chunk(0, "Burs başvurusu", 3), _chunk(1, "İhale komisyonu", 5)]
    database_path = tmp_path / "qdrant"
    client = QdrantClient(path=str(database_path))
    build_dense_index(chunks, FakeEncoder(), client, settings)
    client.close()

    reopened_client = QdrantClient(path=str(database_path))
    try:
        hits = dense_search(
            "ihale usulü",
            chunks,
            FakeEncoder(),
            reopened_client,
            settings,
            top_k=1,
        )
        assert hits[0].chunk.chunk_id == "doc:p5:c1"
    finally:
        reopened_client.close()


def test_existing_collection_requires_explicit_rebuild() -> None:
    settings = _settings("a" * 64)
    chunks = [_chunk(0, "Burs", 1)]
    client = QdrantClient(":memory:")
    try:
        build_dense_index(chunks, FakeEncoder(), client, settings)
        with pytest.raises(DenseRetrievalError, match="use --rebuild explicitly"):
            build_dense_index(chunks, FakeEncoder(), client, settings)
        rebuilt = build_dense_index(
            chunks, FakeEncoder(), client, settings, rebuild=True
        )
        assert rebuilt.indexed_chunks == 1
    finally:
        client.close()


def test_dense_search_rejects_tampered_qdrant_payload() -> None:
    settings = _settings("a" * 64)
    chunk = _chunk(0, "Burs", 1)
    client = QdrantClient(":memory:")
    try:
        build_dense_index([chunk], FakeEncoder(), client, settings)
        client.set_payload(
            collection_name=settings.collection_name,
            payload={"title": "Sahte Başlık"},
            points=[dense_point_id(chunk.chunk_id)],
            wait=True,
        )
        with pytest.raises(DenseRetrievalError, match="untrusted Qdrant payload"):
            dense_search("burs", [chunk], FakeEncoder(), client, settings, top_k=1)
    finally:
        client.close()
