"""Local E5 embeddings and persistent Qdrant dense retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from huggingface_hub import snapshot_download
import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from turkish_local_rag.config import DenseConfig
from turkish_local_rag.retrieve import ChunkRecord, RetrievalError, RetrievalHit


MODEL_REQUIRED_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MODEL_ALLOW_PATTERNS = (*MODEL_REQUIRED_FILES, "README.md")


class DenseRetrievalError(RetrievalError):
    """Raised when the local embedding model or dense index is invalid."""


class EmbeddingEncoder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def encode_query(self, text: str) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class ModelDownloadResult:
    status: str
    model_directory: Path
    model_id: str
    revision: str
    weights_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DenseIndexResult:
    collection_name: str
    indexed_chunks: int
    vector_size: int


class SentenceTransformerE5Encoder:
    """CPU-only E5 encoder that refuses implicit network access."""

    def __init__(
        self,
        model_directory: str | Path,
        settings: DenseConfig,
        *,
        model_factory: Callable[..., Any] = SentenceTransformer,
    ) -> None:
        verify_embedding_model(model_directory, settings)
        self._settings = settings
        self._model = model_factory(
            str(Path(model_directory)), device="cpu", local_files_only=True
        )
        dimension = self._model.get_embedding_dimension()
        if dimension != settings.vector_size:
            raise DenseRetrievalError(
                f"embedding dimension mismatch: expected {settings.vector_size}, got {dimension}"
            )
        self._model.max_seq_length = settings.max_sequence_length

    @property
    def dimension(self) -> int:
        return self._settings.vector_size

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        prefixed = [f"{self._settings.passage_prefix}{text}" for text in texts]
        return self._encode(prefixed)

    def encode_query(self, text: str) -> NDArray[np.float32]:
        return self._encode([f"{self._settings.query_prefix}{text}"])[0]

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._settings.normalize_embeddings,
            device="cpu",
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.shape != (len(texts), self.dimension):
            raise DenseRetrievalError(
                f"unexpected embedding shape: expected {(len(texts), self.dimension)}, "
                f"got {array.shape}"
            )
        return array


def download_embedding_model(
    model_directory: str | Path, settings: DenseConfig
) -> ModelDownloadResult:
    """Download only safetensors/tokenizer files from a pinned model revision."""

    target = Path(model_directory)
    weights_path = target / "model.safetensors"
    if weights_path.exists():
        actual_hash = _sha256_file(weights_path)
        if actual_hash != settings.model_sha256:
            raise DenseRetrievalError(
                "existing embedding weights hash mismatch; file was not overwritten: "
                f"expected={settings.model_sha256}, actual={actual_hash}"
            )
    status = (
        "unchanged"
        if _model_is_complete(target) and (target / "download_manifest.json").is_file()
        else "downloaded"
    )
    snapshot_download(
        repo_id=settings.model_id,
        revision=settings.model_revision,
        local_dir=target,
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
    )
    weights_sha256 = _verify_embedding_files(target, settings)
    size_bytes = sum(
        path.stat().st_size
        for relative_path in MODEL_REQUIRED_FILES
        if (path := target / relative_path).is_file()
    )
    metadata = {
        "model_id": settings.model_id,
        "revision": settings.model_revision,
        "weights_sha256": weights_sha256,
        "size_bytes": size_bytes,
        "downloaded_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "files": list(MODEL_REQUIRED_FILES),
    }
    _write_json_atomic(target / "download_manifest.json", metadata)
    verify_embedding_model(target, settings)
    return ModelDownloadResult(
        status=status,
        model_directory=target,
        model_id=settings.model_id,
        revision=settings.model_revision,
        weights_sha256=weights_sha256,
        size_bytes=size_bytes,
    )


def verify_embedding_model(
    model_directory: str | Path, settings: DenseConfig
) -> str:
    target = Path(model_directory)
    weights_sha256 = _verify_embedding_files(target, settings)
    _verify_download_manifest(target, settings, weights_sha256)
    return weights_sha256


def _verify_embedding_files(model_directory: Path, settings: DenseConfig) -> str:
    target = model_directory
    missing = [name for name in MODEL_REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        raise DenseRetrievalError(
            f"embedding model is incomplete at {target}; missing: {', '.join(missing)}"
        )
    weights_sha256 = _sha256_file(target / "model.safetensors")
    if weights_sha256 != settings.model_sha256:
        raise DenseRetrievalError(
            "embedding weights SHA-256 mismatch: "
            f"expected={settings.model_sha256}, actual={weights_sha256}"
        )
    return weights_sha256


def _verify_download_manifest(
    model_directory: Path, settings: DenseConfig, weights_sha256: str
) -> None:
    manifest_path = model_directory / "download_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DenseRetrievalError(
            f"embedding download manifest not found: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DenseRetrievalError(
            f"invalid embedding download manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise DenseRetrievalError("embedding download manifest must be an object")
    expected_size = sum(
        (model_directory / name).stat().st_size for name in MODEL_REQUIRED_FILES
    )
    expected_values = {
        "model_id": settings.model_id,
        "revision": settings.model_revision,
        "weights_sha256": weights_sha256,
        "size_bytes": expected_size,
        "files": list(MODEL_REQUIRED_FILES),
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise DenseRetrievalError(
                f"embedding download manifest mismatch for {key}: "
                f"expected={expected!r}, actual={manifest.get(key)!r}"
            )
    downloaded_at = manifest.get("downloaded_at_utc")
    if not isinstance(downloaded_at, str) or not downloaded_at.endswith("Z"):
        raise DenseRetrievalError(
            "embedding download manifest has invalid downloaded_at_utc"
        )


def build_dense_index(
    chunks: Sequence[ChunkRecord],
    encoder: EmbeddingEncoder,
    client: QdrantClient,
    settings: DenseConfig,
    *,
    rebuild: bool = False,
) -> DenseIndexResult:
    if not chunks:
        raise DenseRetrievalError("cannot build dense index from an empty chunk corpus")
    if encoder.dimension != settings.vector_size:
        raise DenseRetrievalError("encoder dimension does not match dense config")
    exists = client.collection_exists(settings.collection_name)
    if exists and not rebuild:
        raise DenseRetrievalError(
            f"collection already exists; use --rebuild explicitly: {settings.collection_name}"
        )
    if exists:
        client.delete_collection(settings.collection_name)
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=models.VectorParams(
            size=settings.vector_size, distance=models.Distance.COSINE
        ),
        on_disk_payload=True,
        metadata={
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
        },
    )
    try:
        for start in range(0, len(chunks), settings.batch_size):
            batch = chunks[start : start + settings.batch_size]
            vectors = encoder.encode_passages([chunk.text for chunk in batch])
            points = [
                models.PointStruct(
                    id=dense_point_id(chunk.chunk_id),
                    vector=vector.tolist(),
                    payload=_chunk_payload(chunk, settings),
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            client.upsert(
                collection_name=settings.collection_name, points=points, wait=True
            )
    except Exception:
        client.delete_collection(settings.collection_name)
        raise
    return DenseIndexResult(
        collection_name=settings.collection_name,
        indexed_chunks=len(chunks),
        vector_size=settings.vector_size,
    )


def dense_search(
    question: str,
    trusted_chunks: Sequence[ChunkRecord],
    encoder: EmbeddingEncoder,
    client: QdrantClient,
    settings: DenseConfig,
    *,
    top_k: int,
    document_ids: Sequence[str] | None = None,
) -> list[RetrievalHit]:
    if top_k <= 0:
        raise DenseRetrievalError("top_k must be positive")
    if not question.strip():
        return []
    if not client.collection_exists(settings.collection_name):
        raise DenseRetrievalError(
            f"dense collection not found: {settings.collection_name}"
        )
    trusted_by_id = {chunk.chunk_id: chunk for chunk in trusted_chunks}
    query_filter = None
    if document_ids:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(document_ids))
                )
            ]
        )
    response = client.query_points(
        collection_name=settings.collection_name,
        query=encoder.encode_query(question).tolist(),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=settings.minimum_score,
    )
    hits: list[RetrievalHit] = []
    for point in response.points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in trusted_by_id:
            raise DenseRetrievalError(
                f"Qdrant returned an unknown or invalid chunk_id: {chunk_id}"
            )
        chunk = trusted_by_id[chunk_id]
        _validate_payload(payload, chunk, settings)
        hits.append(
            RetrievalHit(
                rank=len(hits) + 1,
                score=float(point.score),
                retriever="dense",
                chunk=chunk,
            )
        )
    return hits


def dense_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"turkish-local-rag-benchmark:{chunk_id}"))


def _chunk_payload(chunk: ChunkRecord, settings: DenseConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "page_number": chunk.page_number,
        "source_page_url": chunk.source_page_url,
        "pdf_url": chunk.pdf_url,
        "pdf_sha256": chunk.pdf_sha256,
        "source_block_ids": list(chunk.source_block_ids),
        "text": chunk.text,
        "estimated_tokens": chunk.estimated_tokens,
        "token_count_method": chunk.token_count_method,
        "embedding_model_id": settings.model_id,
        "embedding_model_revision": settings.model_revision,
        "embedding_prefix": settings.passage_prefix,
    }


def _validate_payload(
    payload: Mapping[str, Any], chunk: ChunkRecord, settings: DenseConfig
) -> None:
    expected = _chunk_payload(chunk, settings)
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise DenseRetrievalError(
                f"untrusted Qdrant payload mismatch for {chunk.chunk_id}: {key}"
            )


def _model_is_complete(model_directory: Path) -> bool:
    return all((model_directory / name).is_file() for name in MODEL_REQUIRED_FILES)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
