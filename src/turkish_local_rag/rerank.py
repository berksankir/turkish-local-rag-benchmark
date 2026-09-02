"""CPU-only cross-encoder reranking over trusted retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from huggingface_hub import snapshot_download
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import CrossEncoder
import torch
from torch.nn import Identity

from turkish_local_rag.config import RerankerConfig
from turkish_local_rag.retrieve import (
    ChunkRecord,
    FusedHit,
    RetrievalError,
    RetrievalHit,
)


RERANKER_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
RERANKER_ALLOW_PATTERNS = (*RERANKER_REQUIRED_FILES, "README.md")


class RerankerError(RetrievalError):
    """Raised when the local reranker or its output is invalid."""


class PairScorer(Protocol):
    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class RerankerDownloadResult:
    status: str
    model_directory: Path
    model_id: str
    revision: str
    weights_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RerankedHit:
    rank: int
    reranker_score: float
    original_rank: int
    original_retriever: str
    retrieval_score: float
    component_ranks: Mapping[str, int]
    chunk: ChunkRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "reranker_score": self.reranker_score,
            "original_rank": self.original_rank,
            "original_retriever": self.original_retriever,
            "retrieval_score": self.retrieval_score,
            "component_ranks": dict(self.component_ranks),
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "title": self.chunk.title,
            "page_number": self.chunk.page_number,
            "source_page_url": self.chunk.source_page_url,
            "pdf_url": self.chunk.pdf_url,
            "text": self.chunk.text,
        }


class CrossEncoderReranker:
    """Local PyTorch cross-encoder using raw logits for ordering only."""

    def __init__(
        self,
        model_directory: str | Path,
        settings: RerankerConfig,
        *,
        model_factory: Callable[..., Any] = CrossEncoder,
        thread_setter: Callable[[int], None] = torch.set_num_threads,
    ) -> None:
        verify_reranker_model(model_directory, settings)
        self._settings = settings
        self._activation = Identity()
        self._thread_setter = thread_setter
        self._cpu_threads: int | None = None
        self._set_cpu_threads(settings.cpu_threads)
        self._model = model_factory(
            str(Path(model_directory)),
            device="cpu",
            local_files_only=True,
            backend="torch",
            max_length=settings.max_sequence_length,
            activation_fn=self._activation,
        )

    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float32]:
        return self.score_with_options(question, passages)

    def score_with_options(
        self,
        question: str,
        passages: Sequence[str],
        *,
        batch_size: int | None = None,
        cpu_threads: int | None = None,
    ) -> NDArray[np.float32]:
        """Score with bounded runtime overrides while reusing the loaded model."""

        if not question.strip():
            raise RerankerError("reranker question cannot be empty")
        if not passages:
            return np.empty((0,), dtype=np.float32)
        effective_batch_size = self._settings.batch_size if batch_size is None else batch_size
        effective_cpu_threads = (
            self._settings.cpu_threads if cpu_threads is None else cpu_threads
        )
        if effective_batch_size <= 0:
            raise RerankerError("reranker batch_size must be positive")
        if effective_cpu_threads <= 0:
            raise RerankerError("reranker cpu_threads must be positive")
        self._set_cpu_threads(effective_cpu_threads)
        pairs = [(question, passage) for passage in passages]
        scores = self._model.predict(
            pairs,
            batch_size=effective_batch_size,
            show_progress_bar=False,
            activation_fn=self._activation,
            apply_softmax=False,
            convert_to_numpy=True,
            device="cpu",
        )
        array = np.asarray(scores, dtype=np.float32).reshape(-1)
        if array.shape != (len(passages),):
            raise RerankerError(
                f"unexpected reranker score shape: expected {(len(passages),)}, "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise RerankerError("reranker returned a non-finite score")
        return array

    def _set_cpu_threads(self, cpu_threads: int) -> None:
        if self._cpu_threads == cpu_threads:
            return
        self._thread_setter(cpu_threads)
        self._cpu_threads = cpu_threads


def rerank_hits(
    question: str,
    candidates: Sequence[RetrievalHit | FusedHit],
    scorer: PairScorer,
    *,
    limit: int,
) -> list[RerankedHit]:
    """Rerank a bounded candidate list while retaining trusted metadata."""

    if limit <= 0:
        raise RerankerError("reranker limit must be positive")
    if not candidates:
        return []
    seen_chunk_ids: set[str] = set()
    for candidate in candidates:
        if candidate.chunk.chunk_id in seen_chunk_ids:
            raise RerankerError(
                f"duplicate reranker candidate: {candidate.chunk.chunk_id}"
            )
        seen_chunk_ids.add(candidate.chunk.chunk_id)
    scores = scorer.score(question, [candidate.chunk.text for candidate in candidates])
    if scores.shape != (len(candidates),):
        raise RerankerError(
            f"scorer returned {scores.shape} for {len(candidates)} candidates"
        )
    if not np.all(np.isfinite(scores)):
        raise RerankerError("scorer returned a non-finite score")

    ordered = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].rank, item[0].chunk.chunk_id),
    )[:limit]
    results: list[RerankedHit] = []
    for rank, (candidate, score) in enumerate(ordered, start=1):
        # ``python -m turkish_local_rag.retrieve`` executes the retriever as
        # ``__main__`` while this module imports its canonical package name.
        # Attribute-based recognition avoids treating that equivalent FusedHit
        # class as a plain RetrievalHit solely because the module identities differ.
        if hasattr(candidate, "rrf_score") and hasattr(candidate, "component_ranks"):
            original_retriever = "rrf"
            retrieval_score = candidate.rrf_score
            component_ranks = candidate.component_ranks
        else:
            original_retriever = candidate.retriever
            retrieval_score = candidate.score
            component_ranks = {candidate.retriever: candidate.rank}
        results.append(
            RerankedHit(
                rank=rank,
                reranker_score=float(score),
                original_rank=candidate.rank,
                original_retriever=original_retriever,
                retrieval_score=retrieval_score,
                component_ranks=dict(component_ranks),
                chunk=candidate.chunk,
            )
        )
    return results


def download_reranker_model(
    model_directory: str | Path, settings: RerankerConfig
) -> RerankerDownloadResult:
    """Download only pinned PyTorch safetensors and tokenizer files."""

    target = Path(model_directory)
    weights_path = target / "model.safetensors"
    if weights_path.exists():
        actual_hash = _sha256_file(weights_path)
        if actual_hash != settings.model_sha256:
            raise RerankerError(
                "existing reranker weights hash mismatch; file was not overwritten: "
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
        allow_patterns=list(RERANKER_ALLOW_PATTERNS),
    )
    weights_sha256 = _verify_reranker_files(target, settings)
    size_bytes = sum(
        path.stat().st_size
        for relative_path in RERANKER_REQUIRED_FILES
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
        "files": list(RERANKER_REQUIRED_FILES),
        "zero_shot_turkish": settings.zero_shot_turkish,
    }
    _write_json_atomic(target / "download_manifest.json", metadata)
    verify_reranker_model(target, settings)
    return RerankerDownloadResult(
        status=status,
        model_directory=target,
        model_id=settings.model_id,
        revision=settings.model_revision,
        weights_sha256=weights_sha256,
        size_bytes=size_bytes,
    )


def verify_reranker_model(
    model_directory: str | Path, settings: RerankerConfig
) -> str:
    target = Path(model_directory)
    weights_sha256 = _verify_reranker_files(target, settings)
    _verify_download_manifest(target, settings, weights_sha256)
    return weights_sha256


def _verify_reranker_files(model_directory: Path, settings: RerankerConfig) -> str:
    target = model_directory
    missing = [name for name in RERANKER_REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        raise RerankerError(
            f"reranker model is incomplete at {target}; missing: {', '.join(missing)}"
        )
    weights_sha256 = _sha256_file(target / "model.safetensors")
    if weights_sha256 != settings.model_sha256:
        raise RerankerError(
            "reranker weights SHA-256 mismatch: "
            f"expected={settings.model_sha256}, actual={weights_sha256}"
        )
    return weights_sha256


def _verify_download_manifest(
    model_directory: Path, settings: RerankerConfig, weights_sha256: str
) -> None:
    manifest_path = model_directory / "download_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RerankerError(f"reranker download manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise RerankerError(
            f"invalid reranker download manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RerankerError("reranker download manifest must be an object")
    expected_size = sum(
        (model_directory / name).stat().st_size for name in RERANKER_REQUIRED_FILES
    )
    expected_values = {
        "model_id": settings.model_id,
        "revision": settings.model_revision,
        "weights_sha256": weights_sha256,
        "size_bytes": expected_size,
        "files": list(RERANKER_REQUIRED_FILES),
        "zero_shot_turkish": settings.zero_shot_turkish,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise RerankerError(
                f"reranker download manifest mismatch for {key}: "
                f"expected={expected!r}, actual={manifest.get(key)!r}"
            )
    downloaded_at = manifest.get("downloaded_at_utc")
    if not isinstance(downloaded_at, str) or not downloaded_at.endswith("Z"):
        raise RerankerError("reranker download manifest has invalid downloaded_at_utc")


def _model_is_complete(model_directory: Path) -> bool:
    return all((model_directory / name).is_file() for name in RERANKER_REQUIRED_FILES)


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
