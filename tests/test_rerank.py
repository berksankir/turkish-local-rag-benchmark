from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import pytest
from torch.nn import Identity

from turkish_local_rag.config import RerankerConfig
from turkish_local_rag.rerank import (
    RERANKER_REQUIRED_FILES,
    CrossEncoderReranker,
    RerankerError,
    download_reranker_model,
    rerank_hits,
)
from turkish_local_rag.retrieve import ChunkRecord, FusedHit


def _settings(weights_sha256: str) -> RerankerConfig:
    return RerankerConfig(
        model_id="test/reranker",
        model_revision="b" * 40,
        model_sha256=weights_sha256,
        max_sequence_length=64,
        batch_size=2,
        candidate_count=3,
        top_k=2,
        zero_shot_turkish=True,
    )


def _model_directory(tmp_path: Path) -> tuple[Path, str]:
    model_directory = tmp_path / "reranker"
    for relative_path in RERANKER_REQUIRED_FILES:
        path = model_directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reranker" if relative_path == "model.safetensors" else b"{}")
    weights_hash = hashlib.sha256(b"reranker").hexdigest()
    size_bytes = sum(
        (model_directory / relative_path).stat().st_size
        for relative_path in RERANKER_REQUIRED_FILES
    )
    (model_directory / "download_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "test/reranker",
                "revision": "b" * 40,
                "weights_sha256": weights_hash,
                "size_bytes": size_bytes,
                "downloaded_at_utc": "2026-09-01T12:00:00Z",
                "files": list(RERANKER_REQUIRED_FILES),
                "zero_shot_turkish": True,
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


def _fused(chunk: ChunkRecord, rank: int) -> FusedHit:
    return FusedHit(
        rank=rank,
        rrf_score=1.0 / (60 + rank),
        component_ranks={"bm25": rank, "dense": rank},
        chunk=chunk,
    )


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.predict_calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []

    def predict(
        self, pairs: list[tuple[str, str]], **kwargs: Any
    ) -> NDArray[np.float32]:
        self.predict_calls.append((pairs, kwargs))
        return np.asarray(
            [2.0 if "burs" in passage.casefold() else -1.0 for _, passage in pairs],
            dtype=np.float32,
        )


class FakeScorer:
    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = np.asarray(scores, dtype=np.float32)
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append((question, list(passages)))
        return self._scores


def test_cross_encoder_is_cpu_local_only_and_uses_raw_logits(tmp_path: Path) -> None:
    model_directory, weights_hash = _model_directory(tmp_path)
    fake_model = FakeCrossEncoder()
    factory_calls: list[tuple[str, dict[str, Any]]] = []

    def factory(path: str, **kwargs: Any) -> FakeCrossEncoder:
        factory_calls.append((path, kwargs))
        return fake_model

    scorer = CrossEncoderReranker(
        model_directory, _settings(weights_hash), model_factory=factory
    )
    scores = scorer.score("Burs koşulları nedir?", ["Burs başvurusu", "İhale usulü"])

    assert scores.tolist() == [2.0, -1.0]
    assert factory_calls[0][0] == str(model_directory)
    assert factory_calls[0][1]["device"] == "cpu"
    assert factory_calls[0][1]["local_files_only"] is True
    assert factory_calls[0][1]["backend"] == "torch"
    assert factory_calls[0][1]["max_length"] == 64
    assert isinstance(factory_calls[0][1]["activation_fn"], Identity)
    assert fake_model.predict_calls[0][0] == [
        ("Burs koşulları nedir?", "Burs başvurusu"),
        ("Burs koşulları nedir?", "İhale usulü"),
    ]
    assert fake_model.predict_calls[0][1]["batch_size"] == 2
    assert fake_model.predict_calls[0][1]["device"] == "cpu"


def test_cross_encoder_rejects_manifest_revision_before_model_load(
    tmp_path: Path,
) -> None:
    model_directory, weights_hash = _model_directory(tmp_path)
    manifest_path = model_directory / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "wrong-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    factory_called = False

    def factory(path: str, **kwargs: Any) -> FakeCrossEncoder:
        nonlocal factory_called
        factory_called = True
        return FakeCrossEncoder()

    with pytest.raises(RerankerError, match="manifest mismatch for revision"):
        CrossEncoderReranker(
            model_directory, _settings(weights_hash), model_factory=factory
        )

    assert factory_called is False


def test_reranker_reorders_bounded_candidates_and_preserves_metadata() -> None:
    first = _fused(_chunk(0, "İhale usulü", 5), 1)
    second = _fused(_chunk(1, "Burs başvurusu", 3), 2)
    third = _fused(_chunk(2, "Kütüphane", 7), 3)
    scorer = FakeScorer([0.1, 0.9, 0.2])

    hits = rerank_hits("Burs koşulları", [first, second, third], scorer, limit=2)

    assert [hit.chunk.chunk_id for hit in hits] == [second.chunk.chunk_id, third.chunk.chunk_id]
    assert hits[0].original_rank == 2
    assert hits[0].original_retriever == "rrf"
    assert hits[0].component_ranks == {"bm25": 2, "dense": 2}
    assert hits[0].chunk.page_number == 3
    assert hits[0].chunk.source_page_url == "https://example.test/source"
    assert len(scorer.calls[0][1]) == 3


def test_reranker_rejects_non_finite_scores() -> None:
    candidate = _fused(_chunk(0, "Burs", 1), 1)

    with pytest.raises(RerankerError, match="non-finite"):
        rerank_hits("Burs", [candidate], FakeScorer([float("nan")]), limit=1)


def test_selective_downloader_uses_pinned_revision_and_writes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "reranker"
    weights = b"downloaded-reranker"
    settings = _settings(hashlib.sha256(weights).hexdigest())
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        for relative_path in RERANKER_REQUIRED_FILES:
            path = target / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(weights if relative_path == "model.safetensors" else b"{}")
        return str(target)

    monkeypatch.setattr(
        "turkish_local_rag.rerank.snapshot_download", fake_snapshot_download
    )

    result = download_reranker_model(model_directory, settings)
    metadata = json.loads(
        (model_directory / "download_manifest.json").read_text(encoding="utf-8")
    )

    assert result.status == "downloaded"
    assert calls[0]["repo_id"] == settings.model_id
    assert calls[0]["revision"] == settings.model_revision
    assert set(calls[0]["allow_patterns"]) == {
        *RERANKER_REQUIRED_FILES,
        "README.md",
    }
    assert metadata["weights_sha256"] == settings.model_sha256
    assert metadata["zero_shot_turkish"] is True
    assert metadata["downloaded_at_utc"].endswith("Z")


def test_selective_downloader_refuses_to_overwrite_changed_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory, expected_hash = _model_directory(tmp_path)
    weights_path = model_directory / "model.safetensors"
    weights_path.write_bytes(b"changed")
    snapshot_called = False

    def fake_snapshot_download(**kwargs: Any) -> str:
        nonlocal snapshot_called
        snapshot_called = True
        return str(model_directory)

    monkeypatch.setattr(
        "turkish_local_rag.rerank.snapshot_download", fake_snapshot_download
    )

    with pytest.raises(RerankerError, match="file was not overwritten"):
        download_reranker_model(model_directory, _settings(expected_hash))

    assert snapshot_called is False
    assert weights_path.read_bytes() == b"changed"
