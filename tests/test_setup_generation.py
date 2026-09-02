from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from typing import Any

import pytest

from turkish_local_rag.config import load_config
from turkish_local_rag.setup_generation import (
    AssetSpec,
    GenerationSetupError,
    asset_specs,
    download_verified_asset,
)


class FakeResponse:
    def __init__(self, body: bytes, *, final_url: str = "https://files.example.test/a"):
        self._body = BytesIO(body)
        self.status = 200
        self.headers = {"Content-Length": str(len(body))}
        self._final_url = final_url

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._final_url

    def close(self) -> None:
        self._body.close()


def _spec(body: bytes) -> AssetSpec:
    return AssetSpec(
        "fixture",
        "https://files.example.test/a",
        len(body),
        hashlib.sha256(body).hexdigest(),
        ("files.example.test",),
    )


def test_asset_downloader_uses_mock_and_atomic_verified_write(tmp_path: Path) -> None:
    body = b"small fixture"
    destination = tmp_path / "asset.bin"
    calls: list[Any] = []

    def opener(request: Any, timeout: int) -> FakeResponse:
        calls.append((request.full_url, timeout))
        return FakeResponse(body)

    status = download_verified_asset(
        _spec(body), destination, timeout_seconds=7, opener=opener
    )

    assert status == "downloaded"
    assert destination.read_bytes() == body
    assert calls == [("https://files.example.test/a", 7)]
    assert not list(tmp_path.glob("*.tmp"))


def test_asset_downloader_does_not_overwrite_hash_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"existing changed bytes")

    with pytest.raises(GenerationSetupError, match="was not replaced"):
        download_verified_asset(
            _spec(b"expected"),
            destination,
            timeout_seconds=7,
            opener=lambda *_args, **_kwargs: pytest.fail("network must not be called"),
        )

    assert destination.read_bytes() == b"existing changed bytes"


def test_asset_downloader_rejects_untrusted_redirect(tmp_path: Path) -> None:
    body = b"expected"

    with pytest.raises(GenerationSetupError, match="untrusted redirect"):
        download_verified_asset(
            _spec(body),
            tmp_path / "asset.bin",
            timeout_seconds=7,
            opener=lambda *_args, **_kwargs: FakeResponse(
                body, final_url="https://attacker.example/asset"
            ),
        )


def test_pinned_generation_asset_specs_match_config() -> None:
    config = load_config("config/default.toml")
    model, runtime = asset_specs(
        config.resolve_paths("config/default.toml"), config.generation
    )

    assert config.generation.model_revision in model.url
    assert model.size_bytes == 1_117_320_736
    assert runtime.url.endswith("llama-b10621-bin-win-cpu-x64.zip")
    assert runtime.size_bytes == 18_068_018
