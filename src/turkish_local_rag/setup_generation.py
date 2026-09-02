"""Verify or install the pinned local generator model and llama.cpp runtime."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import sys
import tempfile
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import zipfile

from turkish_local_rag.config import GeneratorConfig, ResolvedPaths, load_config


class GenerationSetupError(RuntimeError):
    """Raised when a pinned generation asset cannot be safely installed."""


@dataclass(frozen=True, slots=True)
class AssetSpec:
    name: str
    url: str
    size_bytes: int
    sha256: str
    allowed_host_suffixes: tuple[str, ...]


def asset_specs(paths: ResolvedPaths, settings: GeneratorConfig) -> tuple[AssetSpec, AssetSpec]:
    model_filename = paths.generator_model_file.name
    model_url = (
        f"https://huggingface.co/{settings.model_id}/resolve/"
        f"{settings.model_revision}/{quote(model_filename)}?download=true"
    )
    archive_name = f"llama-{settings.runtime_version}-bin-win-cpu-x64.zip"
    runtime_url = (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        f"{settings.runtime_version}/{archive_name}"
    )
    return (
        AssetSpec(
            "generator_model",
            model_url,
            settings.model_size_bytes,
            settings.model_sha256,
            ("huggingface.co", ".huggingface.co", ".hf.co"),
        ),
        AssetSpec(
            "llama_cpp_runtime_archive",
            runtime_url,
            settings.runtime_size_bytes,
            settings.runtime_sha256,
            ("github.com", ".githubusercontent.com"),
        ),
    )


def verify_generation_assets(
    paths: ResolvedPaths, settings: GeneratorConfig
) -> dict[str, Any]:
    model_spec, runtime_spec = asset_specs(paths, settings)
    runtime_root = paths.llama_server_executable.parent.parent
    runtime_archive = runtime_root / Path(urlparse(runtime_spec.url).path).name
    _verify_file(paths.generator_model_file, model_spec)
    _verify_file(runtime_archive, runtime_spec)
    if not paths.llama_server_executable.is_file():
        raise GenerationSetupError(
            f"llama.cpp executable is missing: {paths.llama_server_executable}"
        )
    return {
        "status": "verified",
        "model": _asset_summary(model_spec, paths.generator_model_file),
        "runtime": _asset_summary(runtime_spec, runtime_archive),
        "runtime_executable": str(paths.llama_server_executable),
    }


def install_generation_assets(
    paths: ResolvedPaths,
    settings: GeneratorConfig,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    model_spec, runtime_spec = asset_specs(paths, settings)
    model_status = download_verified_asset(
        model_spec,
        paths.generator_model_file,
        timeout_seconds=settings.timeout_seconds,
        opener=opener,
    )

    runtime_root = paths.llama_server_executable.parent.parent
    runtime_archive = runtime_root / Path(urlparse(runtime_spec.url).path).name
    if runtime_root.exists():
        _verify_file(runtime_archive, runtime_spec)
        if not paths.llama_server_executable.is_file():
            raise GenerationSetupError(
                "runtime directory exists but the pinned executable is missing; "
                "existing files were not replaced"
            )
        runtime_status = "verified_existing"
    else:
        runtime_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{runtime_root.name}-", dir=runtime_root.parent)
        )
        try:
            temporary_archive = temporary_root / runtime_archive.name
            download_verified_asset(
                runtime_spec,
                temporary_archive,
                timeout_seconds=settings.timeout_seconds,
                opener=opener,
            )
            bin_directory = temporary_root / "bin"
            _extract_runtime_archive(temporary_archive, bin_directory)
            relative_executable = paths.llama_server_executable.relative_to(runtime_root)
            if not (temporary_root / relative_executable).is_file():
                raise GenerationSetupError(
                    f"runtime archive lacks {relative_executable.as_posix()}"
                )
            if runtime_root.exists():
                raise GenerationSetupError(
                    "runtime target appeared during installation; it was not replaced"
                )
            os.replace(temporary_root, runtime_root)
            runtime_status = "downloaded"
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
    result = verify_generation_assets(paths, settings)
    result["model"]["install_status"] = model_status
    result["runtime"]["install_status"] = runtime_status
    return result


def download_verified_asset(
    spec: AssetSpec,
    destination: Path,
    *,
    timeout_seconds: int,
    opener: Callable[..., Any] = urlopen,
) -> str:
    """Stream one pinned asset to a temporary file, verify, then atomically move."""

    if destination.exists():
        _verify_file(destination, spec)
        return "verified_existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    request = Request(spec.url, headers={"User-Agent": "turkish-local-rag-benchmark/0.1"})
    try:
        try:
            response = opener(request, timeout=timeout_seconds)
        except HTTPError as exc:
            raise GenerationSetupError(f"HTTP {exc.code} while downloading {spec.name}") from exc
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise GenerationSetupError(f"network failure while downloading {spec.name}") from exc
        with closing(response):
            status = getattr(response, "status", 200)
            if not 200 <= status < 300:
                raise GenerationSetupError(f"HTTP {status} while downloading {spec.name}")
            _validate_final_url(spec, response)
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    announced = int(content_length)
                except (TypeError, ValueError) as exc:
                    raise GenerationSetupError(
                        f"invalid Content-Length for {spec.name}"
                    ) from exc
                if announced != spec.size_bytes:
                    raise GenerationSetupError(
                        f"size mismatch for {spec.name}: expected={spec.size_bytes}, "
                        f"announced={announced}"
                    )
            digest = hashlib.sha256()
            received = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > spec.size_bytes:
                        raise GenerationSetupError(
                            f"download exceeds pinned size for {spec.name}"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if received != spec.size_bytes or digest.hexdigest() != spec.sha256:
            raise GenerationSetupError(
                f"download verification failed for {spec.name}; destination was not replaced"
            )
        if destination.exists():
            raise GenerationSetupError(
                f"destination appeared during download for {spec.name}; it was not replaced"
            )
        os.replace(temporary, destination)
        temporary = None
        return "downloaded"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_file(path: Path, spec: AssetSpec) -> None:
    if not path.is_file():
        raise GenerationSetupError(f"{spec.name} not found: {path}")
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise GenerationSetupError(
            f"{spec.name} size mismatch; existing file was not replaced: "
            f"expected={spec.size_bytes}, actual={size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != spec.sha256:
        raise GenerationSetupError(
            f"{spec.name} SHA-256 mismatch; existing file was not replaced: "
            f"expected={spec.sha256}, actual={actual_hash}"
        )


def _validate_final_url(spec: AssetSpec, response: Any) -> None:
    final_url = response.geturl() if callable(getattr(response, "geturl", None)) else spec.url
    parsed = urlparse(final_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise GenerationSetupError(f"unsafe final URL for {spec.name}")
    host = parsed.hostname.casefold()
    if not any(
        host == suffix or (suffix.startswith(".") and host.endswith(suffix))
        for suffix in spec.allowed_host_suffixes
    ):
        raise GenerationSetupError(f"untrusted redirect host for {spec.name}: {host}")


def _extract_runtime_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise GenerationSetupError("runtime archive contains an unsafe path")
                if member.is_dir():
                    continue
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (zipfile.BadZipFile, OSError) as exc:
        raise GenerationSetupError("cannot safely extract llama.cpp runtime archive") from exc


def _asset_summary(spec: AssetSpec, path: Path) -> dict[str, Any]:
    return {
        "name": spec.name,
        "url": spec.url,
        "path": str(path),
        "size_bytes": spec.size_bytes,
        "sha256": spec.sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true")
    action.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        paths = config.resolve_paths(args.config)
        result = (
            install_generation_assets(paths, config.generation)
            if args.download
            else verify_generation_assets(paths, config.generation)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (GenerationSetupError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
