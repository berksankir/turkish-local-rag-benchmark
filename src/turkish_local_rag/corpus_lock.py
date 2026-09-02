"""Build and verify the committed corpus lock without committing source PDFs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

from turkish_local_rag.config import ResolvedPaths, load_config
from turkish_local_rag.download import SourceDocument, load_manifest, sha256_file


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CorpusLockError(RuntimeError):
    """Raised when committed corpus identity and local files disagree."""


@dataclass(frozen=True, slots=True)
class CorpusLockRecord:
    document_id: str
    title: str
    source_page_url: str
    pdf_url: str
    final_pdf_url: str
    downloaded_at_utc: str
    size_bytes: int
    sha256: str


def load_corpus_lock(path: str | Path) -> tuple[CorpusLockRecord, ...]:
    lock_path = Path(path)
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusLockError(f"corpus lock not found: {lock_path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusLockError(f"invalid corpus lock JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "kind",
        "documents",
    }:
        raise CorpusLockError("corpus lock root has missing or unknown fields")
    if raw["schema_version"] != 1 or raw["kind"] != "corpus_lock":
        raise CorpusLockError("unsupported corpus lock schema or kind")
    if not isinstance(raw["documents"], list) or not raw["documents"]:
        raise CorpusLockError("corpus lock documents must be a non-empty list")

    expected = {
        "document_id",
        "title",
        "source_page_url",
        "pdf_url",
        "final_pdf_url",
        "downloaded_at_utc",
        "size_bytes",
        "sha256",
    }
    records: list[CorpusLockRecord] = []
    seen: set[str] = set()
    for index, value in enumerate(raw["documents"]):
        if not isinstance(value, dict) or set(value) != expected:
            raise CorpusLockError(
                f"corpus lock documents[{index}] has missing or unknown fields"
            )
        for field in expected - {"size_bytes"}:
            if not isinstance(value[field], str) or not value[field].strip():
                raise CorpusLockError(
                    f"corpus lock documents[{index}].{field} is invalid"
                )
        if (
            isinstance(value["size_bytes"], bool)
            or not isinstance(value["size_bytes"], int)
            or value["size_bytes"] <= 0
        ):
            raise CorpusLockError(
                f"corpus lock documents[{index}].size_bytes is invalid"
            )
        if not SHA256_PATTERN.fullmatch(value["sha256"]):
            raise CorpusLockError(
                f"corpus lock documents[{index}].sha256 is invalid"
            )
        record = CorpusLockRecord(**value)
        if record.document_id in seen:
            raise CorpusLockError(f"duplicate corpus lock id: {record.document_id}")
        seen.add(record.document_id)
        records.append(record)
    return tuple(records)


def build_lock_records(paths: ResolvedPaths) -> tuple[CorpusLockRecord, ...]:
    records: list[CorpusLockRecord] = []
    for source in load_manifest(paths.source_manifest):
        metadata_path = paths.metadata_directory / f"{source.id}.json"
        pdf_path = paths.pdf_directory / f"{source.id}.pdf"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise CorpusLockError(
                f"cannot read verified download metadata for {source.id}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise CorpusLockError(f"download metadata is not an object for {source.id}")
        _verify_source_metadata(source, metadata)
        if not pdf_path.is_file():
            raise CorpusLockError(f"local PDF not found for {source.id}: {pdf_path}")
        actual_size = pdf_path.stat().st_size
        actual_hash = sha256_file(pdf_path)
        if metadata.get("size_bytes") != actual_size or metadata.get("sha256") != actual_hash:
            raise CorpusLockError(
                f"local PDF differs from verified metadata for {source.id}"
            )
        records.append(
            CorpusLockRecord(
                document_id=source.id,
                title=source.title,
                source_page_url=source.source_page_url,
                pdf_url=source.pdf_url,
                final_pdf_url=str(metadata["final_url"]),
                downloaded_at_utc=str(metadata["downloaded_at_utc"]),
                size_bytes=actual_size,
                sha256=actual_hash,
            )
        )
    return tuple(records)


def write_corpus_lock(paths: ResolvedPaths) -> str:
    payload = {
        "schema_version": 1,
        "kind": "corpus_lock",
        "documents": [asdict(record) for record in build_lock_records(paths)],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if paths.corpus_lock.exists():
        existing = paths.corpus_lock.read_text(encoding="utf-8")
        if existing != rendered:
            raise CorpusLockError(
                "corpus lock differs from verified local corpus; existing lock was not replaced"
            )
        return "unchanged"
    paths.corpus_lock.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(paths.corpus_lock, rendered)
    return "written"


def verify_corpus_lock(
    paths: ResolvedPaths, *, require_local_files: bool = True
) -> tuple[CorpusLockRecord, ...]:
    records = load_corpus_lock(paths.corpus_lock)
    sources = load_manifest(paths.source_manifest)
    if [record.document_id for record in records] != [source.id for source in sources]:
        raise CorpusLockError("corpus lock document order/ids differ from source manifest")
    by_id = {record.document_id: record for record in records}
    for source in sources:
        record = by_id[source.id]
        _verify_record_source(record, source)
        if not require_local_files:
            continue
        pdf_path = paths.pdf_directory / f"{source.id}.pdf"
        metadata_path = paths.metadata_directory / f"{source.id}.json"
        if not pdf_path.is_file() or not metadata_path.is_file():
            raise CorpusLockError(f"local corpus files are missing for {source.id}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CorpusLockError(f"invalid metadata JSON for {source.id}: {exc}") from exc
        _verify_source_metadata(source, metadata)
        if (
            pdf_path.stat().st_size != record.size_bytes
            or sha256_file(pdf_path) != record.sha256
            or metadata.get("size_bytes") != record.size_bytes
            or metadata.get("sha256") != record.sha256
            or metadata.get("final_url") != record.final_pdf_url
            or metadata.get("downloaded_at_utc") != record.downloaded_at_utc
        ):
            raise CorpusLockError(f"local corpus differs from lock for {source.id}")
    return records


def _verify_source_metadata(source: SourceDocument, metadata: Mapping[str, Any]) -> None:
    mapping = {
        "id": source.id,
        "title": source.title,
        "source_page_url": source.source_page_url,
        "pdf_url": source.pdf_url,
    }
    for field, expected in mapping.items():
        if metadata.get(field) != expected:
            raise CorpusLockError(f"download metadata {field} mismatch for {source.id}")
    for field in ("final_url", "downloaded_at_utc", "sha256"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise CorpusLockError(f"download metadata {field} is invalid for {source.id}")
    if not SHA256_PATTERN.fullmatch(metadata["sha256"]):
        raise CorpusLockError(f"download metadata sha256 is invalid for {source.id}")


def _verify_record_source(record: CorpusLockRecord, source: SourceDocument) -> None:
    if (
        record.title != source.title
        or record.source_page_url != source.source_page_url
        or record.pdf_url != source.pdf_url
    ):
        raise CorpusLockError(f"corpus lock source metadata mismatch for {source.id}")


def _write_atomic(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        paths = config.resolve_paths(args.config)
        if args.write:
            status = write_corpus_lock(paths)
            print(json.dumps({"status": status, "path": str(paths.corpus_lock)}))
        else:
            records = verify_corpus_lock(paths)
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "documents": len(records),
                        "path": str(paths.corpus_lock),
                    }
                )
            )
        return 0
    except (CorpusLockError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
