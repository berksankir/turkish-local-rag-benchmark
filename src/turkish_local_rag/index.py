"""Build the persistent local Qdrant dense index."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from qdrant_client import QdrantClient

from turkish_local_rag.config import load_config
from turkish_local_rag.dense import (
    DenseRetrievalError,
    SentenceTransformerE5Encoder,
    build_dense_index,
    download_embedding_model,
)
from turkish_local_rag.download import load_manifest
from turkish_local_rag.rerank import download_reranker_model
from turkish_local_rag.retrieve import RetrievalError, load_chunk_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/default.toml", help="TOML config path"
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="index only this manifest id; may be repeated",
    )
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--download-model-only",
        action="store_true",
        help="download and verify the pinned embedding model, then stop",
    )
    download_group.add_argument(
        "--download-reranker-only",
        action="store_true",
        help="download and verify the pinned reranker model, then stop",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="explicitly replace an existing dense collection",
    )
    args = parser.parse_args(argv)

    try:
        config_path = Path(args.config)
        config = load_config(config_path)
        paths = config.resolve_paths(config_path)
        if args.download_model_only:
            result = download_embedding_model(paths.embedding_model_directory, config.dense)
            payload = asdict(result)
            payload["model_directory"] = str(result.model_directory)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.download_reranker_only:
            result = download_reranker_model(
                paths.reranker_model_directory, config.reranker
            )
            payload = asdict(result)
            payload["model_directory"] = str(result.model_directory)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        sources = load_manifest(paths.source_manifest)
        selected_ids = set(args.source_id)
        known_ids = {source.id for source in sources}
        unknown_ids = selected_ids - known_ids
        if unknown_ids:
            raise DenseRetrievalError(
                f"unknown source id(s): {', '.join(sorted(unknown_ids))}"
            )
        selected_sources = tuple(
            source for source in sources if not selected_ids or source.id in selected_ids
        )
        chunks = load_chunk_corpus(paths.chunks_directory, selected_sources)
        encoder = SentenceTransformerE5Encoder(
            paths.embedding_model_directory, config.dense
        )
        paths.qdrant_directory.parent.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(paths.qdrant_directory))
        try:
            result = build_dense_index(
                chunks, encoder, client, config.dense, rebuild=args.rebuild
            )
        finally:
            client.close()
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    except (RetrievalError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
