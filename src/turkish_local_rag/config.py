"""Typed project configuration loaded with Python's standard library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import tomllib


class ConfigError(ValueError):
    """Raised when a configuration file is missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ResolvedPaths:
    project_root: Path
    source_manifest: Path
    pdf_directory: Path
    metadata_directory: Path
    extracted_pages_directory: Path
    chunks_directory: Path
    embedding_model_directory: Path
    reranker_model_directory: Path
    qdrant_directory: Path
    evaluation_candidates: Path
    evaluation_review: Path
    evaluation_gold: Path
    evaluation_results_directory: Path


@dataclass(frozen=True, slots=True)
class PathsConfig:
    project_root: str
    source_manifest: str
    pdf_directory: str
    metadata_directory: str
    extracted_pages_directory: str
    chunks_directory: str
    embedding_model_directory: str
    reranker_model_directory: str
    qdrant_directory: str
    evaluation_candidates: str
    evaluation_review: str
    evaluation_gold: str
    evaluation_results_directory: str

    def validate(self) -> None:
        for name, value in (
            ("project_root", self.project_root),
            ("source_manifest", self.source_manifest),
            ("pdf_directory", self.pdf_directory),
            ("metadata_directory", self.metadata_directory),
            ("extracted_pages_directory", self.extracted_pages_directory),
            ("chunks_directory", self.chunks_directory),
            ("embedding_model_directory", self.embedding_model_directory),
            ("reranker_model_directory", self.reranker_model_directory),
            ("qdrant_directory", self.qdrant_directory),
            ("evaluation_candidates", self.evaluation_candidates),
            ("evaluation_review", self.evaluation_review),
            ("evaluation_gold", self.evaluation_gold),
            ("evaluation_results_directory", self.evaluation_results_directory),
        ):
            if not value.strip():
                raise ConfigError(f"paths.{name} cannot be empty")

    def resolve(self, config_path: str | Path) -> ResolvedPaths:
        """Resolve portable config paths and reject paths escaping project_root."""

        config_directory = Path(config_path).resolve().parent
        configured_root = Path(self.project_root)
        if configured_root.is_absolute():
            raise ConfigError("paths.project_root must be relative to the config file")
        project_root = (config_directory / configured_root).resolve()
        return ResolvedPaths(
            project_root=project_root,
            source_manifest=_resolve_within_project(
                project_root, self.source_manifest, "paths.source_manifest"
            ),
            pdf_directory=_resolve_within_project(
                project_root, self.pdf_directory, "paths.pdf_directory"
            ),
            metadata_directory=_resolve_within_project(
                project_root, self.metadata_directory, "paths.metadata_directory"
            ),
            extracted_pages_directory=_resolve_within_project(
                project_root,
                self.extracted_pages_directory,
                "paths.extracted_pages_directory",
            ),
            chunks_directory=_resolve_within_project(
                project_root, self.chunks_directory, "paths.chunks_directory"
            ),
            embedding_model_directory=_resolve_within_project(
                project_root,
                self.embedding_model_directory,
                "paths.embedding_model_directory",
            ),
            reranker_model_directory=_resolve_within_project(
                project_root,
                self.reranker_model_directory,
                "paths.reranker_model_directory",
            ),
            qdrant_directory=_resolve_within_project(
                project_root, self.qdrant_directory, "paths.qdrant_directory"
            ),
            evaluation_candidates=_resolve_within_project(
                project_root,
                self.evaluation_candidates,
                "paths.evaluation_candidates",
            ),
            evaluation_review=_resolve_within_project(
                project_root, self.evaluation_review, "paths.evaluation_review"
            ),
            evaluation_gold=_resolve_within_project(
                project_root, self.evaluation_gold, "paths.evaluation_gold"
            ),
            evaluation_results_directory=_resolve_within_project(
                project_root,
                self.evaluation_results_directory,
                "paths.evaluation_results_directory",
            ),
        )


@dataclass(frozen=True, slots=True)
class DownloaderConfig:
    timeout_seconds: int
    chunk_size_bytes: int
    maximum_pdf_bytes: int
    user_agent: str

    def validate(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigError("downloader.timeout_seconds must be positive")
        if self.chunk_size_bytes <= 0:
            raise ConfigError("downloader.chunk_size_bytes must be positive")
        if self.maximum_pdf_bytes < 5:
            raise ConfigError("downloader.maximum_pdf_bytes must be at least 5")
        if not self.user_agent.strip():
            raise ConfigError("downloader.user_agent cannot be empty")


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    sort_blocks: bool
    include_empty_pages: bool


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_model_tokens: int
    maximum_model_tokens: int
    overlap_model_tokens: int
    estimated_characters_per_token: int
    respect_page_boundaries: bool
    preserve_article_boundaries: bool

    def validate(self) -> None:
        if not 1 <= self.target_model_tokens <= 512:
            raise ConfigError("chunking.target_model_tokens must be between 1 and 512")
        if not self.target_model_tokens <= self.maximum_model_tokens <= 512:
            raise ConfigError(
                "chunking.maximum_model_tokens must be between target_model_tokens and 512"
            )
        if not 0 <= self.overlap_model_tokens < self.target_model_tokens:
            raise ConfigError(
                "chunking.overlap_model_tokens must be non-negative and smaller than "
                "target_model_tokens"
            )
        if self.target_model_tokens + self.overlap_model_tokens > self.maximum_model_tokens:
            raise ConfigError(
                "chunking target_model_tokens plus overlap_model_tokens cannot exceed "
                "maximum_model_tokens"
            )
        if self.estimated_characters_per_token <= 0:
            raise ConfigError(
                "chunking.estimated_characters_per_token must be positive"
            )
        if not self.respect_page_boundaries:
            raise ConfigError("chunking.respect_page_boundaries must remain true")


@dataclass(frozen=True, slots=True)
class RRFConfig:
    rank_constant: int
    dense_candidates: int
    sparse_candidates: int
    fused_candidates: int

    def validate(self) -> None:
        values = (
            self.rank_constant,
            self.dense_candidates,
            self.sparse_candidates,
            self.fused_candidates,
        )
        if any(value <= 0 for value in values):
            raise ConfigError("all retrieval.rrf integer settings must be positive")
        if self.fused_candidates > self.dense_candidates + self.sparse_candidates:
            raise ConfigError(
                "retrieval.rrf.fused_candidates cannot exceed the combined candidate count"
            )


@dataclass(frozen=True, slots=True)
class BM25Config:
    k1: float
    b: float
    epsilon: float
    minimum_score: float
    top_k: int

    def validate(self) -> None:
        if self.k1 <= 0:
            raise ConfigError("retrieval.bm25.k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ConfigError("retrieval.bm25.b must be between 0 and 1")
        if self.epsilon < 0:
            raise ConfigError("retrieval.bm25.epsilon cannot be negative")
        if self.minimum_score < 0:
            raise ConfigError("retrieval.bm25.minimum_score cannot be negative")
        if self.top_k <= 0:
            raise ConfigError("retrieval.bm25.top_k must be positive")


@dataclass(frozen=True, slots=True)
class DenseConfig:
    model_id: str
    model_revision: str
    model_sha256: str
    vector_size: int
    max_sequence_length: int
    batch_size: int
    query_prefix: str
    passage_prefix: str
    normalize_embeddings: bool
    collection_name: str
    minimum_score: float

    def validate(self) -> None:
        for name, value in (
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("collection_name", self.collection_name),
        ):
            if not value.strip():
                raise ConfigError(f"retrieval.dense.{name} cannot be empty")
        if len(self.model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_sha256
        ):
            raise ConfigError("retrieval.dense.model_sha256 must be lowercase SHA-256")
        if self.vector_size <= 0:
            raise ConfigError("retrieval.dense.vector_size must be positive")
        if not 1 <= self.max_sequence_length <= 512:
            raise ConfigError(
                "retrieval.dense.max_sequence_length must be between 1 and 512"
            )
        if self.batch_size <= 0:
            raise ConfigError("retrieval.dense.batch_size must be positive")
        if self.query_prefix != "query: ":
            raise ConfigError("retrieval.dense.query_prefix must be 'query: '")
        if self.passage_prefix != "passage: ":
            raise ConfigError("retrieval.dense.passage_prefix must be 'passage: '")
        if not self.normalize_embeddings:
            raise ConfigError("retrieval.dense.normalize_embeddings must remain true")
        if not -1 <= self.minimum_score <= 1:
            raise ConfigError("retrieval.dense.minimum_score must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    model_id: str
    model_revision: str
    model_sha256: str
    max_sequence_length: int
    batch_size: int
    candidate_count: int
    top_k: int
    zero_shot_turkish: bool

    def validate(self) -> None:
        for name, value in (
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
        ):
            if not value.strip():
                raise ConfigError(f"retrieval.reranker.{name} cannot be empty")
        if len(self.model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_sha256
        ):
            raise ConfigError("retrieval.reranker.model_sha256 must be lowercase SHA-256")
        if not 1 <= self.max_sequence_length <= 512:
            raise ConfigError(
                "retrieval.reranker.max_sequence_length must be between 1 and 512"
            )
        if self.batch_size <= 0:
            raise ConfigError("retrieval.reranker.batch_size must be positive")
        if self.candidate_count <= 0:
            raise ConfigError("retrieval.reranker.candidate_count must be positive")
        if not 1 <= self.top_k <= self.candidate_count:
            raise ConfigError(
                "retrieval.reranker.top_k must be between 1 and candidate_count"
            )
        if not self.zero_shot_turkish:
            raise ConfigError(
                "retrieval.reranker.zero_shot_turkish must remain true for this model"
            )


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    schema_version: int
    paths: PathsConfig
    downloader: DownloaderConfig
    extraction: ExtractionConfig
    chunking: ChunkingConfig
    rrf: RRFConfig
    bm25: BM25Config
    dense: DenseConfig
    reranker: RerankerConfig

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError(f"unsupported schema_version: {self.schema_version}")
        self.paths.validate()
        self.downloader.validate()
        self.chunking.validate()
        self.rrf.validate()
        self.bm25.validate()
        self.dense.validate()
        self.reranker.validate()

    def resolve_paths(self, config_path: str | Path) -> ResolvedPaths:
        return self.paths.resolve(config_path)


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a TOML config without importing third-party packages."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    root = _table(raw, "root")
    _require_exact_keys(
        root,
        {
            "schema_version",
            "paths",
            "downloader",
            "extraction",
            "chunking",
            "retrieval",
        },
        "root",
    )
    paths = _table(root.get("paths"), "paths")
    downloader = _table(root.get("downloader"), "downloader")
    extraction = _table(root.get("extraction"), "extraction")
    chunking = _table(root.get("chunking"), "chunking")
    retrieval = _table(root.get("retrieval"), "retrieval")
    rrf = _table(retrieval.get("rrf"), "retrieval.rrf")
    bm25 = _table(retrieval.get("bm25"), "retrieval.bm25")
    dense = _table(retrieval.get("dense"), "retrieval.dense")
    reranker = _table(retrieval.get("reranker"), "retrieval.reranker")

    _require_exact_keys(
        paths,
        {
            "project_root",
            "source_manifest",
            "pdf_directory",
            "metadata_directory",
            "extracted_pages_directory",
            "chunks_directory",
            "embedding_model_directory",
            "reranker_model_directory",
            "qdrant_directory",
            "evaluation_candidates",
            "evaluation_review",
            "evaluation_gold",
            "evaluation_results_directory",
        },
        "paths",
    )
    _require_exact_keys(
        downloader,
        {"timeout_seconds", "chunk_size_bytes", "maximum_pdf_bytes", "user_agent"},
        "downloader",
    )
    _require_exact_keys(
        extraction, {"sort_blocks", "include_empty_pages"}, "extraction"
    )
    _require_exact_keys(
        chunking,
        {
            "target_model_tokens",
            "maximum_model_tokens",
            "overlap_model_tokens",
            "estimated_characters_per_token",
            "respect_page_boundaries",
            "preserve_article_boundaries",
        },
        "chunking",
    )
    _require_exact_keys(
        retrieval, {"rrf", "bm25", "dense", "reranker"}, "retrieval"
    )
    _require_exact_keys(
        rrf,
        {"rank_constant", "dense_candidates", "sparse_candidates", "fused_candidates"},
        "retrieval.rrf",
    )
    _require_exact_keys(
        bm25, {"k1", "b", "epsilon", "minimum_score", "top_k"}, "retrieval.bm25"
    )
    _require_exact_keys(
        dense,
        {
            "model_id",
            "model_revision",
            "model_sha256",
            "vector_size",
            "max_sequence_length",
            "batch_size",
            "query_prefix",
            "passage_prefix",
            "normalize_embeddings",
            "collection_name",
            "minimum_score",
        },
        "retrieval.dense",
    )
    _require_exact_keys(
        reranker,
        {
            "model_id",
            "model_revision",
            "model_sha256",
            "max_sequence_length",
            "batch_size",
            "candidate_count",
            "top_k",
            "zero_shot_turkish",
        },
        "retrieval.reranker",
    )

    config = ProjectConfig(
        schema_version=_integer(root, "schema_version", "root"),
        paths=PathsConfig(
            project_root=_string(paths, "project_root", "paths"),
            source_manifest=_string(paths, "source_manifest", "paths"),
            pdf_directory=_string(paths, "pdf_directory", "paths"),
            metadata_directory=_string(paths, "metadata_directory", "paths"),
            extracted_pages_directory=_string(
                paths, "extracted_pages_directory", "paths"
            ),
            chunks_directory=_string(paths, "chunks_directory", "paths"),
            embedding_model_directory=_string(
                paths, "embedding_model_directory", "paths"
            ),
            reranker_model_directory=_string(
                paths, "reranker_model_directory", "paths"
            ),
            qdrant_directory=_string(paths, "qdrant_directory", "paths"),
            evaluation_candidates=_string(
                paths, "evaluation_candidates", "paths"
            ),
            evaluation_review=_string(paths, "evaluation_review", "paths"),
            evaluation_gold=_string(paths, "evaluation_gold", "paths"),
            evaluation_results_directory=_string(
                paths, "evaluation_results_directory", "paths"
            ),
        ),
        downloader=DownloaderConfig(
            timeout_seconds=_integer(downloader, "timeout_seconds", "downloader"),
            chunk_size_bytes=_integer(downloader, "chunk_size_bytes", "downloader"),
            maximum_pdf_bytes=_integer(
                downloader, "maximum_pdf_bytes", "downloader"
            ),
            user_agent=_string(downloader, "user_agent", "downloader"),
        ),
        extraction=ExtractionConfig(
            sort_blocks=_boolean(extraction, "sort_blocks", "extraction"),
            include_empty_pages=_boolean(
                extraction, "include_empty_pages", "extraction"
            ),
        ),
        chunking=ChunkingConfig(
            target_model_tokens=_integer(chunking, "target_model_tokens", "chunking"),
            maximum_model_tokens=_integer(chunking, "maximum_model_tokens", "chunking"),
            overlap_model_tokens=_integer(chunking, "overlap_model_tokens", "chunking"),
            estimated_characters_per_token=_integer(
                chunking, "estimated_characters_per_token", "chunking"
            ),
            respect_page_boundaries=_boolean(
                chunking, "respect_page_boundaries", "chunking"
            ),
            preserve_article_boundaries=_boolean(
                chunking, "preserve_article_boundaries", "chunking"
            ),
        ),
        rrf=RRFConfig(
            rank_constant=_integer(rrf, "rank_constant", "retrieval.rrf"),
            dense_candidates=_integer(rrf, "dense_candidates", "retrieval.rrf"),
            sparse_candidates=_integer(rrf, "sparse_candidates", "retrieval.rrf"),
            fused_candidates=_integer(rrf, "fused_candidates", "retrieval.rrf"),
        ),
        bm25=BM25Config(
            k1=_number(bm25, "k1", "retrieval.bm25"),
            b=_number(bm25, "b", "retrieval.bm25"),
            epsilon=_number(bm25, "epsilon", "retrieval.bm25"),
            minimum_score=_number(bm25, "minimum_score", "retrieval.bm25"),
            top_k=_integer(bm25, "top_k", "retrieval.bm25"),
        ),
        dense=DenseConfig(
            model_id=_string(dense, "model_id", "retrieval.dense"),
            model_revision=_string(dense, "model_revision", "retrieval.dense"),
            model_sha256=_string(dense, "model_sha256", "retrieval.dense"),
            vector_size=_integer(dense, "vector_size", "retrieval.dense"),
            max_sequence_length=_integer(
                dense, "max_sequence_length", "retrieval.dense"
            ),
            batch_size=_integer(dense, "batch_size", "retrieval.dense"),
            query_prefix=_string(dense, "query_prefix", "retrieval.dense"),
            passage_prefix=_string(dense, "passage_prefix", "retrieval.dense"),
            normalize_embeddings=_boolean(
                dense, "normalize_embeddings", "retrieval.dense"
            ),
            collection_name=_string(dense, "collection_name", "retrieval.dense"),
            minimum_score=_number(dense, "minimum_score", "retrieval.dense"),
        ),
        reranker=RerankerConfig(
            model_id=_string(reranker, "model_id", "retrieval.reranker"),
            model_revision=_string(
                reranker, "model_revision", "retrieval.reranker"
            ),
            model_sha256=_string(
                reranker, "model_sha256", "retrieval.reranker"
            ),
            max_sequence_length=_integer(
                reranker, "max_sequence_length", "retrieval.reranker"
            ),
            batch_size=_integer(reranker, "batch_size", "retrieval.reranker"),
            candidate_count=_integer(
                reranker, "candidate_count", "retrieval.reranker"
            ),
            top_k=_integer(reranker, "top_k", "retrieval.reranker"),
            zero_shot_turkish=_boolean(
                reranker, "zero_shot_turkish", "retrieval.reranker"
            ),
        ),
    )
    config.validate()
    return config


def _table(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a TOML table")
    return value


def _require_exact_keys(
    table: Mapping[str, Any], expected: set[str], section: str
) -> None:
    actual = set(table)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ConfigError(f"missing setting(s) in {section}: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"unknown setting(s) in {section}: {', '.join(sorted(unknown))}")


def _integer(table: Mapping[str, Any], key: str, section: str) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def _boolean(table: Mapping[str, Any], key: str, section: str) -> bool:
    value = table[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be a boolean")
    return value


def _string(table: Mapping[str, Any], key: str, section: str) -> str:
    value = table[key]
    if not isinstance(value, str):
        raise ConfigError(f"{section}.{key} must be a string")
    return value


def _number(table: Mapping[str, Any], key: str, section: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{section}.{key} must be a number")
    return float(value)


def _resolve_within_project(project_root: Path, value: str, setting: str) -> Path:
    configured_path = Path(value)
    if configured_path.is_absolute():
        raise ConfigError(f"{setting} must be relative to paths.project_root")
    resolved = (project_root / configured_path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ConfigError(f"{setting} cannot escape paths.project_root") from exc
    return resolved
