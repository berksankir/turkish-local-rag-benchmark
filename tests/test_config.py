from pathlib import Path

import pytest

from turkish_local_rag.config import ConfigError, load_config


DEFAULT_CONFIG = Path("config/default.toml")


def test_default_config_is_valid() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.schema_version == 1
    assert config.paths.source_manifest == "data/manifest.json"
    assert config.downloader.timeout_seconds == 30
    assert config.extraction.sort_blocks is True
    assert config.extraction.include_empty_pages is False
    assert config.chunking.target_model_tokens == 384
    assert config.chunking.estimated_characters_per_token == 3
    assert config.chunking.respect_page_boundaries is True
    assert config.rrf.rank_constant == 60
    assert config.bm25.k1 == 1.5
    assert config.bm25.top_k == 20
    assert config.dense.vector_size == 384
    assert config.dense.query_prefix == "query: "
    assert config.reranker.candidate_count == 20


def test_paths_are_resolved_from_config_location() -> None:
    config = load_config(DEFAULT_CONFIG)
    paths = config.resolve_paths(DEFAULT_CONFIG)

    assert paths.project_root == Path.cwd().resolve()
    assert paths.source_manifest == Path("data/manifest.json").resolve()
    assert paths.pdf_directory == Path("data/pdfs").resolve()
    assert paths.extracted_pages_directory == Path("artifacts/extracted").resolve()
    assert paths.chunks_directory == Path("artifacts/chunks").resolve()
    assert paths.embedding_model_directory == Path(
        "models/multilingual-e5-small"
    ).resolve()
    assert paths.reranker_model_directory == Path(
        "models/mmarco-mMiniLMv2-L12-H384-v1"
    ).resolve()
    assert paths.qdrant_directory == Path("indexes/qdrant").resolve()
    assert paths.evaluation_candidates == Path(
        "evaluation/candidates.jsonl"
    ).resolve()
    assert paths.evaluation_review == Path("evaluation/review.csv").resolve()
    assert paths.evaluation_gold == Path("evaluation/gold.jsonl").resolve()
    assert paths.evaluation_results_directory == Path(
        "evaluation/results"
    ).resolve()


def test_page_boundary_protection_cannot_be_disabled(tmp_path: Path) -> None:
    changed = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "respect_page_boundaries = true", "respect_page_boundaries = false"
    )
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(changed, encoding="utf-8")

    with pytest.raises(ConfigError, match="respect_page_boundaries must remain true"):
        load_config(config_path)


def test_unknown_setting_is_rejected(tmp_path: Path) -> None:
    changed = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "target_model_tokens = 384", "target_model_tokens = 384\nunknown = 1"
    )
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(changed, encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown setting"):
        load_config(config_path)


def test_token_limits_are_validated(tmp_path: Path) -> None:
    changed = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "maximum_model_tokens = 448", "maximum_model_tokens = 300"
    )
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(changed, encoding="utf-8")

    with pytest.raises(ConfigError, match="maximum_model_tokens"):
        load_config(config_path)
