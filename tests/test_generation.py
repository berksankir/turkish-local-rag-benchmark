from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from turkish_local_rag.config import load_config
from turkish_local_rag.generation import (
    ABSTAIN_ANSWER,
    GENERATOR_JSON_SCHEMA,
    MAX_GENERATOR_DEBUG_CHARS,
    GenerationError,
    GenerationTimeout,
    GeneratorOutputError,
    GroundedRAGService,
    LlamaCppServerGenerator,
    RetrievalExecution,
    build_prompts,
    parse_generator_output,
    validate_query_response,
)
from turkish_local_rag.retrieve import ChunkRecord, FusedHit


def _chunk(*, title: str = "Yönetmelik") -> ChunkRecord:
    return ChunkRecord(
        chunk_id="doc:p3:c1",
        document_id="doc",
        title=title,
        page_number=3,
        source_page_url="https://example.test/source",
        pdf_url="https://example.test/doc.pdf",
        pdf_sha256="a" * 64,
        source_block_ids=("doc:p3:b1",),
        text="Üniversitenin en yüksek karar organı Mütevelli Heyetidir.",
        estimated_tokens=20,
        token_count_method="test",
    )


def _hit(*, title: str = "Yönetmelik") -> FusedHit:
    return FusedHit(
        rank=1,
        rrf_score=2 / 61,
        component_ranks={"bm25": 1, "dense": 1},
        chunk=_chunk(title=title),
    )


class FakeGenerator:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    @property
    def metadata(self) -> dict[str, object]:
        return {"backend": "fake", "instance": id(self)}

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        value = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return str(value)


def _service(generator: FakeGenerator, *, hits: tuple[FusedHit, ...] | None = None):
    config = load_config("config/default.toml")
    selected = (_hit(),) if hits is None else hits

    def retrieve(question: str, pipeline: str) -> RetrievalExecution:
        return RetrievalExecution(selected, 4.0, 0.0)

    return GroundedRAGService(retrieve, generator, config.generation, config.evidence)


def test_grounded_answer_uses_only_trusted_citation_metadata() -> None:
    generator = FakeGenerator(
        ['{"answer":"Mütevelli Heyetidir.","supporting_context_ids":["C1"]}']
    )

    result = _service(generator).answer(
        "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
    )

    assert result["abstained"] is False
    assert result["answer"] == "Mütevelli Heyetidir."
    assert result["citations"] == [
        {
            "document_id": "doc",
            "title": "Yönetmelik",
            "physical_page": 3,
            "source_page_url": "https://example.test/source",
            "pdf_url": "https://example.test/doc.pdf",
            "chunk_id": "doc:p3:c1",
        }
    ]


def test_evidence_gate_abstains_without_calling_generator() -> None:
    generator = FakeGenerator(["unused"])

    result = _service(generator).answer("Bugünkü yemekhane menüsü nedir?", "hybrid_rrf")

    assert result["abstained"] is True
    assert result["answer"] == ABSTAIN_ANSWER
    assert result["abstention_reason"] == "query_coverage_below_threshold"
    assert result["citations"] == []
    assert generator.calls == 0


def test_empty_retrieval_abstains_without_generation() -> None:
    generator = FakeGenerator(["unused"])

    result = _service(generator, hits=()).answer("Bir soru", "hybrid_rrf")

    assert result["abstention_reason"] == "empty_retrieval"
    assert generator.calls == 0


def test_generator_timeout_returns_structured_abstention() -> None:
    generator = FakeGenerator([GenerationTimeout("slow")])

    result = _service(generator).answer(
        "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
    )

    assert result["abstention_reason"] == "generator_timeout"
    assert result["citations"] == []
    assert result["generation_error"] == {
        "category": "timeout",
        "validation_summary": "local generator exceeded its configured timeout",
        "debug_excerpt": None,
        "attempts": 1,
    }


def test_invalid_json_has_one_retry_then_abstains() -> None:
    generator = FakeGenerator(["not-json", "still-not-json"])

    result = _service(generator).answer(
        "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
    )

    assert result["abstention_reason"] == "generator_invalid_json_syntax"
    assert result["generation_error"]["category"] == "invalid_json_syntax"
    assert result["generation_error"]["attempts"] == 2
    assert generator.calls == 2


def test_missing_citation_metadata_is_rejected() -> None:
    generator = FakeGenerator(
        ['{"answer":"Mütevelli Heyetidir.","supporting_context_ids":["C1"]}']
    )

    with pytest.raises(GenerationError, match="missing trusted citation metadata"):
        _service(generator, hits=(_hit(title=""),)).answer(
            "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
        )


def test_invented_citation_is_rejected_and_never_serialized() -> None:
    generator = FakeGenerator(
        ['{"answer":"Yanıt.","supporting_context_ids":["C99"]}']
    )

    result = _service(generator).answer(
        "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
    )

    assert result["abstained"] is True
    assert result["citations"] == []
    assert "C99" not in json.dumps(result, ensure_ascii=False)
    assert result["generation_error"]["category"] == "invalid_context_id"


def test_turkish_unicode_serialization_is_not_ascii_escaped() -> None:
    generator = FakeGenerator(
        ['{"answer":"Öğrenim Türkçedir.","supporting_context_ids":["C1"]}']
    )
    result = _service(generator).answer(
        "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
    )

    serialized = json.dumps(result, ensure_ascii=False)

    assert "Öğrenim Türkçedir" in serialized
    assert "\\u00d6" not in serialized


def test_same_generator_instance_is_reused_across_queries() -> None:
    generator = FakeGenerator(
        ['{"answer":"Mütevelli Heyetidir.","supporting_context_ids":["C1"]}']
    )
    service = _service(generator)

    service.answer("Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf")
    service.answer("Üniversitenin karar organı hangisidir?", "hybrid_rrf")

    assert generator.calls == 2
    assert service._generator is generator


def test_prompt_builder_obeys_context_budget() -> None:
    first = _hit()
    second = replace(
        first,
        rank=2,
        chunk=replace(first.chunk, chunk_id="doc:p3:c2", estimated_tokens=2000),
    )

    _, prompt, selected = build_prompts(
        "Soru", [first, second], context_window_tokens=2048, max_output_tokens=128
    )

    assert [hit.chunk.chunk_id for hit in selected] == ["doc:p3:c1"]
    assert "doc:p3:c2" not in prompt


def test_response_validator_rejects_citation_not_in_retrieval() -> None:
    generator = FakeGenerator(
        ['{"answer":"Mütevelli Heyetidir.","supporting_context_ids":["C1"]}']
    )
    result = _service(generator).answer(
        "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
    )
    result["citations"][0]["chunk_id"] = "other:p1:c1"

    with pytest.raises(GenerationError, match="not backed"):
        validate_query_response(result)


def test_parser_rejects_extra_fields() -> None:
    with pytest.raises(GeneratorOutputError) as captured:
        parse_generator_output(
            '{"answer":"x","supporting_context_ids":["C1"],"citation":"fake"}',
            {"C1"},
        )
    assert captured.value.category == "missing_or_extra_fields"


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        ('{"answer":7,"supporting_context_ids":["C1"]}', "wrong_field_type"),
        ('{"answer":"  ","supporting_context_ids":["C1"]}', "empty_answer"),
        ('{"answer":"x","supporting_context_ids":[]}', "empty_supporting_context_list"),
        ('{"answer":"x","supporting_context_ids":["C1","C1"]}', "duplicate_context_id"),
        ('{"answer":"x","supporting_context_ids":["C2"]}', "invalid_context_id"),
    ],
)
def test_parser_classifies_structured_output_failures(
    payload: str, category: str
) -> None:
    with pytest.raises(GeneratorOutputError) as captured:
        parse_generator_output(payload, {"C1"})

    assert captured.value.category == category


def test_invalid_json_debug_excerpt_is_bounded() -> None:
    with pytest.raises(GeneratorOutputError) as captured:
        parse_generator_output("x" * 1000, {"C1"})

    assert captured.value.category == "invalid_json_syntax"
    assert captured.value.debug_excerpt is not None
    assert len(captured.value.debug_excerpt) == MAX_GENERATOR_DEBUG_CHARS


def test_runtime_error_is_fail_closed_and_safely_summarized() -> None:
    generator = FakeGenerator([GenerationError("secret runtime response body")])

    result = _service(generator).answer(
        "Üniversitenin en yüksek karar organı hangisidir?", "hybrid_rrf"
    )

    assert result["abstention_reason"] == "generator_runtime_api_error"
    assert result["generation_error"]["category"] == "runtime_api_error"
    assert "secret" not in json.dumps(result)


def test_b10621_request_uses_documented_direct_schema_shape(tmp_path: Path) -> None:
    config = load_config("config/default.toml")
    adapter = LlamaCppServerGenerator(
        tmp_path / "model.gguf",
        tmp_path / "llama-server.exe",
        config.generation,
    )

    body = adapter._request_body("system", "user")

    assert body["response_format"] == {
        "type": "json_schema",
        "schema": GENERATOR_JSON_SCHEMA,
    }
    assert "json_schema" not in body["response_format"]


def test_adapter_verifies_model_and_runtime_archive_hashes(tmp_path: Path) -> None:
    config = load_config("config/default.toml")
    runtime_root = tmp_path / "runtime" / "llama-test"
    executable = runtime_root / "bin" / "llama-server.exe"
    archive = runtime_root / "llama-test-bin-win-cpu-x64.zip"
    model = tmp_path / "model.gguf"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime executable")
    archive.write_bytes(b"runtime archive")
    model.write_bytes(b"model")
    settings = replace(
        config.generation,
        runtime_version="test",
        runtime_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        model_size_bytes=model.stat().st_size,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    )

    adapter = LlamaCppServerGenerator(model, executable, settings)

    adapter._verify_assets()


def test_adapter_rejects_changed_runtime_archive(tmp_path: Path) -> None:
    config = load_config("config/default.toml")
    runtime_root = tmp_path / "runtime" / "llama-test"
    executable = runtime_root / "bin" / "llama-server.exe"
    archive = runtime_root / "llama-test-bin-win-cpu-x64.zip"
    model = tmp_path / "model.gguf"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime executable")
    archive.write_bytes(b"changed archive")
    model.write_bytes(b"model")
    settings = replace(
        config.generation,
        runtime_version="test",
        runtime_sha256="0" * 64,
        model_size_bytes=model.stat().st_size,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    )

    adapter = LlamaCppServerGenerator(model, executable, settings)

    with pytest.raises(GenerationError, match="runtime SHA-256 mismatch"):
        adapter._verify_assets()
