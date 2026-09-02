"""Grounded local generation, deterministic evidence gating, and trusted citations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from turkish_local_rag.config import EvidenceConfig, GeneratorConfig
from turkish_local_rag.retrieve import ChunkRecord, turkish_tokenize


OUTPUT_SCHEMA_VERSION = "1.1"
MAX_GENERATOR_DEBUG_CHARS = 240
ABSTAIN_ANSWER = "Yeterli kanıt bulunamadı."
GENERATOR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "supporting_context_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 1,
            "uniqueItems": True,
        },
    },
    "required": ["answer", "supporting_context_ids"],
    "additionalProperties": False,
}


class GenerationError(RuntimeError):
    """Raised when local generation or its validated output fails."""


class GenerationTimeout(GenerationError):
    """Raised when the local generator exceeds its configured timeout."""


class GeneratorOutputError(GenerationError):
    """A safely categorized generator-output validation failure."""

    def __init__(
        self,
        category: str,
        summary: str,
        *,
        debug_excerpt: str | None = None,
    ) -> None:
        super().__init__(summary)
        self.category = category
        self.summary = summary
        self.debug_excerpt = debug_excerpt


class GeneratorRuntimeError(GenerationError):
    """Raised for a local runtime or API response failure."""


class Generator(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class RetrievalExecution:
    hits: tuple[Any, ...]
    retrieval_latency_ms: float
    reranking_latency_ms: float
    embedding_metadata: Mapping[str, Any] | None = None
    reranker_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    sufficient: bool
    reason: str | None
    query_coverage: float
    top_retrieval_score: float
    evidence_score: float


@dataclass(frozen=True, slots=True)
class ParsedGeneration:
    answer: str
    supporting_context_ids: tuple[str, ...]


STOPWORDS = frozenset(
    {
        "acaba",
        "bir",
        "bu",
        "da",
        "de",
        "hangi",
        "ile",
        "için",
        "kaç",
        "mıdır",
        "nedir",
        "nelerdir",
        "ne",
        "olarak",
        "olan",
        "ve",
        "veya",
    }
)


def _lexical_keys(text: str) -> set[str]:
    """Use conservative prefixes to reduce common Turkish suffix mismatch."""

    return {
        token if len(token) < 6 else token[:6]
        for token in turkish_tokenize(text)
        if len(token) >= 3 and token not in STOPWORDS
    }


def evaluate_evidence(
    question: str, hits: Sequence[Any], settings: EvidenceConfig
) -> EvidenceDecision:
    """Apply a deterministic gate before any model call."""

    if not hits:
        return EvidenceDecision(False, "empty_retrieval", 0.0, 0.0, 0.0)
    selected = hits[: settings.context_top_k]
    query_terms = _lexical_keys(question)
    context_terms: set[str] = set()
    for hit in selected:
        context_terms.update(_lexical_keys(f"{hit.chunk.title}\n{hit.chunk.text}"))
    coverage = (
        len(query_terms & context_terms) / len(query_terms) if query_terms else 0.0
    )
    top_score = _retrieval_score(selected[0])
    normalized_score = min(1.0, top_score / (2.0 / 61.0))
    evidence_score = 0.65 * coverage + 0.35 * normalized_score
    if top_score < settings.minimum_rrf_score:
        return EvidenceDecision(
            False, "retrieval_score_below_threshold", coverage, top_score, evidence_score
        )
    if coverage < settings.minimum_query_coverage:
        return EvidenceDecision(
            False, "query_coverage_below_threshold", coverage, top_score, evidence_score
        )
    return EvidenceDecision(True, None, coverage, top_score, evidence_score)


def select_context_hits(
    question: str, hits: Sequence[Any], limit: int
) -> tuple[Any, ...]:
    """Reorder only the generation context, without changing retrieval results."""

    query_terms = _lexical_keys(question)

    def key(hit: Any) -> tuple[float, float, int, str]:
        terms = _lexical_keys(f"{hit.chunk.title}\n{hit.chunk.text}")
        coverage = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
        return (-coverage, -_retrieval_score(hit), int(hit.rank), hit.chunk.chunk_id)

    return tuple(sorted(hits, key=key)[:limit])


def build_prompts(
    question: str,
    hits: Sequence[Any],
    *,
    context_window_tokens: int,
    max_output_tokens: int,
) -> tuple[str, str, tuple[Any, ...]]:
    """Build a bounded Turkish prompt from trusted retrieved chunks only."""

    budget = context_window_tokens - max_output_tokens - 500
    if budget <= 0:
        raise GenerationError("configured context leaves no room for retrieved evidence")
    selected: list[Any] = []
    used = 0
    for hit in hits:
        estimated = max(1, int(hit.chunk.estimated_tokens))
        if selected and used + estimated > budget:
            break
        selected.append(hit)
        used += estimated
    if not selected:
        raise GenerationError("no retrieved chunk fits the context budget")
    context = "\n\n".join(
        (
            f"[CONTEXT C{index}]\n"
            f"Metin:\n{hit.chunk.text}"
        )
        for index, hit in enumerate(selected, start=1)
    )
    system_prompt = (
        "Sen yalnız verilen bağlamla yanıt veren bir Türkçe RAG bileşenisin. "
        "Bağlam dışı bilgi kullanma. Kısa ve doğrudan yanıt ver. Citation, URL, "
        "belge adı veya sayfa numarası uydurma. Çıktın yalnız iki alanlı geçerli JSON "
        "olsun: answer ve cevabı gerçekten destekleyen supporting_context_ids. "
        "supporting_context_ids yalnız C1, C2 gibi verilen CONTEXT etiketleri olsun; "
        "cevabı destekleyen en az sayıda, mümkünse tek context seç."
    )
    user_prompt = (
        f"SORU:\n{question}\n\nBAĞLAM:\n{context}\n\n"
        "answer alanına sorunun istediği bilgi/değeri yaz; belge başlığını yalnız soru "
        "açıkça belge adını soruyorsa kullan. Yalnız bağlamdan cevapla ve destekleyen "
        "CONTEXT etiketlerini seç."
    )
    return system_prompt, user_prompt, tuple(selected)


def parse_generator_output(text: str, allowed_context_ids: set[str]) -> ParsedGeneration:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeneratorOutputError(
            "invalid_json_syntax",
            f"JSON syntax error at line {exc.lineno}, column {exc.colno}",
            debug_excerpt=_safe_debug_excerpt(text),
        ) from exc
    if not isinstance(raw, dict):
        raise GeneratorOutputError(
            "wrong_field_type", "generator output must be a JSON object"
        )
    expected_fields = {"answer", "supporting_context_ids"}
    if set(raw) != expected_fields:
        missing = sorted(expected_fields - set(raw))
        extra = sorted(set(raw) - expected_fields)
        raise GeneratorOutputError(
            "missing_or_extra_fields",
            f"generator object fields differ: missing={missing}, extra={extra}",
        )
    answer = raw["answer"]
    context_ids = raw["supporting_context_ids"]
    if not isinstance(answer, str) or not isinstance(context_ids, list):
        raise GeneratorOutputError(
            "wrong_field_type",
            "answer must be a string and supporting_context_ids must be a list",
        )
    if not answer.strip():
        raise GeneratorOutputError("empty_answer", "answer must not be empty")
    if not context_ids:
        raise GeneratorOutputError(
            "empty_supporting_context_list",
            "supporting_context_ids must contain at least one context id",
        )
    if not all(isinstance(value, str) and value.strip() for value in context_ids):
        raise GeneratorOutputError(
            "wrong_field_type",
            "supporting_context_ids must contain non-empty strings",
        )
    if len(context_ids) != len(set(context_ids)):
        raise GeneratorOutputError(
            "duplicate_context_id", "supporting_context_ids must be unique"
        )
    invented = sorted(set(context_ids) - allowed_context_ids)
    if invented:
        raise GeneratorOutputError(
            "invalid_context_id",
            f"generator selected {len(invented)} untrusted context id(s)",
        )
    return ParsedGeneration(answer.strip(), tuple(context_ids))


def _safe_debug_excerpt(text: str) -> str | None:
    """Return a single-line, bounded diagnostic excerpt, never a full raw response."""

    compact = " ".join(text.split())
    if not compact:
        return None
    if len(compact) <= MAX_GENERATOR_DEBUG_CHARS:
        return compact
    return compact[: MAX_GENERATOR_DEBUG_CHARS - 1] + "…"


def _generation_error_payload(
    category: str,
    summary: str,
    *,
    attempts: int,
    debug_excerpt: str | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "validation_summary": summary,
        "debug_excerpt": debug_excerpt,
        "attempts": attempts,
    }


def build_citation(chunk: ChunkRecord) -> dict[str, Any]:
    required = {
        "document_id": chunk.document_id,
        "title": chunk.title,
        "physical_page": chunk.page_number,
        "source_page_url": chunk.source_page_url,
        "pdf_url": chunk.pdf_url,
        "chunk_id": chunk.chunk_id,
    }
    if any(value is None or value == "" for value in required.values()):
        raise GenerationError(f"missing trusted citation metadata: {chunk.chunk_id}")
    return required


class GroundedRAGService:
    """Orchestrate retrieval, evidence gate, generation, and trusted citations."""

    def __init__(
        self,
        retriever: Callable[[str, str], RetrievalExecution],
        generator: Generator,
        generation_settings: GeneratorConfig,
        evidence_settings: EvidenceConfig,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._generation = generation_settings
        self._evidence = evidence_settings

    def answer(self, question: str, pipeline: str) -> dict[str, Any]:
        if pipeline not in {"hybrid_rrf", "hybrid_reranked"}:
            raise GenerationError("pipeline must be hybrid_rrf or hybrid_reranked")
        if not question.strip():
            raise GenerationError("question cannot be empty")
        total_start = perf_counter()
        execution = self._retriever(question, pipeline)
        candidate_context = select_context_hits(
            question, execution.hits, self._evidence.context_top_k
        )
        decision = evaluate_evidence(question, candidate_context, self._evidence)
        generation_ms = 0.0
        if not decision.sufficient:
            return self._response(
                question,
                pipeline,
                execution,
                decision,
                answer=ABSTAIN_ANSWER,
                abstained=True,
                abstention_reason=decision.reason,
                citations=[],
                generation_error=None,
                generation_latency_ms=generation_ms,
                total_start=total_start,
            )

        system_prompt, user_prompt, context_hits = build_prompts(
            question,
            candidate_context,
            context_window_tokens=self._generation.context_window_tokens,
            max_output_tokens=self._generation.max_output_tokens,
        )
        context_by_id = {
            f"C{index}": hit for index, hit in enumerate(context_hits, start=1)
        }
        allowed = set(context_by_id)
        parsed: ParsedGeneration | None = None
        failure_reason = "generator_runtime_api_error"
        generation_error: dict[str, Any] | None = None
        generation_start = perf_counter()
        for attempt in range(self._generation.max_retries + 1):
            prompt = user_prompt
            if attempt:
                prompt += (
                    "\n\nÖnceki çıktı doğrulanamadı. Yalnız geçerli JSON üret; "
                    "başka açıklama veya markdown ekleme."
                )
            try:
                raw = self._generator.generate(system_prompt, prompt)
                parsed = parse_generator_output(raw, allowed)
                break
            except GenerationTimeout:
                failure_reason = "generator_timeout"
                generation_error = _generation_error_payload(
                    "timeout",
                    "local generator exceeded its configured timeout",
                    attempts=attempt + 1,
                )
                break
            except GeneratorOutputError as exc:
                failure_reason = f"generator_{exc.category}"
                generation_error = _generation_error_payload(
                    exc.category,
                    exc.summary,
                    attempts=attempt + 1,
                    debug_excerpt=exc.debug_excerpt,
                )
            except GenerationError:
                failure_reason = "generator_runtime_api_error"
                generation_error = _generation_error_payload(
                    "runtime_api_error",
                    "local generator runtime or API request failed",
                    attempts=attempt + 1,
                )
                break
        generation_ms = (perf_counter() - generation_start) * 1000.0
        if parsed is None:
            return self._response(
                question,
                pipeline,
                execution,
                decision,
                answer=ABSTAIN_ANSWER,
                abstained=True,
                abstention_reason=failure_reason,
                citations=[],
                generation_error=generation_error,
                generation_latency_ms=generation_ms,
                total_start=total_start,
            )
        citations = [
            build_citation(context_by_id[context_id].chunk)
            for context_id in parsed.supporting_context_ids
        ]
        return self._response(
            question,
            pipeline,
            execution,
            decision,
            answer=parsed.answer,
            abstained=False,
            abstention_reason=None,
            citations=citations,
            generation_error=None,
            generation_latency_ms=generation_ms,
            total_start=total_start,
        )

    def _response(
        self,
        question: str,
        pipeline: str,
        execution: RetrievalExecution,
        decision: EvidenceDecision,
        *,
        answer: str,
        abstained: bool,
        abstention_reason: str | None,
        citations: list[dict[str, Any]],
        generation_error: dict[str, Any] | None,
        generation_latency_ms: float,
        total_start: float,
    ) -> dict[str, Any]:
        retrieved = [_serialize_hit(hit) for hit in execution.hits]
        payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "question": question,
            "pipeline": pipeline,
            "answer": answer,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "citations": citations,
            "generation_error": generation_error,
            "retrieved_chunks": retrieved,
            "scores": {
                "query_coverage": decision.query_coverage,
                "top_retrieval_score": decision.top_retrieval_score,
                "evidence_score": decision.evidence_score,
                "minimum_query_coverage": self._evidence.minimum_query_coverage,
                "minimum_rrf_score": self._evidence.minimum_rrf_score,
            },
            "latency_ms": {
                "retrieval": execution.retrieval_latency_ms,
                "reranking": execution.reranking_latency_ms,
                "generation": generation_latency_ms,
                "total": (perf_counter() - total_start) * 1000.0,
            },
            "models": {
                "embedding": (
                    dict(execution.embedding_metadata)
                    if execution.embedding_metadata is not None
                    else None
                ),
                "reranker": (
                    dict(execution.reranker_metadata)
                    if execution.reranker_metadata is not None
                    else None
                ),
                "generator": dict(self._generator.metadata),
            },
        }
        validate_query_response(payload)
        return payload


def validate_query_response(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "question",
        "pipeline",
        "answer",
        "abstained",
        "abstention_reason",
        "citations",
        "generation_error",
        "retrieved_chunks",
        "scores",
        "latency_ms",
        "models",
    }
    if set(payload) != expected or payload.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise GenerationError(
            f"query response does not match schema version {OUTPUT_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("question"), str) or not payload["question"].strip():
        raise GenerationError("query response question is invalid")
    if payload.get("pipeline") not in {"hybrid_rrf", "hybrid_reranked"}:
        raise GenerationError("query response pipeline is invalid")
    if not isinstance(payload.get("answer"), str) or not payload["answer"].strip():
        raise GenerationError("query response answer is invalid")
    if not isinstance(payload.get("abstained"), bool):
        raise GenerationError("query response abstained must be boolean")
    if payload["abstained"]:
        if not isinstance(payload.get("abstention_reason"), str) or payload["citations"]:
            raise GenerationError("abstained response requires a reason and no citations")
    elif payload.get("abstention_reason") is not None or not payload["citations"]:
        raise GenerationError("successful response requires citations and no abstention reason")
    generation_error = payload.get("generation_error")
    if (
        payload["abstained"]
        and str(payload["abstention_reason"]).startswith("generator_")
        and generation_error is None
    ):
        raise GenerationError("generator abstention requires generation_error details")
    if (
        payload["abstained"]
        and not str(payload["abstention_reason"]).startswith("generator_")
        and generation_error is not None
    ):
        raise GenerationError("evidence abstention must not contain generation_error")
    if generation_error is not None:
        if not payload["abstained"] or not str(payload["abstention_reason"]).startswith(
            "generator_"
        ):
            raise GenerationError("generation_error is only valid for generator abstention")
        if set(generation_error) != {
            "category",
            "validation_summary",
            "debug_excerpt",
            "attempts",
        }:
            raise GenerationError("generation_error diagnostic has invalid fields")
        if generation_error["category"] not in {
            "invalid_json_syntax",
            "missing_or_extra_fields",
            "wrong_field_type",
            "empty_answer",
            "empty_supporting_context_list",
            "duplicate_context_id",
            "invalid_context_id",
            "timeout",
            "runtime_api_error",
        }:
            raise GenerationError("generation_error category is invalid")
        if not isinstance(generation_error["validation_summary"], str):
            raise GenerationError("generation_error summary is invalid")
        excerpt = generation_error["debug_excerpt"]
        if excerpt is not None and (
            not isinstance(excerpt, str) or len(excerpt) > MAX_GENERATOR_DEBUG_CHARS
        ):
            raise GenerationError("generation_error debug excerpt is invalid")
        if not isinstance(generation_error["attempts"], int) or generation_error[
            "attempts"
        ] < 1:
            raise GenerationError("generation_error attempts is invalid")
    if not isinstance(payload.get("retrieved_chunks"), list):
        raise GenerationError("retrieved_chunks must be a list")
    trusted = {item.get("chunk_id") for item in payload["retrieved_chunks"]}
    for citation in payload["citations"]:
        if set(citation) != {
            "document_id",
            "title",
            "physical_page",
            "source_page_url",
            "pdf_url",
            "chunk_id",
        }:
            raise GenerationError("citation has missing or unknown metadata")
        if citation["chunk_id"] not in trusted:
            raise GenerationError("citation is not backed by a retrieved chunk")
    if set(payload.get("latency_ms", {})) != {
        "retrieval",
        "reranking",
        "generation",
        "total",
    }:
        raise GenerationError("latency_ms is invalid")


class LlamaCppServerGenerator:
    """Persistent llama-server adapter using only the Python standard library."""

    def __init__(
        self,
        model_path: str | Path,
        executable_path: str | Path,
        settings: GeneratorConfig,
        *,
        opener: Callable[..., Any] = urlopen,
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.model_path = Path(model_path)
        self.executable_path = Path(executable_path)
        self.settings = settings
        self._opener = opener
        self._process_factory = process_factory
        self._process: Any | None = None
        self.start_count = 0

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": self.settings.backend,
            "model_id": self.settings.model_id,
            "revision": self.settings.model_revision,
            "sha256": self.settings.model_sha256,
            "model_file": self.model_path.name,
            "model_size_bytes": self.settings.model_size_bytes,
            "quantization": "Q4_K_M",
            "runtime": f"llama.cpp-{self.settings.runtime_version}",
            "runtime_sha256": self.settings.runtime_sha256,
            "runtime_size_bytes": self.settings.runtime_size_bytes,
            "context_window_tokens": self.settings.context_window_tokens,
            "max_output_tokens": self.settings.max_output_tokens,
            "seed": self.settings.seed,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
            "cpu_threads": self.settings.cpu_threads,
        }

    @property
    def process_id(self) -> int | None:
        return int(self._process.pid) if self._process is not None else None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._verify_assets()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        command = [
            str(self.executable_path),
            "--model",
            str(self.model_path),
            "--host",
            self.settings.server_host,
            "--port",
            str(self.settings.server_port),
            "--ctx-size",
            str(self.settings.context_window_tokens),
            "--threads",
            str(self.settings.cpu_threads),
            "--parallel",
            "1",
            "--device",
            "none",
            "--gpu-layers",
            "0",
            "--offline",
            "--reasoning",
            "off",
            "--no-webui",
            "--log-disable",
        ]
        self._process = self._process_factory(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self.start_count += 1
        deadline = time.monotonic() + self.settings.timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise GenerationError("llama-server exited during model loading")
            try:
                request = Request(self._base_url + "/health", method="GET")
                with self._opener(request, timeout=1) as response:
                    if getattr(response, "status", 200) == 200:
                        return
            except (OSError, HTTPError, URLError):
                time.sleep(0.1)
        self.close()
        raise GenerationTimeout("llama-server model loading timed out")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.start()
        body = self._request_body(system_prompt, user_prompt)
        request = Request(
            self._base_url + "/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.settings.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise GenerationTimeout("llama-server generation timed out") from exc
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            raise GeneratorRuntimeError("llama-server request failed") from exc
        try:
            return raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GeneratorRuntimeError(
                "llama-server response is missing message content"
            ) from exc

    def _request_body(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Build the b10621-compatible chat-completions request body."""

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
            "seed": self.settings.seed,
            "stream": False,
            "response_format": {"type": "json_object"},
            "json_schema": GENERATOR_JSON_SCHEMA,
        }

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None

    @property
    def _base_url(self) -> str:
        return f"http://{self.settings.server_host}:{self.settings.server_port}"

    def _verify_assets(self) -> None:
        if not self.executable_path.is_file():
            raise GenerationError(
                f"llama.cpp runtime not found: {self.executable_path}"
            )
        runtime_archive = (
            self.executable_path.parent.parent
            / f"llama-{self.settings.runtime_version}-bin-win-cpu-x64.zip"
        )
        if not runtime_archive.is_file():
            raise GenerationError(
                f"llama.cpp runtime archive not found: {runtime_archive}"
            )
        runtime_hash = _sha256_file(runtime_archive)
        if runtime_hash != self.settings.runtime_sha256:
            raise GenerationError(
                "llama.cpp runtime SHA-256 mismatch: "
                f"expected={self.settings.runtime_sha256}, actual={runtime_hash}"
            )
        if not self.model_path.is_file():
            raise GenerationError(
                f"generator model not found: {self.model_path}; download it first"
            )
        actual_size = self.model_path.stat().st_size
        if actual_size != self.settings.model_size_bytes:
            raise GenerationError(
                "generator model size mismatch: "
                f"expected={self.settings.model_size_bytes}, actual={actual_size}"
            )
        actual_hash = _sha256_file(self.model_path)
        if actual_hash != self.settings.model_sha256:
            raise GenerationError(
                "generator model SHA-256 mismatch: "
                f"expected={self.settings.model_sha256}, actual={actual_hash}"
            )


def _retrieval_score(hit: Any) -> float:
    for attribute in ("rrf_score", "retrieval_score", "score"):
        value = getattr(hit, attribute, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _serialize_hit(hit: Any) -> dict[str, Any]:
    chunk = hit.chunk
    return {
        "rank": int(hit.rank),
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "physical_page": chunk.page_number,
        "source_page_url": chunk.source_page_url,
        "pdf_url": chunk.pdf_url,
        "pdf_sha256": chunk.pdf_sha256,
        "retrieval_score": _retrieval_score(hit),
        "reranker_score": (
            float(hit.reranker_score) if hasattr(hit, "reranker_score") else None
        ),
        "text": chunk.text,
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
