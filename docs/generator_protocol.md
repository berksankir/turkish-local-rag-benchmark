# Generator protocol lock

The local generator runtime is pinned to `llama.cpp` release `b10621`
(`0.3.0-dev`, commit `c1d0e7a00`). Its server documentation defines the
OpenAI-compatible chat endpoint as `/v1/chat/completions` and, for schema-constrained
JSON, documents the following request shape:

```json
{
  "response_format": {
    "type": "json_schema",
    "schema": { "type": "object" }
  }
}
```

Source: [llama.cpp b10621 server README](https://github.com/ggml-org/llama.cpp/blob/b10621/tools/server/README.md#post-v1chatcompletions).

The Phase 8 implementation used the newer nested `json_schema` wrapper. Phase 10
aligns the request with the fixed b10621 documentation. A networkless behavioral
test asserts the exact request body, and real smoke/evaluation runs exercise the
pinned binary. Prompt text, model, quantization, retrieval, evidence thresholds,
sampling values, and frozen silver splits are unchanged.

Generator output errors are fail-closed. The query response records a category,
short validation summary, attempt count, and at most 240 characters of whitespace-
normalized debug text. Full raw model responses are never stored in benchmark
artifacts.
