# Final error analysis

This analysis uses the real Phase 10 grounded-generation run in
[`evaluation/results/silver/generation_benchmark.json`](../evaluation/results/silver/generation_benchmark.json).
It contains 50 frozen silver records evaluated once with each pipeline (100 query
runs). The Phase 8 baseline remains unchanged under
[`phase8_baseline/`](../evaluation/results/silver/phase8_baseline/).

## Evaluation boundary

The benchmark is an AI-assisted silver set, not gold. Its release and benchmark
use were approved by the project owner after automated answerable-span/page checks
and an item-level human audit of 20 out of 50 records. The other 30 records were
not reviewed item by item. In machine-readable terms:
`creation_method=ai_assisted`, `dataset_release_approved=true`,
`approved_by=berksankir`,
`approval_scope=dataset_level_with_sample_audit`,
`all_records_human_reviewed=false`, audit `20/50`, and `final_gold=false`.
The test split was not used to select a model, prompt, pipeline, or threshold.

## Aggregate comparison

| Metric (all 50 records) | Hybrid RRF | Hybrid reranked |
|---|---:|---:|
| Recall@1 / @3 / @5 (40 answerable) | 0.725 / 0.925 / 0.950 | 0.775 / 0.950 / 1.000 |
| MRR | 0.833 | 0.868 |
| Correct document / page at rank 1 | 0.900 / 0.725 | 0.925 / 0.775 |
| Citation accuracy on generated answerable responses | 0.676 | 0.686 |
| Answerable coverage | 0.925 | 0.875 |
| Correct abstention (10 unanswerable) | 0.700 | 0.500 |
| False abstention (40 answerable) | 0.075 | 0.125 |
| Turkish token-level F1 | 0.420 | 0.406 |
| Deterministic key-fact coverage | 0.390 | 0.397 |

Reranking improves retrieval recall, MRR, and rank-1 document/page accuracy in
this run. It does not improve the end-to-end abstention trade-off: five of ten
unanswerable questions are answered, versus three with RRF, while false
abstention rises from three to five of forty answerable questions. The small
zero-shot reranker therefore remains optional; `hybrid_rrf` remains the default.

## Error taxonomy

Counts below are query-run counts, not unique questions. Categories can overlap;
for example, a wrong-page top result can also lead to a citation mismatch.

| Error type | Hybrid RRF | Hybrid reranked | Interpretation |
|---|---:|---:|---|
| Retrieval miss at rank 5 | 2/40 | 0/40 | RRF misses candidates 015 and 016 within the first five; reranking recovers both. |
| Correct document, wrong physical page at rank 1 | 7/40 | 6/40 | Document-level retrieval is materially easier than page-precise evidence retrieval. |
| Evidence-gate false abstention | 0/40 | 0/40 | No answerable query was rejected by the deterministic gate in this run. Reported false abstentions came from generator validation failures. |
| Unanswerable question answered | 3/10 | 5/10 | RRF incorrectly answers 043, 046, 048; reranked also answers 045 and 047. This is the clearest current abstention weakness. |
| Generator/schema failure | 5/50 | 5/50 | Ten outputs fail closed: seven invalid JSON syntax and three invalid/untrusted context IDs. |
| Citation mismatch | 12/37 | 11/35 | The strict metric requires the trusted cited document/page and the normalized exact source span in that cited chunk. |
| Token F1 below 0.50 | 23/37 | 23/35 | Many answers are plausible but lexically incomplete or differently phrased relative to the silver reference. |
| Incomplete key-fact coverage (<1.0) | 36/37 | 34/35 | Short generation frequently omits one or more deterministic reference facts. |

### Generator/schema failures

The parser now distinguishes invalid JSON syntax, missing/extra fields, wrong
types, empty answer, empty support list, duplicate context IDs, invalid context
IDs, timeout, and runtime/API errors. Only two categories occurred in the real
run:

- `invalid_json_syntax` (7): RRF 032, 033, 045, 047; reranked 019, 032, 033.
- `invalid_context_id` (3): RRF 004; reranked 004, 035.

All ten remain `Yeterli kanıt bulunamadı.` responses with no citation. The
artifact stores a bounded validation diagnostic (maximum 240 characters), never
an unlimited raw model response. The b10621 top-level `json_schema` request fixes
the previously undocumented request mismatch and passes a real dev smoke query,
but it does not eliminate content-length or context-selection failures.

### Retrieval and citation

RRF's rank-5 misses are candidates 015 and 016. Rank-1 correct-document/wrong-page
cases are 002, 005, 021, 024, 033, 034, 037 for RRF and 005, 016, 024, 026, 032,
037 for reranked. Citation accuracy remains about 0.68 for both pipelines. Trusted
metadata prevents fabricated document/page/URL fields from entering a response,
but it cannot make an irrelevant or insufficient retrieved chunk support the
answer. That distinction explains why citation integrity and citation accuracy
are separate properties.

### Answer quality

The real RRF success for candidate 001 answers `Mütevelli Heyet` with trusted
chunk `sabanci-ana-yonetmeligi:p2:c5`, but its token F1 is 0.50 and key-fact
coverage 0.333 because the short output omits wording present in the reference
fact. This illustrates both the value and the limitation of deterministic lexical
metrics. No LLM-as-a-judge score is used.

## Latency and memory

| Pipeline | Retrieval mean / p95 | Reranking mean / p95 | Generation mean / p95 | Total mean / p95 |
|---|---:|---:|---:|---:|
| Hybrid RRF | 60.5 / 99.8 ms | 0 / 0 ms | 15.26 / 30.12 s | 15.33 / 30.18 s |
| Hybrid reranked | 50.7 / 67.8 ms | 1.59 / 2.20 s | 6.33 / 17.88 s | 8.14 / 19.70 s |

The two pipelines were evaluated sequentially, with different response lengths
and cache warmth. The lower observed reranked generation time is therefore not a
causal speed benefit from reranking. Independent Phase 8 profiling measured warm
RRF retrieval at 23.8 ms and configured reranking alone at 963.5 ms on dev. The
Phase 10 run took 1,173.2 seconds after initialization and measured an approximate
peak process-tree RSS of 2,928,771,072 bytes (2.73 GiB). CPU generation remains a
few seconds to more than 30 seconds per called query.

## Conclusions and unresolved limitations

- Retrieval quality improves with reranking, but abstention behavior worsens and
  CPU latency increases.
- The deterministic evidence gate avoids model calls for clear misses, yet it is
  not sufficient to prevent all answers to corpus-external questions.
- Structured generation fails closed, but 10% of query runs still fail validation.
- Trusted citations prevent the model from inventing metadata; citation relevance
  remains only about 68% under the strict span-aware metric.
- Short-answer lexical F1 and key-fact coverage remain low. These results are not
  repaired by tuning on the test split and should be treated as open limitations.
- Conclusions are limited by a small, AI-assisted silver dataset and one CPU-only
  machine run.
