# AI-assisted silver grounded-generation benchmark

## Dataset provenance

The benchmark uses an AI-assisted silver evaluation set. Its release and use were approved by the project owner after automated grounding checks and a human audit of 20 out of 50 records. The complete dataset was not reviewed item by item and is not presented as a human-reviewed gold set.

Benchmark, AI destekli bir silver evaluation seti kullanmaktadır. Veri setinin yayımlanmasına ve benchmarkta kullanılmasına, otomatik grounding kontrolleri ve 50 kaydın 20’si üzerinde yapılan insan audit’i sonrasında proje sahibi tarafından onay verilmiştir. Kayıtların tamamı tek tek insan incelemesinden geçmemiştir ve veri seti human-reviewed gold set olarak sunulmamaktadır.

`human_reviewed=false` ve `all_records_human_reviewed=false`, yalnızca 50 kaydın tamamının item-level insan incelemesinden geçmediğini belirtir; dataset-level yayımlama ve benchmark kullanım onayının bulunmadığı anlamına gelmez.

Machine-readable scope: `creation_method=ai_assisted`, `dataset_release_approved=true`, `approved_by=berksankir`, `approval_scope=dataset_level_with_sample_audit`, audit `20/50`, `final_gold=false`.

Test split model, pipeline veya threshold seçimi için kullanılmamıştır. LLM-as-a-judge yoktur.

| Split | Pipeline | R@1 | R@3 | R@5 | MRR | Citation | Coverage | Correct abstain | False abstain | Token F1 | Key facts | Mean total ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | hybrid_rrf | 0.750 | 0.875 | 0.875 | 0.830 | 0.750 | 1.000 | 1.000 | 0.000 | 0.263 | 0.333 | 11773.5 |
| dev | hybrid_reranked | 0.625 | 0.875 | 1.000 | 0.775 | 0.857 | 0.875 | 1.000 | 0.125 | 0.215 | 0.428 | 10045.8 |
| test | hybrid_rrf | 0.719 | 0.938 | 0.969 | 0.834 | 0.655 | 0.906 | 0.625 | 0.094 | 0.464 | 0.406 | 16267.0 |
| test | hybrid_reranked | 0.812 | 0.969 | 1.000 | 0.892 | 0.643 | 0.875 | 0.375 | 0.125 | 0.453 | 0.390 | 8687.4 |
| all | hybrid_rrf | 0.725 | 0.925 | 0.950 | 0.833 | 0.676 | 0.925 | 0.700 | 0.075 | 0.420 | 0.390 | 15368.3 |
| all | hybrid_reranked | 0.775 | 0.950 | 1.000 | 0.868 | 0.686 | 0.875 | 0.500 | 0.125 | 0.406 | 0.397 | 8959.1 |

## Latency (all split, milliseconds)

| Pipeline | Stage | Mean | p50 | p95 |
|---|---|---:|---:|---:|
| hybrid_rrf | retrieval | 114.7 | 121.9 | 173.5 |
| hybrid_rrf | reranking | 0.0 | 0.0 | 0.0 |
| hybrid_rrf | generation | 15245.7 | 15154.4 | 28507.5 |
| hybrid_rrf | total | 15368.3 | 15281.3 | 28558.6 |
| hybrid_reranked | retrieval | 65.0 | 43.2 | 164.7 |
| hybrid_reranked | reranking | 1809.8 | 1389.3 | 3355.3 |
| hybrid_reranked | generation | 6864.8 | 4129.0 | 21427.7 |
| hybrid_reranked | total | 8959.1 | 6155.0 | 24578.0 |

Generator initialization: 9.913 s; benchmark: 1216.431 s; generator start count: 1.
Approximate peak process-tree RAM: 2669043712 bytes.

Latency alanları retrieval, reranking, generation ve total olarak ayrı ölçülmüştür. Citation ve cevap metrikleri deterministic karşılaştırmalardır; semantik judge kullanılmaz.
Pipeline'lar ardışık çalıştırıldığı, response uzunlukları değiştiği ve cache ısındığı için generation latency farkı doğrudan reranker hız etkisi olarak yorumlanamaz. Her pipeline'da beş model çıktısı `generator_invalid_json` nedeniyle fail-closed abstention olmuştur.
