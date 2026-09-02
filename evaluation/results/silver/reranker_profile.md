# Faz 8.1 reranker profiling — silver dev

## Dataset provenance

The benchmark uses an AI-assisted silver evaluation set. Its release and use were approved by the project owner after automated grounding checks and a human audit of 20 out of 50 records. The complete dataset was not reviewed item by item and is not presented as a human-reviewed gold set.

Benchmark, AI destekli bir silver evaluation seti kullanmaktadır. Veri setinin yayımlanmasına ve benchmarkta kullanılmasına, otomatik grounding kontrolleri ve 50 kaydın 20’si üzerinde yapılan insan audit’i sonrasında proje sahibi tarafından onay verilmiştir. Kayıtların tamamı tek tek insan incelemesinden geçmemiştir ve veri seti human-reviewed gold set olarak sunulmamaktadır.

`human_reviewed=false` ve `all_records_human_reviewed=false`, yalnızca 50 kaydın tamamının item-level insan incelemesinden geçmediğini belirtir; dataset-level yayımlama ve benchmark kullanım onayının bulunmadığı anlamına gelmez.

Machine-readable scope: `creation_method=ai_assisted`, `dataset_release_approved=true`, `approved_by=berksankir`, `approval_scope=dataset_level_with_sample_audit`, audit `20/50`, `final_gold=false`.

Bu profiling yalnız `dev` split üzerinde çalıştırılmış; test split ayar seçimi için okunmamıştır.
Reranker model instance'ı bir kez yüklenmiş ve tüm sorgular ile varyantlarda tekrar kullanılmıştır: 1 instance.

## Model yükleme ve cold/warm latency

- corpus: `15.615 ms`
- bm25: `55.059 ms`
- embedding_model: `3912.677 ms`
- reranker_model: `3595.617 ms`
- qdrant: `162.334 ms`
- Cold hybrid_rrf: `327.723 ms`
- Cold reranking-only: `1803.151 ms`
- Cold hybrid_reranked total: `2130.875 ms`

## Dev karşılaştırması

| Pipeline/varyant | top-n | batch | threads | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Rerank ms | Total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_rrf | — | — | — | 0.7500 | 0.8750 | 0.8750 | 0.8304 | 0.7500 | 0.7500 | — | 23.849 |
| configured_top20_b4_t4 | 20 | 4 | 4 | 0.6250 | 0.8750 | 1.0000 | 0.7750 | 0.8750 | 0.6250 | 963.489 | 1017.725 |
| bounded_top10_b4_t4 | 10 | 4 | 4 | 0.5000 | 0.8750 | 1.0000 | 0.7125 | 0.7500 | 0.5000 | 530.512 | 584.748 |
| bounded_top10_b2_t4 | 10 | 2 | 4 | 0.5000 | 0.8750 | 1.0000 | 0.7125 | 0.7500 | 0.5000 | 509.116 | 563.353 |
| bounded_top10_b4_t2 | 10 | 4 | 2 | 0.5000 | 0.8750 | 1.0000 | 0.7125 | 0.7500 | 0.5000 | 754.290 | 808.527 |

## Karar

Varsayılan hızlı pipeline `hybrid_rrf`; opsiyonel kalite modu `hybrid_reranked` olarak kalır.
Configured reranker dev farkları: R@1 `-0.1250`, R@5 `+0.1250`, MRR `-0.0554`, Doc@1 `+0.1250`, Page@1 `-0.1250`.
Reranking bazı metrikleri iyileştirebilirken diğerlerini düşürür ve belirgin CPU latency maliyeti ekler; test split bu kararda kullanılmamıştır.

## Runtime

- Toplam süre: `46.376 s`
- Peak process RAM: `1368723456 byte`
- Chunk/Qdrant point: `436` / `436`
