# Faz 8.1 reranker profiling — silver dev

Bu profiling AI-assisted silver benchmark'ın yalnız `dev` split'i üzerinde çalıştırılmıştır; gold veya tamamen human-reviewed değildir.
`human_reviewed=false`; test split ayar seçimi için okunmamıştır.
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
