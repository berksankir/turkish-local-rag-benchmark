# Provisional retrieval run — AI-generated candidates

Bu rapor, insan tarafından tek tek incelenip onaylanmamış 50 AI-generated
candidate kaydı üzerinde CPU ve local modellerle yapılan provisional teknik koşuyu
gösterir. Nihai gold benchmark değildir. Answerable kayıtların exact source
span/page bütünlüğü otomatik kontrol edilmiştir; bu kontrol insan onayı yerine geçmez.
Unanswerable kayıtların gerçekten corpus dışı olduğu insan tarafından
doğrulanmamıştır.

Unanswerable kayıtlar retrieval kalite paydalarına alınmamış, latency ölçümüne dahil edilmiştir.
MRR, ortak top-10 sonuç listesi üzerinde hesaplanmıştır.

## dev

| Pipeline | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5000 | 0.8750 | 0.8750 | 0.6875 | 0.7500 | 0.5000 | 47.087 |
| bm25 | 0.6250 | 1.0000 | 1.0000 | 0.8125 | 0.6250 | 0.6250 | 3.503 |
| hybrid_rrf | 0.7500 | 0.8750 | 0.8750 | 0.8304 | 0.7500 | 0.7500 | 41.013 |
| hybrid_reranked | 0.6250 | 0.8750 | 1.0000 | 0.7750 | 0.8750 | 0.6250 | 1752.881 |

## test

| Pipeline | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5625 | 0.8750 | 0.9375 | 0.7207 | 0.8125 | 0.5625 | 44.215 |
| bm25 | 0.6875 | 0.8750 | 0.9375 | 0.7956 | 0.9375 | 0.6875 | 2.345 |
| hybrid_rrf | 0.7188 | 0.9375 | 0.9688 | 0.8342 | 0.9375 | 0.7188 | 42.728 |
| hybrid_reranked | 0.8125 | 0.9688 | 1.0000 | 0.8917 | 0.9375 | 0.8125 | 1468.325 |

## all

| Pipeline | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5500 | 0.8750 | 0.9250 | 0.7140 | 0.8000 | 0.5500 | 44.789 |
| bm25 | 0.6750 | 0.9000 | 0.9500 | 0.7990 | 0.8750 | 0.6750 | 2.577 |
| hybrid_rrf | 0.7250 | 0.9250 | 0.9500 | 0.8334 | 0.9000 | 0.7250 | 42.385 |
| hybrid_reranked | 0.7750 | 0.9500 | 1.0000 | 0.8683 | 0.9250 | 0.7750 | 1525.236 |

## Reproducibility

- Timestamp: `2026-09-01T20:42:14.700510Z`
- Provisional AI candidate set SHA-256: `0599ae0c761db73e7512f4cd4eef0a2ced2d258319f127076a0949096c9aa268`
- Logical corpus SHA-256: `0b359c6b60ed955a3abbac5c00cf699c3cc1506868cb92c11ae8d909ddada504`
- Embedding revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Reranker revision: `1427fd652930e4ba29e8149678df786c240d8825`
- Benchmark süresi: `80.756 s`
- Peak process working set: `1390444544 byte`

Bu sonuçlar yalnız bu küçük corpus, makine çalıştırması ve AI-generated
candidate set için geçerlidir. Yalnızca retrieval pipeline'ının teknik olarak
çalıştığını gösteren provisional sonuçlar olarak yorumlanmalıdır. Reranker
Türkçe için zero-shot'tır; test split'ine göre ayar yapılmamıştır.
