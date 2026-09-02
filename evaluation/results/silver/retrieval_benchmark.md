# Silver retrieval benchmark

Bu rapor AI-generated silver evaluation set üzerinde üretilmiştir. Answerable span/page kontrolleri otomatiktir ve insan onayı değildir.
Human audit durumu: approved=20, needs_changes=0, pending=0, rejected=0.
Unanswerable kayıtlar retrieval kalite paydalarına alınmamış, latency ölçümüne dahil edilmiştir.
MRR, ortak top-10 sonuç listesi üzerinde hesaplanmıştır.

## dev

| Pipeline | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5000 | 0.8750 | 0.8750 | 0.6875 | 0.7500 | 0.5000 | 90.173 |
| bm25 | 0.6250 | 1.0000 | 1.0000 | 0.8125 | 0.6250 | 0.6250 | 3.198 |
| hybrid_rrf | 0.7500 | 0.8750 | 0.8750 | 0.8304 | 0.7500 | 0.7500 | 91.099 |
| hybrid_reranked | 0.6250 | 0.8750 | 1.0000 | 0.7750 | 0.8750 | 0.6250 | 4408.911 |

## test

| Pipeline | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5625 | 0.8750 | 0.9375 | 0.7207 | 0.8125 | 0.5625 | 85.921 |
| bm25 | 0.6875 | 0.8750 | 0.9375 | 0.7956 | 0.9375 | 0.6875 | 3.140 |
| hybrid_rrf | 0.7188 | 0.9375 | 0.9688 | 0.8342 | 0.9375 | 0.7188 | 88.256 |
| hybrid_reranked | 0.8125 | 0.9688 | 1.0000 | 0.8917 | 0.9375 | 0.8125 | 4482.092 |

## all

| Pipeline | R@1 | R@3 | R@5 | MRR | Doc@1 | Page@1 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5500 | 0.8750 | 0.9250 | 0.7140 | 0.8000 | 0.5500 | 86.771 |
| bm25 | 0.6750 | 0.9000 | 0.9500 | 0.7990 | 0.8750 | 0.6750 | 3.151 |
| hybrid_rrf | 0.7250 | 0.9250 | 0.9500 | 0.8334 | 0.9000 | 0.7250 | 88.824 |
| hybrid_reranked | 0.7750 | 0.9500 | 1.0000 | 0.8683 | 0.9250 | 0.7750 | 4467.455 |

## Reproducibility

- Timestamp: `2026-09-02T08:50:16.475182Z`
- Dataset kind: `silver`
- Evaluation set SHA-256: `0599ae0c761db73e7512f4cd4eef0a2ced2d258319f127076a0949096c9aa268`
- Review/audit artifact SHA-256: `3cc487046596ab31ad7950b3eb115d360d502d693cc4c4093e8cfb82a60f684a`
- Logical corpus SHA-256: `0b359c6b60ed955a3abbac5c00cf699c3cc1506868cb92c11ae8d909ddada504`
- Embedding revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Reranker revision: `1427fd652930e4ba29e8149678df786c240d8825`
- Benchmark süresi: `232.319 s`
- Peak process working set: `1284526080 byte`

Bu sonuçlar yalnız bu küçük corpus ve makine çalıştırması için geçerlidir. Reranker Türkçe için zero-shot'tır; test split'ine göre ayar yapılmamıştır.
