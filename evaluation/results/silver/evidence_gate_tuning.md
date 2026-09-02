# Silver dev evidence-gate tuning

## Dataset provenance

The benchmark uses an AI-assisted silver evaluation set. Its release and use were approved by the project owner after automated grounding checks and a human audit of 20 out of 50 records. The complete dataset was not reviewed item by item and is not presented as a human-reviewed gold set.

Benchmark, AI destekli bir silver evaluation seti kullanmaktadır. Veri setinin yayımlanmasına ve benchmarkta kullanılmasına, otomatik grounding kontrolleri ve 50 kaydın 20’si üzerinde yapılan insan audit’i sonrasında proje sahibi tarafından onay verilmiştir. Kayıtların tamamı tek tek insan incelemesinden geçmemiştir ve veri seti human-reviewed gold set olarak sunulmamaktadır.

`human_reviewed=false` ve `all_records_human_reviewed=false`, yalnızca 50 kaydın tamamının item-level insan incelemesinden geçmediğini belirtir; dataset-level yayımlama ve benchmark kullanım onayının bulunmadığı anlamına gelmez.

Machine-readable scope: `creation_method=ai_assisted`, `dataset_release_approved=true`, `approved_by=berksankir`, `approval_scope=dataset_level_with_sample_audit`, audit `20/50`, `final_gold=false`.

Bu çalışma yalnız silver `dev` split üzerindedir.
Test split threshold seçimi sırasında okunmamıştır.

| Variant | Min coverage | Min RRF | Answerable coverage | Correct abstention | False abstention | Balanced |
|---|---:|---:|---:|---:|---:|---:|
| coverage_020_score_020 | 0.200 | 0.020 | 1.000 | 0.000 | 0.000 | 0.500 |
| coverage_030_score_020 | 0.300 | 0.020 | 1.000 | 0.500 | 0.000 | 0.750 |
| coverage_040_score_020 | 0.400 | 0.020 | 1.000 | 1.000 | 0.000 | 1.000 |
| coverage_030_score_024 | 0.300 | 0.024 | 1.000 | 0.500 | 0.000 | 0.750 |

Seçim: `coverage_040_score_020`. Bu küçük dev set üzerinde yapılmış provisional bir threshold seçimidir; gold veya tamamen human-reviewed sonuç değildir.
