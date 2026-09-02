# Silver dev evidence-gate tuning

Bu çalışma yalnız AI-assisted silver `dev` split üzerindedir; `human_reviewed=false`. 
Test split threshold seçimi sırasında okunmamıştır.

| Variant | Min coverage | Min RRF | Answerable coverage | Correct abstention | False abstention | Balanced |
|---|---:|---:|---:|---:|---:|---:|
| coverage_020_score_020 | 0.200 | 0.020 | 1.000 | 0.000 | 0.000 | 0.500 |
| coverage_030_score_020 | 0.300 | 0.020 | 1.000 | 0.500 | 0.000 | 0.750 |
| coverage_040_score_020 | 0.400 | 0.020 | 1.000 | 1.000 | 0.000 | 1.000 |
| coverage_030_score_024 | 0.300 | 0.024 | 1.000 | 0.500 | 0.000 | 0.750 |

Seçim: `coverage_040_score_020`. Bu küçük dev set üzerinde yapılmış provisional bir threshold seçimidir; gold veya tamamen human-reviewed sonuç değildir.
