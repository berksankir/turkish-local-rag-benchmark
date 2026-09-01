# Provisional retrieval run — AI-generated candidates

Bu klasördeki dosyalar, 50 AI-generated candidate kaydı kullanılarak 1 Eylül
2026 tarihinde yapılan teknik pipeline koşusunun arşivlenmiş çıktılarıdır.
Yanlış gold/human-reviewed etiketleri 2 Eylül 2026 tarihinde kullanıcı izniyle
JSON, CSV, JSONL dosya adı ve Markdown raporda düzeltilmiştir. Metrik değerleri,
query sonuçları ve provisional evaluation kayıtları değiştirilmemiştir.
Kayıtlar kullanıcı tarafından tek tek incelenip onaylanmamıştır.
Bu nedenle bu sonuçlar nihai gold benchmark değildir ve yalnızca retrieval
pipeline'ının teknik olarak çalıştığını gösteren provisional sonuçlardır.

JSON ve CSV artık `provisional_ai_candidates` dataset türünü ve
`human_reviewed: false` bilgisini kendi içlerinde taşır. Tarihsel config snapshot
içindeki `evaluation_gold` yolu aynen korunmuş, bunun insan onayı anlamına
gelmediği JSON içindeki `config_snapshot_note` alanında belirtilmiştir.

Provenance SHA-256 değerleri:

- `provisional_candidates.original.jsonl` (yalnızca yeniden adlandırıldı):
  `0599ae0c761db73e7512f4cd4eef0a2ced2d258319f127076a0949096c9aa268`
- `retrieval_benchmark.original.csv` etiket düzeltmesi öncesi:
  `b66889c2d20085142bdd3b2e2562ff2c75538af3b531e1806be0152c2b51c041`
- `retrieval_benchmark.original.csv` etiket düzeltmesi sonrası:
  `8c669bc677b633df64e03ac427ab66f96e30e9f0329be7b62f6331bb110cc969`
- `retrieval_benchmark.original.json` etiket düzeltmesi öncesi:
  `3d9f6e259fc526adaaa15f2fa32f03fa3663c6f3d584e861a413204290de53b5`
- `retrieval_benchmark.original.json` etiket düzeltmesi sonrası:
  `7e9fe306bf7a51ad5ebc8138306ce4334669093fbd1032561086a48c64481378`
- `retrieval_benchmark.original.md` düzeltme öncesi:
  `34a0046dd340d2ab9acff41cc67d8a3baa85202d05d4e2a0d437611d8d7ce824`
- `retrieval_benchmark.original.md` metodolojik düzeltme sonrası:
  `5ac8eec3203dee3876c0d6c7dca75e330e10ebb66e4bff3065e7032ca3740343`

Nihai benchmark ancak `evaluation/review.csv` içindeki kayıtlar gerçek insan
incelemesinden geçtikten ve canonical gold set yalnızca `approved` kayıtlardan
yeniden oluşturulduktan sonra çalıştırılacaktır.
