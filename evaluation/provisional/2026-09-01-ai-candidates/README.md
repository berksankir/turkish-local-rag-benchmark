# Provisional retrieval run — AI-generated candidates

Bu klasördeki dosyalar, 50 AI-generated candidate kaydı kullanılarak 1 Eylül
2026 tarihinde yapılan teknik pipeline koşusunun değiştirilmemiş orijinal
çıktılarıdır. Kayıtlar kullanıcı tarafından tek tek incelenip onaylanmamıştır.
Bu nedenle bu sonuçlar nihai gold benchmark değildir ve yalnızca retrieval
pipeline'ının teknik olarak çalıştığını gösteren provisional sonuçlardır.

`retrieval_benchmark.original.md` içindeki “insan onaylı gold set” ifadesi
yanlıştır. Dosya, mevcut çıktıları silmeme veya üzerine yazmama gereği nedeniyle
yalnızca provenance amacıyla orijinal haliyle korunmuştur; bu README o ifadeyi
geçersiz kılar.

Korunan orijinal SHA-256 değerleri:

- `provisional_gold.original.jsonl`: `0599ae0c761db73e7512f4cd4eef0a2ced2d258319f127076a0949096c9aa268`
- `retrieval_benchmark.original.csv`: `b66889c2d20085142bdd3b2e2562ff2c75538af3b531e1806be0152c2b51c041`
- `retrieval_benchmark.original.json`: `3d9f6e259fc526adaaa15f2fa32f03fa3663c6f3d584e861a413204290de53b5`
- `retrieval_benchmark.original.md`: `34a0046dd340d2ab9acff41cc67d8a3baa85202d05d4e2a0d437611d8d7ce824`

Nihai benchmark ancak `evaluation/review.csv` içindeki kayıtlar gerçek insan
incelemesinden geçtikten ve canonical gold set yalnızca `approved` kayıtlardan
yeniden oluşturulduktan sonra çalıştırılacaktır.
