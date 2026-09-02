# Faz 8.2 — Yerel üretici model seçimi

Araştırma tarihi: 2026-09-02

Hedef ortam: Windows, Intel i5-12450H, 8 GB RAM, yalnız CPU. Seçimden sonra
kullanıcı onayıyla yalnız seçilen model ve sabit CPU runtime indirilip doğrulanmış,
smoke sorguları ve gerçek AI-assisted silver benchmark çalıştırılmıştır.

## Karşılaştırma

| Aday | Parametre / quantization | Gerçek GGUF boyutu | Lisans ve kaynak | Türkçe / instruction / JSON kanıtı | Thinking | Tahmini LLM RAM ve CPU hızı |
|---|---|---:|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1,54B / Q4_K_M | 1.117.320.736 bayt (1,12 GB; 1,041 GiB) | Apache-2.0; model üreticisinin resmî GGUF repository'si | Resmî kart 29'dan fazla dil, gelişmiş instruction following ve özellikle JSON structured output bildiriyor. Listelenen örnek diller Türkçe'yi açıkça saymıyor; Türkçe yeterlilik yerelde ölçülmeden doğrulanmış kabul edilmeyecek. | Ayrı reasoning modu olmayan standart instruct model | Yaklaşık 1,5–2,0 GiB; yaklaşık 12–25 token/sn |
| Qwen3-1.7B | 1,7B / resmî repository'de yalnız Q8_0 | 1,83 GB | Apache-2.0; model üreticisinin resmî GGUF repository'si | Resmî kart 100'den fazla dil/diyalekt ve multilingual instruction following bildiriyor; Türkçe adı ayrıca verilmemiş. JSON biçimi prompt örneği var, fakat bu görevdeki schema başarısı ölçülmedi. | `/no_think` ile kapatılabiliyor | Yaklaşık 2,2–2,8 GiB; yaklaşık 6–15 token/sn |
| Gemma 3 1B IT | 1B / Q4_K_M topluluk yerine llama.cpp ekibinin GGUF dönüşümü | 806 MB | Gemma kullanım koşulları; ggml-org dönüşümü, temel model Google DeepMind | Google kartı eğitim verisinde 140'tan fazla dil ve question answering uygunluğu bildiriyor; Türkçe ve bu şemadaki JSON başarısı ayrıca kanıtlanmış değil. | Ayrı bir thinking anahtarı belgelenmiyor | Yaklaşık 1,2–1,7 GiB; yaklaşık 18–35 token/sn |

Tablodaki RAM ve token/sn değerleri seçim anındaki mühendislik tahminleridir;
benchmark sonucu değildir. LLM RAM tahmini model ağırlığı, 2048–3072
token KV cache ve runtime overhead'ini içerir. Qwen2.5 için mevcut retrieval,
embedding, opsiyonel reranker, Qdrant ve Python çalışma kümesi de hesaba katıldığında
toplam uygulama working set'inin yaklaşık 3,0–4,5 GiB aralığında kalması beklenir.
Gerçek benchmark'ta peak Python RSS 753.774.592, peak llama-server RSS
1.915.269.120 ve yaklaşık peak process-tree toplamı 2.669.043.712 bayt
(yaklaşık 2,49 GiB) ölçülmüştür.

## Seçim

Seçilen tek model **Qwen/Qwen2.5-1.5B-Instruct-GGUF Q4_K_M**'dir.

- Repository: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- Sabit revision: `91cad51170dc346986eccefdc2dd33a9da36ead9`
- Dosya: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- Quantization: `Q4_K_M`
- Boyut: `1.117.320.736` bayt
- SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`
- Lisans: Apache-2.0
- Runtime: `llama.cpp` b10621 (`0.3.0-dev`, commit `c1d0e7a00`) Windows CPU x64;
  arşiv 18.068.018 bayt, SHA-256
  `0e8b65e650e369f70f8307d890508886f171ef4fb00facccddd4a1b7ffdaca51`
- Local HTTP üzerinden tek ve tekrar kullanılan `llama-server` instance'ı

Qwen3 yalnız daha yeni olduğu için seçilmedi: resmî repository Q4_K_M dosyasını
silmiş ve yalnız 1,83 GB Q8_0 bırakmıştır; daha büyük ağırlık ve KV cache, 8 GB RAM'de
retrieval bileşenleriyle birlikte daha az güvenli marj sağlar. Gemma 3 1B daha küçük
olmasına rağmen Apache-2.0 yerine Gemma koşullarını taşır, resmî Google dosyaları
erişim koşulu kabulü ister ve bu görevdeki structured JSON davranışı resmî kartta
Qwen2.5 kadar açık belgelenmemiştir. Qwen2.5'in model üreticisi tarafından yayımlanan
Q4_K_M dosyası; boyut, lisans, JSON instruction iddiası ve yeniden üretilebilir dosya
hash'i bakımından daha dengeli seçimdir.

Seçim Türkçe yeterliliğini genel olarak kanıtlamaz. Gerçek smoke testinde iki
pipeline da context'e bağlı doğru kısa Türkçe cevap ve trusted citation üretmiş;
unanswerable smoke sorgusu model çağrılmadan abstain etmiştir. Silver benchmark'ta
generator yalnız bir kez başlatılmış, ancak her pipeline'da beş şema/parse hatası
`generator_invalid_json` olarak fail-closed abstention'a dönüşmüştür. Bu nedenle
structured output davranışı kullanılabilir fakat kusursuz değildir.

## Runtime ve structured output

`llama.cpp`, GGUF için yerel CPU inference ve Windows'ta WinGet/prebuilt kurulumunu
resmî olarak destekler. `llama-server` sürekli bir process olarak çalışabildiği için
modelin her sorguda yeniden yüklenmesini önler. Server, completion isteklerinde
`json_schema`/GBNF ile constrained generation destekler. Buna rağmen çıktı uygulama
katmanında tekrar parse ve validate edilecek; citation'lar modelden alınmayacaktır.

Planlanan pratik sınırlar: 2048–3072 context token, 128–192 output token, sabit seed
ve düşük varyanslı sampling. Qwen2.5 ayrı bir thinking/reasoning modu kullanmadığı
için kapatılacak düşünme modu yoktur.

## Birincil ve resmî kaynaklar

- [Qwen2.5 resmî model kartı](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [Seçilen Q4_K_M dosyasının boyutu ve SHA-256 değeri](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf)
- [Qwen2.5 sabit revision](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/commit/91cad51170dc346986eccefdc2dd33a9da36ead9)
- [Qwen3 resmî GGUF model kartı](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)
- [Qwen3 resmî Q8_0 dosyası](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/blob/90862c4b9d2787eaed51d12237eafdfe7c5f6077/Qwen3-1.7B-Q8_0.gguf)
- [Gemma 3 resmî Google model kartı ve kullanım koşulları](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf)
- [ggml-org Gemma 3 Q4_K_M dosyası](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/blob/main/gemma-3-1b-it-Q4_K_M.gguf)
- [llama.cpp resmî repository ve server](https://github.com/ggml-org/llama.cpp)
- [llama.cpp v0.3.0 ve sabit b10621 runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b10621)
- [b10621 Windows asset build provenance/attestation](https://github.com/ggml-org/llama.cpp/attestations/42818481)
- [llama.cpp JSON Schema/GBNF dokümantasyonu](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- [llama.cpp Windows kurulum dokümantasyonu](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)

## Phase gate

1 GB üzerindeki GGUF için açık kullanıcı onayı alınmış; seçilen tek model ve runtime
indirilerek hash'leri doğrulanmıştır. Model/runtime Git-ignore kapsamında kalır.
Generation sonuçları AI-assisted silver'dır, `human_reviewed=false` bilgisini taşır
ve gold ya da tamamen human-reviewed benchmark olarak yorumlanamaz.
