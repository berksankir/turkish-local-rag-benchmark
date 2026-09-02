# Turkish Local RAG Benchmark

Gerçek Türkçe PDF belgeleri üzerinde dense, BM25, RRF hybrid ve isteğe bağlı
reranking yaklaşımlarını karşılaştırmayı hedefleyen, tamamen yerel ve tekrar
üretilebilir bir RAG benchmark projesidir.

## Proje durumu

Repository; typed config, doğrulanmış kaynak manifesti, güvenli downloader,
sayfa-seviyesi extraction, yapı-farkındalıklı chunking, BM25, yerel E5 dense
retrieval, RRF hybrid retrieval ve optional cross-encoder reranking katmanlarını
içerir. Geliştirme bağımlılıkları kurulmuş, embedding ve reranker modelleri
yerelde doğrulanmıştır. Dokuz kaynak PDF ignore edilen yerel corpus klasörüne
indirilip hash'leri doğrulanmış ve gerçek sayfa-seviyesi extraction tamamlanmıştır;
gerçek page-safe chunking ve Qdrant local-mode dense corpus indeksi de tamamlanmıştır.
Elli AI-generated evaluation adayının kaynak span bütünlüğü otomatik olarak
doğrulanmış ve ayrı bir synthetic silver sete yansıtılmıştır; bu otomatik kontrol
insan onayı değildir. Daha düşük inceleme yükü için 20 kayıtlık deterministik audit
örneği hazırlanmıştır. Bu adaylarla yapılan eski retrieval koşusu yalnızca
provisional teknik sonuç olarak arşivlenmiştir ve nihai gold benchmark değildir.

Pipeline; sayfa sınırını aşmayan chunk'lar, güvenilir metadata, dense/BM25
retrieval, RRF fusion ve isteğe bağlı reranking kullanır. Faz 8'de deterministic
evidence gate, trusted citation katmanı ve CPU-only Qwen2.5 generation eklenmiş;
gerçek AI-assisted silver koşusu tamamlanmıştır. Bu sonuç gold veya tamamen
human-reviewed değildir.

## Config ve testler

Varsayılan başlangıç ayarları `config/default.toml` içindedir. Şema ve temel
tutarlılık kontrolleri `turkish_local_rag.config` modülünde tanımlanır.

Windows üzerinde izole geliştirme ortamı ve test komutu:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Testler ağ erişimi, PDF, Ollama veya model indirmesi gerektirmeyecek şekilde
tasarlanır.

## Kaynak manifesti ve downloader

Commit edilen `data/manifest.json`, Sabancı Üniversitesi yönetmelik sayfasında
yer alan dokuz PDF'nin güvenilir başlık ve kaynak URL metadata'sını içerir.
Downloader yolları ve ağ limitleri `config/default.toml` üzerinden çözülür.

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.download
.\.venv\Scripts\python.exe -m turkish_local_rag.download --source-id sabanci-ana-yonetmeligi
```

Bu komutlar gerçek PDF indirmesi başlatır; veri kullanım koşulları değerlendirilip
indirme bilinçli olarak istendiğinde çalıştırılmalıdır. Downloader HTTP/content,
redirect domain, bildirilen byte sayısı, `%PDF-` imzası ve `%%EOF` trailer
kontrollerinden geçen veriyi geçici dosyaya yazar ve atomik olarak taşır. Original
ve redirect sonrası final URL indirme metadata'sında kaydedilir. Mevcut
PDF ile yeni indirilen içeriğin SHA-256 değeri farklıysa mevcut dosya korunur ve
iki hash'i de içeren açık hata üretilir. İndirilen PDF'ler ile çalışma zamanı
metadata'sı Git tarafından ignore edilir.

2026-09-01 tarihli yerel corpus checkpoint'inde dokuz PDF'nin tamamı başarıyla
doğrulanmış, toplam boyut 3.193.812 bayt olmuş ve ikinci downloader çalıştırması
dokuz dosyayı da `unchanged` olarak raporlamıştır. Bu runtime dosyaları
repository'ye eklenmez; güncel hash'ler `data/downloads/` altındaki yerel
metadata kayıtlarında tutulur.

## Sayfa-seviyesi extraction

Extractor yalnızca downloader metadata'sındaki boyut ve SHA-256 ile yeniden
doğrulanan yerel PDF'leri işler. `page.get_text("blocks", sort=True)` kullanır;
çıktıdaki her JSONL satırı tek bir fiziksel PDF sayfasına karşılık gelir ve
manifestten gelen belge başlığı, sayfa numarası, kaynak URL, PDF URL ve PDF
hash'ini taşır. Çıktılar `artifacts/extracted/` altında atomik olarak yazılır ve
Git'e eklenmez.

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.extract
.\.venv\Scripts\python.exe -m turkish_local_rag.extract --source-id sabanci-ana-yonetmeligi
```

PyMuPDF `1.28.2`, GNU AGPL v3 veya Artifex ticari lisansı altında sunulur.
AGPL şartlarının bu projenin dağıtım biçimiyle uyumluluğu kullanıcı tarafından
değerlendirilmelidir. Bu aşamada OCR uygulanmaz; metin katmanı olmayan sayfalar
varsayılan olarak boş çıktı üretmeden atlanır ancak fiziksel sayfa numarası
değiştirilmez.

2026-09-01 extraction checkpoint'inde PDF'lerdeki 69 fiziksel sayfa yeniden
sayılmış; 68 metin sayfası için 68 JSONL kaydı ve toplam 259.126 karakter
üretilmiştir. Tümleşik Üretim Teknolojileri Merkezi Yönetmeliği'nin fiziksel 4.
sayfası boş olduğu için tek boş sayfa olarak korunmuş fakat JSONL kaydı
üretilmemiştir. İhale Yönetmeliği PDF'sindeki hatalı gömülü font eşlemelerinin
Türkçe `i` yerine verdiği iki yabancı Unicode glifi kanıtlanmış dönüşümle
onarılmış; sayfaların en az %80'inde aynı marj konumlarında yinelenen header,
footer ve fiziksel sayfa sayaçları çıkarımdan temizlenmiştir. OCR kullanılmamıştır.
PyMuPDF kaynak metni sayfa ve block düzeyinde `raw_text` alanında ayrıca korunur.
Kaynak text layer'da yalnız tam sözcük olarak görülen `ılgili`, kanıtlanmış yedi
konumda `(?<!\w)ılgili(?!\w)` kuralıyla `İlgili` olarak normalize edilir; diğer
noktasız `ı` karakterlerine veya doğru küçük `ilgili` sözcüklerine dokunulmaz.

## Yapı-farkındalıklı chunking

Chunker her fiziksel sayfayı bağımsız işler; hiçbir chunk sayfa sınırını aşmaz.
`MADDE`, bölüm ve numaralı paragraf başlangıçlarını yapısal sınırlar olarak
değerlendirir, `MADDE` başlangıçlarını yeni chunk'a taşır ve config'deki overlap
bütçesini yalnızca aynı sayfa içinde uygular. Çıktılar trusted belge/sayfa/URL ve
PDF hash metadata'sını taşır:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.chunk
.\.venv\Scripts\python.exe -m turkish_local_rag.chunk --source-id sabanci-ana-yonetmeligi
```

Bu aşamada bir model tokenizer'ı indirilmez. `estimated_tokens` değeri gerçek E5
token sayısı değildir; Unicode lexeme sayısı ile config'deki karakter/token
oranının büyük olanını kullanan deterministik bir güvenlik tahminidir. Kullanılan
yöntem her chunk'ın `token_count_method` alanında açıkça kaydedilir. Gerçek model
tokenizer'ı ayrıca doğrulanmalıdır.

2026-09-01 gerçek chunk checkpoint'inde 68 metin sayfasından 436 chunk üretilmiştir.
Yereldeki doğrulanmış E5 tokenizer `local_files_only=True`, `passage: ` öneki ve
özel tokenlarla bütün chunk'lara uygulanmış; gerçek token sayısı min/ortalama/max
`18 / 139,81 / 289` olmuş ve 512 sınırını aşan chunk bulunmamıştır. Deterministik
tahmin ortalaması `205,39`; gerçek eksi tahmini token farkı ortalama `-65,59`
olmuştur. 109 chunk pozitif overlap taşımış; tahmini overlap min/ortalama/max
`42 / 53,33 / 95` olarak ölçülmüştür. Boş, yalnız sayfa chrome'u içeren veya birden
fazla `MADDE` başlangıcı taşıyan chunk yoktur. Farklı belge ve sayfalardaki gerçek
ortak mevzuat metinlerinden kaynaklanan 9 exact-duplicate grup, trusted citation
metadata'sını kaybetmemek için korunmuştur.

## BM25 sparse retrieval ve RRF

Sparse retrieval, `rank-bm25` içindeki `BM25Okapi` sınıfını kullanır. Paket metin
ön işleme yapmadığı için corpus ve sorgular aynı proje-içi normalizer'dan geçer:
Unicode NFKC uygulanır, `I` → `ı` ve `İ` → `i` dönüşümleri Türkçe kurallarıyla
yapılır, diakritikler korunur ve noktalama token sınırı kabul edilir.

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.retrieve --mode bm25 --question "Burs başvurusu ne zaman yapılır?"
```

Retriever yalnızca doğrulanmış chunk metadata'sını döndürür; başlık, sayfa ve URL
sorgu metninden veya modelden üretilmez. BM25 indeksi bu küçük corpus aşamasında
bellekte oluşturulur ve pickle kullanılmaz. RRF fonksiyonu BM25 ile ilerideki
dense retriever'ın ham skorlarını karşılaştırmaz; yalnızca sıralama pozisyonlarını
config'deki `rank_constant` ile birleştirir.

`rank-bm25 0.2.2` Apache-2.0 lisanslıdır. Transitif NumPy bağımlılığı kendi
dağıtımındaki açık kaynak lisans bildirimleriyle sunulur.

## Dense retrieval ve Qdrant local mode

Dense katman `intfloat/multilingual-e5-small` modelini CPU üzerinde kullanır.
Model revision'ı ve safetensors SHA-256 değeri config'te sabittir; model yükleme
öncesinde yerel `download_manifest.json` içindeki model kimliği, revision, boyut
ve hash tekrar doğrulanır. Yükleme `local_files_only=True` ile yapılır. İndirme
ayrı ve açık bir komuttur:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.index --download-model-only
.\.venv\Scripts\python.exe -m turkish_local_rag.index
```

İlk komut yalnızca safetensors, tokenizer ve gerekli küçük config dosyalarını
indirir; repository'deki alternatif ONNX/OpenVINO ağırlıklarını indirmez. Passage
embedding'leri `passage: `, sorgular `query: ` prefix'iyle ve normalize edilerek
üretilir. Embedding boyutu 384, maksimum sequence length 512 ve varsayılan CPU
batch size 8'dir.

Qdrant, `qdrant-client` disk-persistent local mode ile `indexes/qdrant/` altında
çalışır; Docker, server veya cloud bağlantısı yoktur. Mevcut collection ancak
`--rebuild` açıkça verilirse değiştirilir. Qdrant payload metadata'sı her sorguda
trusted chunk JSONL kaydıyla karşılaştırılır.

2026-09-01 gerçek dense index checkpoint'inde 436 chunk, 384 boyutlu normalize
vektörlerle `chunks_e5_small_v1` collection'ına yazılmıştır. Tüm point ID'leri,
payload'lar ve vektörler üzerinde hesaplanan mantıksal SHA-256 fingerprint
`ed36a52e3d0d39d2aee348e4d19c4834a25b6a023b367ce2a8bcd9f9a0c44566` olmuş;
explicit `--rebuild` sonrasında aynı fingerprint yeniden elde edilmiştir. Final
index boyutu 2.495.132 bayttır. İlk oluşturma 77,549 saniye; ölçümlü rebuild
40,430 saniye sürmüş ve Windows process sayacında yaklaşık 971.206.656 bayt
(926 MiB) peak working set görülmüştür. Bu süre ve bellek değerleri yalnız bu
makinedeki smoke ölçümüdür, benchmark sonucu değildir.

Altı dense smoke sorgusunda trusted `document_id`, fiziksel sayfa, source URL,
PDF URL ve PDF SHA-256 payload'ları kaynak chunk kayıtlarıyla eşleşmiştir. İlk
ısınma sorgusu 206,264 ms, sonraki beş sorgu 18,238–21,185 ms sürmüştür. Mütevelli
Heyet, öğrenim ücretleri, ihale komisyonları, doktora tez izleme komitesi, İngilizce
muafiyet ve Veri Analitiği Merkezi sorgularının top-1 sonuçları ilgili gerçek
belge ve sayfalardan gelmiştir; bunlar gold evaluation olarak kullanılmaz.

İndeks oluşturulduktan sonra aynı sorgu arayüzü dense veya RRF hybrid modunda
çalıştırılabilir. Varsayılan mod `hybrid`'dir; BM25 ve cosine skorları doğrudan
toplanmaz, config'teki aday sayıları ve `rank_constant` kullanılarak sıra
pozisyonları birleştirilir:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.retrieve --mode dense --question "Burs başvurusu ne zaman yapılır?"
.\.venv\Scripts\python.exe -m turkish_local_rag.retrieve --mode hybrid --question "Burs başvurusu ne zaman yapılır?"
```

Model MIT; Sentence Transformers, Hugging Face Hub ve Qdrant Client Apache-2.0
lisanslıdır. PyTorch ve transitif bağımlılıklar kendi dağıtımlarındaki açık
kaynak lisans bildirimleriyle sunulur.

## Optional cross-encoder reranking

Reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` modelini yalnızca RRF
sonrasındaki sınırlı aday havuzuna uygular. Revision ve safetensors SHA-256
config'te sabittir ve yerel indirme manifesti model yüklenmeden önce doğrulanır;
indirme yalnızca gerekli PyTorch safetensors, tokenizer ve config dosyalarını
alır:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.index --download-reranker-only
.\.venv\Scripts\python.exe -m turkish_local_rag.retrieve --mode hybrid-reranked --question "Burs başvurusu ne zaman yapılır?"
```

Model CPU, `local_files_only=True`, maksimum 512 token ve batch size 4 ile
çalıştırılır. Raw cross-encoder logit'leri yalnızca adayları yeniden sıralamak
için kullanılır; citation alanları trusted chunk metadata'sından gelir. Model
Apache-2.0 lisanslıdır. mMARCO model kartının eğitim dili listesinde Türkçe
bulunmadığından bu bileşen açıkça zero-shot deneydir. Fayda sağladığı
varsayılmaz. Silver değerlendirme yalnız provisional teknik karşılaştırma sağlar;
nihai karar gerçek kullanıcı onayından geçen gold değerlendirmeye bırakılır.

Rerank edilecek aday sayısı `rerank_top_n`, inference batch büyüklüğü `batch_size`
ve PyTorch CPU thread sayısı `cpu_threads` olarak config'te tutulur. Model instance'ı
sorgular arasında yeniden kullanılır. Varsayılan hızlı mod `hybrid_rrf`, daha yüksek
latency kabul edildiğinde kullanılabilen opsiyonel mod `hybrid_reranked`tır.

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.profile_reranker --config config\default.toml
```

2026-09-02 tarihli Faz 8.1 profili yalnız silver `dev` split üzerinde, top‑20/top‑10,
batch 4/2 ve CPU thread 4/2 için dört küçük varyantla çalıştırılmıştır. Test split
profiling veya ayar seçimi için kullanılmamıştır. Embedding modeli 3.912,677 ms,
reranker 3.595,617 ms'de yüklenmiş; cold `hybrid_rrf` 327,723 ms, cold yalnız
reranking 1.803,151 ms ölçülmüştür. Warm `hybrid_rrf` ortalaması 23,849 ms iken
configured top‑20 reranking-only ortalaması 963,489 ms ve toplamı 1.017,725 ms'dir.
Tek reranker instance'ı tüm sorgu ve varyantlarda yeniden kullanılmıştır. Peak
process working set yaklaşık 1.368.723.456 bayttır.

Dev kalitesinde reranking, R@5 ve Doc@1'i 0,125 artırırken R@1 ve Page@1'i 0,125,
MRR'ı 0,0554 düşürmüştür. Bu nedenle hızlı varsayılan `hybrid_rrf` olarak kalır;
reranker yalnız opsiyonel kalite/karşılaştırma modudur. Ayrıntılı JSON, CSV ve
Markdown profilleri `evaluation/results/silver/reranker_profile.*` dosyalarındadır.

## Faz 8 yerel üretici model seçimi

Resmî Qwen2.5 1.5B Q4_K_M, Qwen3 1.7B Q8_0 ve Gemma 3 1B seçenekleri Windows,
8 GB RAM ve CPU-only hedefi için karşılaştırılmıştır. Qwen2.5'in resmî Q4_K_M
dosyası; 1,12 GB boyutu, Apache-2.0 lisansı, model kartındaki instruction/JSON
desteği ve sabit dosya SHA-256 değeri nedeniyle seçilmiştir. Model kartının örnek
dil listesi Türkçe'yi açıkça saymadığından Türkçe kalitesi gerçek yerel test
öncesinde doğrulanmış kabul edilmez.

Kullanıcı onayından sonra yalnız seçilen `qwen2.5-1.5b-instruct-q4_k_m.gguf`
dosyası indirilmiş; 1.117.320.736 bayt boyut ve sabit SHA-256 doğrulanmıştır.
`llama.cpp` b10621 Windows CPU runtime arşivi 18.068.018 bayttır ve sabit SHA-256
ile doğrulanır. Karşılaştırma, revision, hash, lisans ve resmî kaynaklar
`docs/phase8_model_selection.md` içinde kayıtlıdır. Model ağırlıkları `models/`,
runtime dosyaları `runtime/` altında ignore edilir ve Git'e eklenmez.

## Grounded generation, evidence gate ve citation

Sorgu arayüzü varsayılan hızlı `hybrid_rrf` ve opsiyonel `hybrid_reranked`
pipeline'larını destekler:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.query --question "Üniversitenin en yüksek karar organı hangisidir?" --pipeline hybrid_rrf
.\.venv\Scripts\python.exe -m turkish_local_rag.query --question "Üniversitenin en yüksek karar organı hangisidir?" --pipeline hybrid_reranked
```

Generator yalnız retrieved context'i görür; kısa Türkçe cevap, sabit seed ve
deterministic sampling kullanır. Çıktı parse edilip katı JSON şemasına göre
doğrulanır. Citation'ı LLM yazmaz: uygulama seçilen dahili context kimliğini trusted
chunk metadata'sındaki belge, başlık, fiziksel sayfa, URL'ler ve `chunk_id` ile
eşler. Uydurulmuş veya metadata'sı eksik citation reddedilir. Aynı `llama-server`
instance'ı sorgular arasında yeniden kullanılır.

Evidence gate, dev split üzerinde seçilen `query_coverage >= 0,40` ve
`top_rrf_score >= 0,020` eşiklerini kullanır. Kanıt zayıfsa model çağrılmaz ve aynı
versioned JSON şemasında `Yeterli kanıt bulunamadı.` yanıtı döner. Successful ve
abstain çıktılar; retrieval, reranking, generation ve total latency ile embedding,
reranker ve generator metadata'sını taşır. Threshold seçimi için test split
kullanılmamıştır.

Gerçek smoke sorgusunda her iki pipeline da “Mütevelli Heyet” cevabını trusted
`sabanci-ana-yonetmeligi:p2:c5` citation'ıyla üretmiştir. Unanswerable smoke sorgusu
gate'te model çağrılmadan durmuş, citation üretmemiştir. Gerçek 50 kayıt × iki
pipeline koşusunda generator yalnız bir kez başlatılmıştır.

2026-09-02 AI-assisted silver generation sonuçları:

| Pipeline | R@1/R@3/R@5 | MRR | Citation | Coverage | Correct abstain | False abstain | Token F1 | Key facts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_rrf | 0,725 / 0,925 / 0,950 | 0,833 | 0,676 | 0,925 | 0,700 | 0,075 | 0,420 | 0,390 |
| hybrid_reranked | 0,775 / 0,950 / 1,000 | 0,868 | 0,686 | 0,875 | 0,500 | 0,125 | 0,406 | 0,397 |

| Pipeline | Stage | Ortalama ms | p50 ms | p95 ms |
|---|---|---:|---:|---:|
| hybrid_rrf | retrieval / reranking / generation / total | 114,7 / 0,0 / 15.245,7 / 15.368,3 | 121,9 / 0,0 / 15.154,4 / 15.281,3 | 173,5 / 0,0 / 28.507,5 / 28.558,6 |
| hybrid_reranked | retrieval / reranking / generation / total | 65,0 / 1.809,8 / 6.864,8 / 8.959,1 | 43,2 / 1.389,3 / 4.129,0 / 6.155,0 | 164,7 / 3.355,3 / 21.427,7 / 24.578,0 |

Benchmark 1.216,431 saniye sürmüş; model initialization 9,913 saniye olmuştur.
Peak Python RSS 753.774.592, peak llama-server RSS 1.915.269.120 ve yaklaşık peak
process-tree toplamı 2.669.043.712 bayttır (yaklaşık 2,49 GiB). Pipeline'ların
ardışık çalıştırılması, response uzunluğu ve cache ısınması generation sürelerini
etkilediğinden reranked hattın daha düşük generation ortalaması doğrudan reranker
hız kazanımı olarak yorumlanmamalıdır.

RRF genel correct-abstention oranı 0,700; reranked oranı 0,500'dür. Her pipeline'da
beş çıktı `generator_invalid_json` nedeniyle güvenli abstention'a düşmüştür.
Bu sınırlamalar sonuçlarda korunur; test split'e bakılarak eşik veya pipeline
değiştirilmemiştir. Ayrıntılı, networksüz tekrar puanlanabilir JSON/CSV/Markdown
raporları `evaluation/results/silver/generation_benchmark.*` altındadır.

## Evaluation adayları

`evaluation/candidates.jsonl`, değiştirilmeden korunan 40 cevaplanabilir ve 10
cevaplanamaz AI-generated aday içerir. Otomatik kontrol, cevaplanabilir kayıtların
exact source span'lerinin belirtilen extracted belge ve fiziksel sayfada birebir
bulunduğunu doğrular; bu kontrol insan onayı değildir. Cevaplanamaz kayıtların
gerçekten corpus dışı olup olmadığı da insan tarafından incelenmelidir.

`evaluation/silver.jsonl`, 50 adayın tamamını kimlikleri ve içerikleri değişmeden
içeren synthetic silver settir. Answerable kayıtların span/page bütünlüğü otomatik
doğrulanmıştır; unanswerable kayıtların corpus dışı olduğu otomatik olarak kabul
edilmez. Silver hiçbir yerde human-reviewed veya gold olarak sunulmamalıdır.

İnceleme yükünü azaltan `evaluation/silver_audit.csv`, 10 unanswerable kaydın
tamamını ve dokuz belgenin her birini temsil edecek şekilde seçilmiş 10 answerable
kaydı içerir. Ek answerable kota, en çok adayı bulunan belgeye deterministik olarak
verilir; seçim retrieval sonuçlarından türetilmez. Yirmi kaydın tamamı kullanıcı
tarafından kontrol edilmiş ve `berksankir` reviewer kimliğiyle `approved` olarak
işaretlenmiştir; silver set bu sınırlı anlamda “human-audited sample”dır, ancak
“human-reviewed gold” değildir.

Tam gold incelemesi için `evaluation/review.csv` içindeki 50 kayıt korunur ve
başlangıçta `pending` kalır. Her iki CSV'de de yalnızca `review_status`,
`review_notes`, `reviewer` ve `reviewed_at_utc` alanları düzenlenmelidir; candidate
alanları ve `proposed_split` değiştirilirse doğrulayıcı kaydı reddeder. Geçerli
durumlar `pending`, `approved`, `needs_changes` ve `rejected` değerleridir.
`pending` dışındaki her karar reviewer ve `Z` ile biten ISO-8601 UTC zamanı
gerektirir; `needs_changes` ve `rejected` kararlarında review notes zorunludur.
Kaynak span'deki satır sonları CSV'de okunabilirlik için `\n` olarak gösterilir.

Review dosyasını ve otomatik span bütünlüğünü kontrol etmek için:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.review --config config\default.toml validate
```

Mevcut artifact'lar sessizce ezilmez. Silver'ı yeniden üretmek için açık
overwrite gerekir. Audit oluşturma komutu yalnızca dosya henüz yokken kullanılır;
var olan insan notlarını ezmez:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.review --config config\default.toml build-silver --overwrite
.\.venv\Scripts\python.exe -m turkish_local_rag.review --config config\default.toml prepare-silver-audit
.\.venv\Scripts\python.exe -m turkish_local_rag.review --config config\default.toml validate-silver-audit
```

Gold builder yalnızca `approved` kayıtları aktarır; diğer üç durum gold dışında
kalır. Kullanıcı review'u tamamlandıktan sonra çalıştırılacak komut:

```powershell
.\.venv\Scripts\python.exe -m turkish_local_rag.review --config config\default.toml build-gold
```

Final evaluator, bütün kayıtlar insan tarafından `approved` veya `rejected`
olarak sonuçlandırılmadan çalışmayı reddeder. `rejected` kayıtlar gold'a
girmez; gold benchmark yalnızca `approved` kayıtları kullanır. Dataset seçimi
açıkça yapılır ve sonuçlar birbirinden ayrı klasörlere yazılır:

```powershell
# Provisional synthetic silver benchmark
.\.venv\Scripts\python.exe -m turkish_local_rag.evaluate --config config\default.toml --dataset silver

# Nihai human-reviewed gold benchmark; review tamamlanana kadar bloke
.\.venv\Scripts\python.exe -m turkish_local_rag.evaluate --config config\default.toml --dataset gold
```

Silver çıktıları `evaluation/results/silver/`, gold çıktıları
`evaluation/results/gold/` altında üretilir. Silver JSON/CSV/Markdown çıktıları
dataset türünü ve audit durumlarını taşır; otomatik doğrulama insan onayı gibi
sunulmaz.

2026-09-02 canonical silver koşusu 50 soru ve dört pipeline üzerinde tamamlanmıştır.
JSON raporu `kind=\"silver\"`, `human_reviewed=false`, 20 provenance sahibi audit
kararı (`berksankir`, 20 `approved`) ve 436 chunk bilgisini taşır. Benchmark süresi
232,319 saniye, yaklaşık peak process working set 1.284.526.080 bayttır. Pipeline
seçimi yalnız sekiz answerable kayıt içeren `dev` split sonuçlarıyla yapılmıştır:

| Pipeline | dev R@1 | dev R@3 | dev R@5 | dev MRR | dev Page@1 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0,5000 | 0,8750 | 0,8750 | 0,6875 | 0,5000 | 90,173 |
| bm25 | 0,6250 | 1,0000 | 1,0000 | 0,8125 | 0,6250 | 3,198 |
| hybrid_rrf | 0,7500 | 0,8750 | 0,8750 | 0,8304 | 0,7500 | 91,099 |
| hybrid_reranked | 0,6250 | 0,8750 | 1,0000 | 0,7750 | 0,6250 | 4408,911 |

Bu dev karşılaştırmasına göre Faz 8 için provisional retrieval pipeline'ı
`hybrid_rrf` seçilmiştir. `test` split yalnız dokunulmamış holdout sonucu olarak
raporlanır; threshold, pipeline seçimi veya başka bir ayar için kullanılmamıştır.
Bu koşu human-reviewed gold benchmark değildir.

AI adaylarıyla yapılan eski teknik koşunun metrikleri korunmuş, metodolojik
etiketleri düzeltilmiş çıktıları
`evaluation/provisional/2026-09-01-ai-candidates/` altındadır. Bunlar pipeline'ın
çalıştığını gösterir; nihai benchmark sonucu olarak kullanılamaz.

## Veri ve kullanım uyarıları

- Kaynak PDF'ler repository'ye commit edilmez; yerel `data/pdfs/` klasörü Git
  tarafından ignore edilir.
- Bir belgenin kamuya açık olması, yeniden dağıtım izni verildiği anlamına
  gelmez. Kullanıcılar kaynakların kullanım koşullarını ayrıca değerlendirmelidir.
- Harici kaynaklar zaman içinde değişebilir veya kaldırılabilir. Downloader
  çalıştırmaları SHA-256 değişikliklerini sessizce üzerine yazmak yerine raporlar.
- Bu proje hukuki veya akademik danışmanlık sistemi değildir.
- PDF çıkarımı için değerlendirilen PyMuPDF, AGPL-3.0 veya ticari lisans altında
  sunulur. Bağımlılık eklenmeden önce lisans uyumluluğu ayrıca değerlendirilmelidir.
