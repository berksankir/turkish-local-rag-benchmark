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
gerçek page-safe chunking de tamamlanmıştır. Corpus indeksi ve benchmark deneyleri
henüz çalıştırılmamıştır. Dolayısıyla yayımlanmış bir benchmark sonucu yoktur.

Pipeline; sayfa sınırını aşmayan chunk'lar, güvenilir metadata, dense/BM25
retrieval, RRF fusion ve isteğe bağlı reranking kullanır. Deterministic evidence
gate, citation uygulama katmanı, generation ve evaluation sonraki aşamalardır.

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
`18 / 139,84 / 289` olmuş ve 512 sınırını aşan chunk bulunmamıştır. Deterministik
tahmin ortalaması `205,39`; gerçek eksi tahmini token farkı ortalama `-65,56`
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
varsayılmaz; karar aynı insan-onaylı gold sette yapılacak retrieval evaluation'a
bırakılır.

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
