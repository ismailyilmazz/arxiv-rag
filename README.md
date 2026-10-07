# arXiv Araştırma Asistanı

Kullanıcı bir konu yazar (Türkçe veya İngilizce). Sistem arXiv'deki ilgili makaleleri bulur,
kullanıcı içlerinden seçim yapar, LLM seçilen makalelere atıf yapan özgün bir araştırma
kağıdı taslağı üretir. Makale linkleri arXiv'in özet sayfasına yönlendirir.

## Klasör yapısı

```
core/          ortak kod: ayarlar, arXiv kimlikleri, veritabanı, arama, metrikler
scripts/       komut satırından çalışan işler (yükleme, değerlendirme)
notebooks/     Kaggle'da çalışan ağır işler (korpus, tam korpus ölçümü)
eval/          değerlendirme seti ve ölçüm raporları
tests/         birim testleri
data/          veriler (repoya girmez)
```

## Kurulum (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Proje kökünde bir `.env` dosyası oluştur:

```
LLM_API_KEY=
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_MODEL_FAST=openai/gpt-oss-20b
HF_TOKEN=
HF_REPO_ID=
ARXIV_CONTACT_EMAIL=
```

## Adım 1: Korpus

1. Kaggle'da yeni bir notebook aç, "Add Input" ile `Cornell-University/arxiv` veri setini ekle.
2. `notebooks/01_build_corpus.py` içeriğini yapıştır ve çalıştır, ardından "Save Version" ile kaydet.
3. Çıktıdan `dev_sample.parquet` ve `stats.json` dosyalarını indirip `data/` klasörüne koy.
4. Yerel veritabanını oluştur:

```
python -m scripts.load_sqlite data/dev_sample.parquet
```

## Adım 2: Değerlendirme seti ve BM25

```
python -m scripts.build_eval_set
python -m scripts.evaluate --method bm25
```

Tam korpus ölçümü Kaggle'da yapılır: `notebooks/02_bm25_full_eval.py`.
Raporlar `eval/results/` klasörüne yazılır.

## Adım 3: Anlamsal arama

Model seçimi Kaggle'da GPU ile yapılır: `notebooks/03_model_selection.py`.
Seçilen modelle tam korpusun vektörleri iki GPU'da üretilir: `notebooks/04_full_embedding.py`.

```
python -m scripts.evaluate --method dense --vectors-dir <vektör klasörü>
python -m scripts.evaluate --method hybrid --vectors-dir <vektör klasörü>
python -m scripts.build_term_df
python -m scripts.evaluate --method bm25-gate
```

`bm25-gate`, aramayı sorgudaki en nadir iki kelimeden birini içeren makalelerle sınırlar,
sıralamayı bütün kelimelerle yapar. Kelimelerin ne kadar yaygın olduğu `term_df` tablosundan okunur.
Tam korpus ölçümü: `notebooks/05_bm25_gate.py`.

## Adım 4a: Eğitim verisi

Sıralayıcı ve bekçi için eğitim sorguları Groq ile üretilir. Makaleler sadece eğitim havuzundan
seçilir, değerlendirme setindeki makaleler hariç tutulur. Her makale için üç sorgu: İngilizce,
Türkçe ve Türkçe ile İngilizce terimlerin karıştığı "mühendis ağzı". Birden fazla model sırayla
kullanılır, bir modelin günlük kotası dolunca diğerleriyle devam edilir, kesilen iş kaldığı yerden sürer.

```
python -m scripts.gen_train_queries --n-papers 3 --n-offtopic 0
python -m scripts.gen_train_queries
```

Çıktılar `train/queries.jsonl` ve `train/offtopic.jsonl` (git'e girer). Modeller `.env` içindeki
`LLM_GEN_MODELS` satırından okunur.

## Adım 4b: Sınıflandırıcı, sıralayıcı ve bekçi

Kaggle'da çalışır: `notebooks/06_train_models.py` (girdiler: 04 vektörler, 05 term_df'li veritabanı).

- `scripts/train_field_classifier.py`: granite vektörleri üzerinde lojistik regresyon, yaklaşık 16 ana alan.
  Rakibi TF-IDF. Test: 2026 makaleleri ve değerlendirme sorguları (İngilizce, Türkçe ayrı).
- `scripts/build_features.py`: her sorgu için aday havuzu (anlamsal ilk 100 ∪ kapılı BM25 ilk 100)
  ve özellikler. Aynı kod (`core/features.py`) eğitimde, değerlendirmede ve sunucuda kullanılır.
- `scripts/train_ranker.py`: LightGBM LambdaRank. Rakibi aynı aday havuzunda tek başına anlamsal arama.
  Doğrulama için eğitim makalelerinin %15'i makale bazında ayrılır.
- `scripts/train_guard.py`: lojistik regresyon. Eşik, gerçek sorguların en fazla %5'ini reddedecek şekilde
  eğitim verisinden seçilir. Rakibi aynı kuralla en iyi anlamsal skora konan eşik.

Not: Nadir kelime kapısı artık sözlükte bulunan kelime sayısı 4'ten azsa kurulmuyor (Türkçe sorgulardaki
fazla sıkılığın düzeltmesi). `bm25-gate` sonucu 06 notebook'unda yeniden ölçülür.

## Adım 4c: Arama akışı

`core/pipeline.py` içindeki `SearchPipeline`, bir sorguyu uçtan uca işler: anlamsal arama ve kapılı BM25
ile aday havuzu, özellikler, sıralayıcı ve bekçi. Çıktı: ilk k makale, "kaynak yeterli mi" kararı ve
aşama süreleri. Tam korpus ölçümü (CPU): `notebooks/07_pipeline.py`.

```
python -m scripts.evaluate --method pipeline --vectors-dir <vektörler> --models-dir <modeller>
```

Kabul kriteri: canlı akışın değerlendirme sonucu, Adım 4b'deki çevrimdışı sıralayıcı sonucuyla aynı olmalı.

Sonuç (tam korpus, CPU, 770 sorgu): hit@1, hit@3 ve hit@10 bütün gruplarda çevrimdışı sonuçla aynı;
MRR'de en fazla 0,001 fark (06'da sorgular GPU'da fp16, burada CPU'da fp32 kodlandı).

Bekçinin kabul oranı: title/en %99,3, synthetic/en %99,3, manual/tr %92,9, synthetic/tr %88,0;
konu dışı sorularda İngilizce %2, Türkçe %6. Türkçe gerçek sorgular daha sık reddediliyor (açık konu).

Aşamaların ortanca süresi (ms, ikinci çalıştırma): kodlama 176, anlamsal arama 251, BM25 217, özellikler 80,
sıralayıcı 1,4, bekçi 0,9; toplam 734. Rapor: `eval/results/pipeline-granite-311m-384_3182775.json`.

## Adım 5: Araştırma kağıdı üretimi

Seçilen makalelerden (1-5) okuma kartları çıkarılır, LLM bu kartlardan atıflı bir metin yazar,
kod atıfları doğrular ve kaynakçayı makale bilgilerinden kendisi ekler.

- `core/fulltext.py`: önce arXiv HTML'i, yoksa PDF, o da yoksa özet. İstekler arası en az 3 saniye.
  Yerel veritabanında olmayan makalelerin bilgileri arXiv API'sinden çekilir.
- `core/cards.py`: makale başına İngilizce okuma kartı (problem, yöntem, deney kurulumu, bulgular, sınırlar;
  alan başına 3-8 cümle, metin ne kadar taşıyorsa; sayılar aynen), `data/cache.db` içinde önbellekli. Kart
  sürümü değişince eski kartlar yenilenir. `FULLTEXT_POLICY=cc` ile tam metin sadece CC lisanslı makalelerde
  kullanılır.
- `core/writer.py`: üç tür (`survey`, `proposal`, `synthesis`) ve iki dil (`en`, `tr`), **bölüm bölüm** yazılır.
  Her bölüme sadece ona lazım olan malzeme verilir: giriş ve arka plana kartların problem ve yöntem özeti,
  kaynak alt bölümlerine (her kaynak ayrı istek) kartla birlikte tam metinden yöntem ve sonuç parçaları,
  karşılaştırmaya bütün kartlar, sonuç ve özete yazılmış bölümlerin özeti. Özet en başta durur ama en son
  yazılır. Sabit uzunluk hedefi yok, dolgu yasak; her cümle özne ve yüklem içerir. Başlık ayrı bir istekle
  üretilir. Kesilen bölümler raporda işaretlenir.
- `core/llm.py`: model başına son 60 saniyenin token kullanımı sayılır; `LLM_TPM` (varsayılan 8000) aşılacaksa
  istek atılmadan beklenir. Ücretli katmanda bu değer yükseltilebilir.
- Tam metin `data/cache.db` içindeki `texts` tablosunda önbelleğe alınır, kullanıcıya hiçbir zaman gösterilmez.
- `core/citations.py`: geçersiz `[n]` numaralarını sayar ve temizler, atıf almayan kaynakları ve atıfsız
  paragrafları raporlar (alt başlıktaki ya da listeyi açan satırdaki atıf sayılır), kaynakçayı yazar.
- Modeller: kartlar `LLM_CARD_MODEL` (varsayılan gpt-oss-20b), yazım `LLM_WRITE_MODEL` (varsayılan gpt-oss-120b).

```
python -m scripts.generate_paper --ids 1701.06538 2101.03961 2401.04088 --type survey --lang en
```

Kabul kriteri: üç farklı konuda üretim, geçersiz atıf sıfır, kaynakça makale bilgileriyle birebir aynı.

İlk kabul testi (çıktılar `eval/generations/`):

| tür / dil | konu | geçersiz atıf | atıfsız paragraf | token (kart + yazım) | süre |
|---|---|---|---|---|---|
| survey / en | Mixture of Experts | 0 | 23/25 | 6.529 + 4.537 | 20 sn |
| proposal / tr | RAG | 0 | 10/13 | 6.293 + 4.596 | 30 sn |
| synthesis / en | Ricci flow (Perelman) | 0 | 2/13 | 8.497 + 3.217 | 60 sn |

Dokuz makalenin dokuzu da arXiv HTML'inden geldi (2002 tarihli makaleler dahil). Taramada kaynaklar
çoğunlukla adıyla anıldığı için paragraf bazında atıf ölçümü eklendi ve prompt güçlendirildi. Kartlar kısa
olduğu için uzun metinlerde tahmin ve dolgu görüldü; kartlar zenginleştirildi ve sabit uzunluk hedefi
kaldırıldı ("gerektiği kadar"). Tek istekte model kart malzemesini eksiksiz kullanıp yaklaşık 1.000-1.700
kelimede durdu (ölçüm); ortalama bir arXiv makalesi uzunluğu için bölüm bölüm yazıma geçildi.

## Ölçümler

Değerlendirme seti: 730 sorgu (300 başlık, 150+150 sentetik İngilizce/Türkçe, 30 elle yazılmış Türkçe,
100 konu dışı). Tam korpus: 3.182.775 makale. Sorgular CPU'da kodlandı. Ana metrik hit@10.

| yöntem | title/en | synthetic/en | synthetic/tr | manual/tr | manual/tr hit@1 | ortanca gecikme |
|---|---|---|---|---|---|---|
| BM25 | 0,947 | 0,840 | 0,127 | 0,200 | 0,067 | 1.397 ms |
| BM25 + nadir kelime kapısı | 0,950 | 0,840 | 0,080 | 0,200 | 0,067 | 127 ms |
| **Anlamsal (granite-311m-384)** | **0,990** | 0,893 | **0,407** | **0,433** | **0,200** | 497 ms |
| Hibrit (RRF) | 0,987 | **0,927** | 0,353 | 0,367 | 0,100 | 1.980 ms |
| Hibrit + kapı | 0,983 | 0,920 | 0,367 | 0,367 | 0,100 | 683 ms |
| Dil kuralı (TR → anlamsal, EN → hibrit) | 0,983 | 0,920 | 0,407 | 0,433 | 0,133 | 597 ms |

Bulgular:
- Anlamsal arama, gerçek Türkçe sorgularda BM25'in iki katından fazla isabet veriyor.
- Nadir kelime kapısı BM25'i 11 kat hızlandırıyor, İngilizce isabeti korunuyor.
- Elle yazılmış sorgularda BM25 zayıf: yöntem adı geçen sorgularda, o yöntemi kullanan sonraki
  makaleler orijinal makalenin önüne geçiyor.
- RRF iki listeye eşit güvendiği için Türkçede kaybettiriyor. Varsayılan yöntem anlamsal arama;
  sinyalleri birleştirmek Adım 4'teki sıralayıcının işi.

### Adım 4: Öğrenen katman (değerlendirme seti v2, 770 sorgu)

Alan sınıflandırıcısı (16 ana alan):

| model | 2026 makaleleri doğruluk | İngilizce sorgu | Türkçe sorgu |
|---|---|---|---|
| **granite vektörleri + lojistik regresyon** | 0,846 | **0,787** | **0,664** |
| TF-IDF + lojistik regresyon (SGD) | 0,866 | 0,731 | 0,473 |

Sıralayıcı (LightGBM LambdaRank) ve tek başına anlamsal arama, aynı aday havuzunda:

| grup | hit@10 | hit@3 | hit@1 | aday havuzu tavanı |
|---|---|---|---|---|
| title/en | 0,990 → **1,000** | 0,970 → **0,993** | 0,930 → **0,973** | 1,00 |
| synthetic/en | 0,893 → **0,940** | 0,820 → **0,867** | 0,700 → **0,747** | 1,00 |
| synthetic/tr | 0,413 → 0,413 | 0,247 → **0,293** | 0,173 → **0,193** | 0,64 |
| manual/tr (70) | 0,557 → **0,586** | 0,443 → **0,457** | 0,257 → **0,271** | 0,80 |

En etkili özellikler: anlamsal sıra, BM25 sırası, sorgunun Türkçe olması. Sıralayıcı hiçbir grupta
anlamsal aramanın gerisinde kalmıyor; RRF'nin Türkçedeki kaybı yok.

Bekçi (670 gerçek sorgu, 100 konu dışı soru; eşik eğitim verisinde %5 yanlış ret hedefiyle seçildi):

| yöntem | ROC-AUC | yanlış ret | konu dışı ret |
|---|---|---|---|
| **lojistik regresyon bekçi** | **0,994** | 3,9% | **96%** |
| en iyi anlamsal skora tek eşik | 0,973 | 3,4% | 78% |

Not: `manual/tr` grubu bu tabloda 70 sorgu; yukarıdaki 30 sorguluk tabloyla karşılaştırılmamalı.

### Embedding modeli seçimi

200 bin makalelik alt küme (değerlendirme setinin 300 hedef makalesi dahil), hit@10:

| model | title/en | synthetic/en | synthetic/tr | vektör üretme hızı (T4) |
|---|---|---|---|---|
| multilingual-e5-small | 0,990 | 0,947 | 0,553 | 396 makale/sn |
| granite-embedding-97m-multilingual-r2 | 1,000 | 0,980 | 0,427 | 350 makale/sn |
| **granite-embedding-311m-multilingual-r2 (384 boyut)** | **1,000** | **0,993** | **0,620** | 127 makale/sn |

Üreticinin kıyaslamasında e5-small'dan belirgin şekilde iyi görünen granite-97m, bu projenin
Türkçe sorgu ile İngilizce makale eşleşmesinde en kötü sonucu verdi. Seçim granite-311m-384.
