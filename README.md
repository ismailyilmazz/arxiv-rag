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

Bölüm bölüm yazımla kabul testi (aynı konular, ölçüm):

| tür / dil | uzunluk | yazım isteği | kesilen bölüm | tekrar eden 8 kelimelik dizi | geçersiz atıf | token / süre |
|---|---|---|---|---|---|---|
| survey / en | 3.987 kelime | 8 | 0 | 0 | 0 | 20,5 bin / 4,7 dk |
| proposal / tr | 3.187 kelime | 10 | 0 | 0 | 0 | 32,4 bin / 7,7 dk |
| synthesis / en | 2.695 kelime | 7 | 0 | 0 | 0 | 20,2 bin / 4,7 dk |

Atıf ölçümü bölümleri ayırır: plan ve özet bölümleri (önerilen yöntem, değerlendirme planı, katkılar,
riskler, özet, sonuç) atıfsız olabilir; diğer bölümlerde her paragraf atıflı olmalıdır. Kaynakların
sınırlarından türetilen bölümlerde (açık problemler, çıkarımlar) her nokta kaynağına bağlanır.

Açık konular:
- İndirme şimdilik .md; PDF dışa aktarma Adım 6'dan sonra eklenecek.
- Genişlik: tarama türü sadece seçilen kaynakları derinlemesine işler ve bunu girişte açıkça belirtir
  (insan değerlendirmesi: "üç kilometre taşının iyi kurgulanmış incelemesi, ama alanın genişliğini
  kapsamıyor"). İleride arama hattıyla seçilen makalelere en yakın makaleler bulunup özet düzeyinde
  "bağlam kaynağı" olarak eklenecek: seçilenler derinlik, bağlam kaynakları genişlik sağlayacak.

## Adım 6a: Bağlam kaynakları ve yerel veri

**Bağlam kaynakları (genişlik):** Seçilen makaleler derinlik sağlar (tam metin, kart, kendi alt bölümü).
`core/related.py`, seçilen makalelerin her birinin vektörüyle anlamsal arama yapıp en yakın makaleleri bulur.
Bu makaleler yazıma sadece başlık, yıl ve özetin ilk iki cümlesiyle girer (kaynak başına yaklaşık 80 token,
hesap); ek yazım isteği ve kart üretimi yoktur. Giriş, arka plan, karşılaştırma, açık problemler gibi bölümlerde
"yalnızca özetinde yazanlar için" atıflanabilirler. Kaynakça iki gruptur: incelenen çalışmalar ve bağlam
çalışmaları. En fazla 10 bağlam kaynağı.

```
python -m scripts.generate_paper --ids 1701.06538 2101.03961 --auto-context 7 --type survey
python -m scripts.generate_paper --ids 1701.06538 2101.03961 --context 2006.16668 --type survey
```

**Yerel veri:** Yerelde vektör ve model yoktur. `notebooks/08_local_bundle.py` (Kaggle, CPU; girdiler 01, 04, 06)
30 binlik örneğin vektörlerini tam vektör dosyasından seçer, modellerle birlikte `local_bundle.zip` yapar.
Paket satır numarası değil makale kimliği taşır; yerelde içe aktarılırken kimlikler yerel veritabanının satır
numaralarına eşlenir:

```
python -m scripts.import_bundle local_bundle.zip
```

Sonuç: `data/vectors_dev/`, `data/models/` ve yerel veritabanında `term_df`.

## Adım 5b: Geniş akış (üç tür)

Az sayıda derin kaynakla yazılan metin "seçilmiş çalışmaların incelemesi" olarak kalıyordu (insan değerlendirmesi);
aynı sorun üç türde de vardı. Geniş akış, arama katmanını üretime bağlar ve üç türde de kullanılır. Türler sadece
temaların kullanımında ayrışır: literatür taramasında her tema bir bölümdür; araştırma önerisinde temalar
"İlgili Çalışmalar: {tema}" bölümleridir ve araştırma soruları bunların bıraktığı boşluklardan türetilir; sentez
raporunda temalar "Bulgular: {tema}" bölümleridir ve temalar arası karşılaştırma tablosu eklenir.

1. Sorgu genişletme: konudan 5 alt sorgu (LLM, 1 istek).
2. Aday toplama: her alt sorgu için arama akışının (anlamsal + BM25 + sıralayıcı + bekçi) ilk 20 sonucu ve tohum
   makalelere yakın makaleler, RRF ile birleştirilir; bekçinin reddettiği sorgular atlanır.
3. Eleme: en iyi 30 makale (tohumlar her zaman dahil).
4. Temalar: makale vektörleri K-Means ile 3-6 kümeye ayrılır (küme sayısı silhouette ile seçilir); LLM kümeleri
   adlandırır, konu dışı kümeleri atar ve sırayı belirler.
5. Derinlik: tohumlar ve her temanın merkezine en yakın makale tam metin kartı alır (en fazla 6); diğerleri
   özetin katkı cümleleriyle girer.
6. Yazım: giriş, arka plan, her tema için bir bölüm, karşılaştırma tablosu, açık problemler, sonuç.
7. Doğrulama: geçersiz atıf, atıfsız paragraf, isim-atıf uyuşmazlığı (bir çalışmanın adını anan cümle başka
   numara veriyorsa) ve sistemin iç dilinin metne sızması ölçülür.
8. Çıktı: numaralar ilk geçiş sırasına göre yeniden verilir; kaynakçada sadece atıflananlar, elenen ama
   atıflanmayan ilgili makaleler linkli "İleri okuma" listesinde.

Kapsama ölçümü: arXiv'deki gerçek bir survey hakem olarak kullanılır; kaynakçasındaki arXiv kimliklerinden
korpusumuzda olanların ne kadarının aday listesine girdiği, elemeden geçtiği ve atıflandığı ölçülür (hakem survey
aday listesinden çıkarılır). Tam korpusla Kaggle'da çalışır: `notebooks/09_broad_survey.py`.

```
python -m scripts.generate_broad --topic "Mixture of experts for large language models" --type survey \
    --seeds 1701.06538 2101.03961 2401.04088 --gold-survey 2407.06204
```

## Adım 5c: Kaynak kalitesi

İlk geniş çalıştırmada (Kaggle, tam korpus) hakem survey'lerin korpustaki kaynaklarının sadece yaklaşık %5'i aday
listesine girdi; arama, konuyu metinsel olarak en iyi anlatan yeni makaleleri buluyor, alanın en etkili
çalışmalarını değil. Ayrıca survey'ler malzemeyi işgal etti, yazım dili araştırma odağı sanıldı ve küçük temalar
oluştu (insan ve ölçüm değerlendirmesi). Düzeltmeler:

- **Atıf farkındalıklı seçim (Semantic Scholar):** tohumların (yoksa ilk 3 adayın) kaynakçasındaki, korpusta olan
  arXiv çalışmaları adaylara eklenir; adaylar `0.6 × arama skoru + 0.4 × yıllık atıf skoru` ile yeniden sıralanır
  (yıllık atıf, yeni makalelerin ezilmemesi için). Servise ulaşılamazsa üretim atıf sinyali olmadan sürer ve
  raporda belirtilir. `S2_API_KEY` isteğe bağlıdır.
- **Survey sınırı:** en fazla 2 survey malzemeye girer, derin okunmaz; fazlası "İleri okuma" listesine gider.
- **Küçük temalar:** 3 makaleden küçük kümeler anlamca en yakın kümeye katılır.
- **Prompt:** yazım dili araştırma odağını belirlemez; açıklar sadece malzemenin söylediğinden türetilir;
  istenirse `--focus` ile odak verilir.
- **Sayı denetimi:** tek bir kaynağa atıflanan cümlelerdeki sayıların o kaynağın malzemesinde geçip geçmediği
  ölçülür (uydurma göstergesi; yazıyla yazılmış oranları yakalamaz).

Kapsama ölçümü notu: hakem seti sadece arXiv linki olan kaynaklardan çıkarılır, bu yüzden mutlak oranlar düşük
görünür; anlamlı olan aynı hakemle önce/sonra karşılaştırmasıdır.

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
