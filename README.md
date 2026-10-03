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

## Ölçümler

Değerlendirme seti v1 (700 sorgu), tam korpus (3.182.775 makale):

| yöntem | grup | hit@1 | hit@10 | MRR@10 | ortanca gecikme |
|---|---|---|---|---|---|
| BM25 | title/en | 0,853 | 0,947 | 0,888 | 1.454 ms |
| BM25 | synthetic/en | 0,660 | 0,840 | 0,715 | |
| BM25 | synthetic/tr | 0,053 | 0,127 | 0,072 | |
| Anlamsal (granite-311m-384) | title/en | 0,930 | 0,990 | 0,952 | 245 ms |
| Anlamsal (granite-311m-384) | synthetic/en | 0,700 | 0,893 | 0,762 | |
| Anlamsal (granite-311m-384) | synthetic/tr | 0,173 | 0,413 | 0,232 | |
| Hibrit (RRF) | title/en | 0,910 | 0,987 | 0,940 | 1.528 ms |
| Hibrit (RRF) | synthetic/en | 0,680 | 0,927 | 0,769 | |
| Hibrit (RRF) | synthetic/tr | 0,080 | 0,353 | 0,161 | |

Anlamsal arama Türkçede BM25'in 3,3 katı isabet veriyor. Hibrit arama İngilizcede kazandırıyor ama
Türkçede kaybettiriyor: RRF iki sıralamaya eşit güveniyor, BM25'in Türkçe sıralaması ise gürültü.

### Embedding modeli seçimi

200 bin makalelik alt küme (değerlendirme setinin 300 hedef makalesi dahil), hit@10:

| model | title/en | synthetic/en | synthetic/tr | vektör üretme hızı (T4) |
|---|---|---|---|---|
| multilingual-e5-small | 0,990 | 0,947 | 0,553 | 396 makale/sn |
| granite-embedding-97m-multilingual-r2 | 1,000 | 0,980 | 0,427 | 350 makale/sn |
| **granite-embedding-311m-multilingual-r2 (384 boyut)** | **1,000** | **0,993** | **0,620** | 127 makale/sn |

Üreticinin kıyaslamasında e5-small'dan belirgin şekilde iyi görünen granite-97m, bu projenin
Türkçe sorgu ile İngilizce makale eşleşmesinde en kötü sonucu verdi. Seçim granite-311m-384.
