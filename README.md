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

## Ölçümler

Değerlendirme seti v1 (700 sorgu), tam korpus (3.182.775 makale):

| yöntem | grup | hit@1 | hit@10 | MRR@10 | ortanca gecikme |
|---|---|---|---|---|---|
| BM25 | title/en | 0,853 | 0,947 | 0,888 | 1.454 ms |
| BM25 | synthetic/en | 0,660 | 0,840 | 0,715 | |
| BM25 | synthetic/tr | 0,053 | 0,127 | 0,072 | |
