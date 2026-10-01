# arXiv Araştırma Asistanı

Kullanıcı bir konu yazar (Türkçe veya İngilizce). Sistem arXiv'deki ilgili makaleleri bulur,
kullanıcı içlerinden seçim yapar, LLM seçilen makalelere atıf yapan özgün bir araştırma
kağıdı taslağı üretir. Makale linkleri arXiv'in özet sayfasına yönlendirir.

## Klasör yapısı

```
core/         ortak kod: ayarlar, arXiv kimlikleri, veritabanı
scripts/      komut satırından çalışan işler (yükleme, değerlendirme)
notebooks/    Kaggle'da çalışan ağır işler (korpus, embedding)
eval/         değerlendirme seti (git'e girer, ölçümler hep aynı sete göre yapılır)
tests/        birim testleri
data/         veriler (repoya girmez)
legacy/       ders projesinin ilk hali (sadece referans için)
```

## Kurulum (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
pytest
```

## Adım 1: Korpus

1. Kaggle'da yeni bir Notebook aç, sağ panelden "Add Input" ile `Cornell-University/arxiv` veri setini ekle.
2. `notebooks/01_build_corpus.py` içeriğini hücrelere yapıştır (`# %%` satırları hücre sınırıdır) ve çalıştır.
3. Çıktı panelinden `dev_sample.parquet` ve `stats.json` dosyalarını indirip `data/` klasörüne koy.
4. Yerelde depoyu oluştur:

```
python -m scripts.load_sqlite data/dev_sample.parquet
```

`corpus.parquet` tüm arXiv'i içerdiği için büyük, şimdilik Kaggle'da kalıyor. Sunucuya Adım 7'de taşınacak.

## Adım 2: Değerlendirme seti ve BM25

```
pip install -r requirements.txt
pytest
python -m scripts.build_eval_set
python -m scripts.evaluate --method bm25
```

Değerlendirme seti `eval/queries.jsonl` dosyasına yazılır ve git'e girer.
Resmi ölçüm Kaggle'da tam korpusla yapılır: `notebooks/02_bm25_full_eval.py`.
