# Kaggle notebook'ları

Ağır işler (tam korpus, GPU ile vektör üretimi, model eğitimi, tam korpusla belge üretimi) Kaggle'da çalışır.
Her dosya Kaggle'da tek bir hücreye yapıştırılıp **Save & Run All** ile çalıştırılır; repoyu kendisi klonlar.
Bir notebook'un çıktısı diğerine **Add Input** ile girdi olarak verilir. 04 ve 06'nın kaydedilmiş sürümleri
silinmemelidir: tam korpus vektörleri ve eğitilmiş modeller sadece oradadır.

| # | Dosya | Girdi | Donanım | Ne yapar | Önemli çıktı |
|---|---|---|---|---|---|
| 01 | `01_build_corpus.py` | Cornell arXiv veri seti | CPU | Tüm arXiv metadatasını okur, temizler | `corpus.parquet`, `dev_sample.parquet` |
| 02 | `02_bm25_full_eval.py` | 01 | CPU | BM25'i tam korpusta ölçer | BM25 raporu |
| 03 | `03_model_selection.py` | 01 | GPU T4 | 3 embedding modelini 200 binlik alt kümede karşılaştırır | model seçim raporu |
| 04 | `04_full_embedding.py` | 03 | GPU T4 x2 | Tam korpusu iki GPU'da parçalı kodlar | `papers_full.db`, `vectors_full/` |
| 05 | `05_bm25_gate.py` | 04 | CPU | `term_df` tablosunu kurar, nadir kelime kapısını ölçer | `term_df`'li veritabanı |
| 06 | `06_train_models.py` | 04 + 05 | GPU T4 x2 | Alan sınıflandırıcısı, sıralayıcı ve bekçiyi eğitir | `models/` |
| 07 | `07_pipeline.py` | 04 + 06 | CPU | Canlı arama akışını çevrimdışı sonuçla karşılaştırır | akış raporu |
| 08 | `08_local_bundle.py` | 01 + 04 + 06 | CPU | 30 binlik örneğin vektörleri ve modelleri | `local_bundle.zip` |
| 09 | `09_broad_survey.py` | 04 + 06 | CPU, internet, Secrets | Geniş akışla belge üretimi ya da kuru çalıştırma (`DRY_RUN`), hakem survey'e karşı kapsama | `generations/` (isteğe bağlı otomatik push) |

| 10 | `10_publish_data.py` | 04 + 06 | CPU, internet, Secrets | Sunucu verisini Hugging Face veri deposuna yükler, model sürümünü ve boyutları doğrular | `ismailylmz283/arxiv-rag-data` |

Notebook 10 için Kaggle **Secrets**: `HF_TOKEN` (Hugging Face yazma token'ı).

Notebook 09 için Kaggle **Secrets**: `LLM_API_KEY`, `ARXIV_CONTACT_EMAIL` (zorunlu); `GITHUB_TOKEN` (sonuçları repoya
göndermek için, isteğe bağlı); `S2_API_KEY` (isteğe bağlı).
