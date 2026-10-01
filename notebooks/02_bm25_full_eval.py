# %% [markdown]
# # Adım 2: BM25'i tam korpusta ölç (Kaggle)
#
# Hazırlık (sağ panel):
#   1. "Add Input" > "Your Work" > Adım 1 notebook'unu ekle (corpus.parquet buradan gelecek).
#   2. Settings > Internet: On  (GitHub'dan kodu çekmek için)
#   3. Settings > Accelerator: None
#
# Çıktılar (/kaggle/working altında):
#   papers_full.db                     tam korpusun veritabanı (Adım 3 ve 4'te tekrar kullanılacak)
#   arxiv-rag/data/eval_results/*.json ölçüm raporu

# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag"   # kendi repo adresinle değiştir

# %%
import glob

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q python-dotenv

corpus = glob.glob("/kaggle/input/**/corpus.parquet", recursive=True)
assert corpus, "corpus.parquet bulunamadı. Adım 1 notebook'unu Input olarak ekledin mi?"
CORPUS = corpus[0]
print(CORPUS)

# %%
# Tam korpusu veritabanına yükle. Notebook'un en uzun süren adımı bu.
!python -m scripts.load_sqlite {CORPUS} --db /kaggle/working/papers_full.db

# %%
!python -m scripts.evaluate --method bm25 --db /kaggle/working/papers_full.db
