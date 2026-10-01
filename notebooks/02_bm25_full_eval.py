# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"

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
!python -m scripts.load_sqlite {CORPUS} --db /kaggle/working/papers_full.db

# %%
!python -m scripts.evaluate --method bm25 --db /kaggle/working/papers_full.db
